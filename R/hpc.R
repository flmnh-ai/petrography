# ============================================================================
# HPC Helper Functions
# ProcessX + SSH multiplexing for fast, authenticated connections
# ============================================================================

#' Authenticate with HPC using SSH multiplexing (handles Duo MFA)
#' @param hpc_host Hostname (e.g., 'hpg')
#' @param hpc_user Optional username
#' @return SSH target string for subsequent connections
#' @export
hpc_authenticate <- function(hpc_host = "hpg", hpc_user = NULL) {
  target <- if (!is.null(hpc_user) && nzchar(hpc_user)) paste0(hpc_user, "@", hpc_host) else hpc_host

  cli::cli_h2("HPC Authentication with SSH Multiplexing")
  cli::cli_alert_info("Connecting to: {.strong {target}}")

  # Check if master connection already exists
  check_result <- processx::run("ssh", c("-O", "check", target), timeout = 5, error_on_status = FALSE)
  if (check_result$status == 0) {
    cli::cli_alert_success("SSH master connection already active for {.strong {target}}")
    return(invisible(target))
  }

  # Start master connection (handles Duo interactively)
  cli::cli_alert_info("Starting SSH master connection (Duo authentication required)...")
  cli::cli_alert_warning("Please respond to any authentication prompts below:")

  # Start master: -M=master, -N=no command, -f=background
  processx::run("ssh", c("-MNf", target), timeout = 60, error_on_status = TRUE)

  # Wait a moment for connection to establish
  Sys.sleep(2)

  # Verify master connection works
  processx::run("ssh", c(target, "echo 'Master connection ready'"), timeout = 10, error_on_status = TRUE)

  cli::cli_alert_success("SSH master connection established for {.strong {target}}")
  cli::cli_alert_info("All subsequent SSH and rsync operations will use this connection without re-authentication")

  invisible(target)
}

#' Upload data/code and submit SLURM job
#'
#' Builds remote directories, syncs data and code via rsync over an SSH
#' multiplexed connection, and submits a SLURM job. Optional `hpc_env`
#' preamble lines can be injected to load modules or set environment variables.
#'
#' @param target SSH target (e.g., "user@host" or a configured host).
#' @param data_dir Local dataset directory.
#' @param hpc_base_dir Remote base directory (the job will use a timestamped subdir).
#' @param output_name Output/model name (used for remote subdirectory).
#' @param training_params Character vector of CLI args passed to the Python trainer.
#' @param gpus Number of GPUs to request.
#' @param hpc_env Optional character vector of preamble lines (e.g., module loads).
#' @keywords internal
hpc_sync_and_submit <- function(target, data_dir, hpc_base_dir, output_name, training_params, gpus = 1, hpc_env = NULL) {
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  remote_base <- fs::path(hpc_base_dir, output_name, timestamp)

  # Create remote directories
  remote_dirs <- c(
    remote_base,
    fs::path(remote_base, "data"),
    fs::path(remote_base, "src"),
    fs::path(remote_base, "output")
  )

  cli::cli_progress_step("Creating remote directories")
  mkdir_cmd <- paste("mkdir -p", paste(shQuote(remote_dirs), collapse = " "))
  processx::run("ssh", c(target, mkdir_cmd), timeout = 30, error_on_status = TRUE)

  # Fast rsync for data
  cli::cli_progress_step("Syncing data to HPC")
  processx::run("rsync", c(
    "-avz", "--stats",
    paste0(data_dir, "/"),
    paste0(target, ":", remote_base, "/data/")
  ), timeout = Inf, spinner = TRUE, error_on_status = TRUE)

  # Fast rsync for code
  cli::cli_progress_step("Syncing training code to HPC")
  src_dir <- system.file("python", package = "petrographer")
  processx::run("rsync", c(
    "-avz", "--stats", "--exclude", "__pycache__/",
    paste0(src_dir, "/"),
    paste0(target, ":", remote_base, "/src/")
  ), timeout = Inf, spinner = TRUE, error_on_status = TRUE)

  # Generate SLURM script inline
  # Adjust resources based on GPU count
  cpus <- if (gpus > 1) gpus * 4 else 4
  mem <- if (gpus > 1) paste0(gpus * 16, "gb") else "16gb"

  # Environment/preamble lines
  # Default to your exact conda activation sequence
  preamble_lines <- c(
    "module purge",
    "module load conda",
    "conda activate /blue/nicolas.gauthier/share/conda/envs/petrographer"
  )
  preamble <- paste0(paste(preamble_lines, collapse = "\n"), "\n")
  if (!is.null(hpc_env) && length(hpc_env) > 0) {
    preamble <- paste(paste(hpc_env, collapse = "\n"), "\n", sep = "")
  }

  slurm_script <- paste0("#!/bin/bash\n",
    "#SBATCH --job-name=petrographer_train\n",
    "#SBATCH --time=02:00:00\n",
    "#SBATCH --cpus-per-task=", cpus, "\n",
    "#SBATCH --mem=", mem, "\n",
    "#SBATCH --gpus=", gpus, "\n",
    "#SBATCH --output=%x_%j.out\n",
    "#SBATCH --error=%x_%j.err\n",
    preamble,
    "cd ", shQuote(remote_base), "\n",
    "python src/train.py ", paste(training_params, collapse = " "))

  # Submit job via SSH
  cli::cli_progress_step("Submitting SLURM job")
  submit_cmd <- paste0("sbatch <<'EOF'\n", slurm_script, "\nEOF")
  submit_result <- processx::run("ssh", c(target, submit_cmd), timeout = 30, error_on_status = TRUE)

  if (!grepl("Submitted batch job", submit_result$stdout)) {
    cli::cli_abort("Unexpected SLURM response: {submit_result$stdout}")
  }

  job_id <- gsub("Submitted batch job ([0-9]+).*", "\\1", submit_result$stdout)
  cli::cli_alert_success("Job submitted with ID: {.strong {job_id}}")

  list(job_id = job_id, remote_base = remote_base)
}



#' Monitor SLURM job until completion
#' @keywords internal
hpc_monitor <- function(target, job_id, remote_base) {
  start_time <- Sys.time()

  # Helper function to get job status
  get_job_status <- function() {
    result <- processx::run("ssh", c(target, paste("squeue -j", job_id, "-h -o %T")), timeout = 30, error_on_status = FALSE)
    if (nchar(result$stdout) == 0) {
      return("FINISHED")
    }
    trimws(result$stdout)
  }

  # Helper function to show error details and abort
  show_error_and_abort <- function(status) {
    # Get the actual error output - this is usually most informative
    err_result <- processx::run("ssh", c(target,
      paste0("cat ", fs::path(remote_base, paste0("petrographer_train_", job_id, ".err")))),
      timeout = 30, error_on_status = FALSE)

    if (nchar(err_result$stdout) > 0) {
      cli::cli_h3("Error output")
      cli::cli_code(err_result$stdout)
    }

    # Get exit code for quick diagnosis
    exit_result <- processx::run("ssh", c(target,
      paste("sacct -j", job_id, "-n -o ExitCode | tail -n 1")),
      timeout = 30, error_on_status = FALSE)

    if (nchar(exit_result$stdout) > 0) {
      exit_code <- trimws(exit_result$stdout)
      cli::cli_alert_info("Exit code: {exit_code}")
    }

    cli::cli_abort("Job {.strong {job_id}} failed with status: {.emph {status}}")
  }

  # Check initial status
  current_status <- get_job_status()
  if (current_status %in% c("FAILED", "CANCELLED", "TIMEOUT")) {
    show_error_and_abort(current_status)
  }

  # Monitor PENDING state with spinner
  if (current_status == "PENDING") {
    cli::cli_progress_step("Job {.strong {job_id}} waiting in queue", spinner = TRUE)
    repeat {
      for (i in 1:30) {  # 30 seconds with updates every second
        cli::cli_progress_update()
        Sys.sleep(1)
      }
      current_status <- get_job_status()

      if (current_status != "PENDING") break
      if (current_status %in% c("FAILED", "CANCELLED", "TIMEOUT")) {
        show_error_and_abort(current_status)
      }
      if (difftime(Sys.time(), start_time, units = "hours") > 8) {
        cli::cli_abort("Job monitoring timeout after 8 hour{?s}")
      }
    }
  }

  # Monitor RUNNING state with spinner
  if (current_status == "RUNNING") {
    cli::cli_progress_step("Job {.strong {job_id}} training model", spinner = TRUE)
    repeat {
      for (i in 1:30) {  # 30 seconds with updates every second
        cli::cli_progress_update()
        Sys.sleep(1)
      }
      current_status <- get_job_status()

      if (current_status == "FINISHED") break
      if (current_status %in% c("FAILED", "CANCELLED", "TIMEOUT")) {
        show_error_and_abort(current_status)
      }
      if (difftime(Sys.time(), start_time, units = "hours") > 8) {
        cli::cli_abort("Job monitoring timeout after 8 hour{?s}")
      }
    }
  }

  # Check final completion status
  if (current_status == "FINISHED") {
    final <- processx::run("ssh", c(target, paste("sacct -j", job_id, "-n -o State | tail -n 1")), timeout = 30, error_on_status = FALSE)
    status <- trimws(final$stdout)
    if (status != "COMPLETED") {
      show_error_and_abort(status)
    }
  }

  cli::cli_alert_success("Training completed successfully on HPC!")
  return(TRUE)
}


#' Download trained model from HPC
#' @keywords internal
hpc_download <- function(target, remote_base, output_name, local_output_dir) {
  local_model_dir <- fs::path(local_output_dir, output_name)
  fs::dir_create(local_model_dir)

  cli::cli_progress_step("Downloading trained model from HPC")
  processx::run("rsync", c(
    "-avz", "--stats",
    paste0(target, ":", remote_base, "/output/"),
    paste0(local_model_dir, "/")
  ), timeout = Inf, spinner = TRUE, error_on_status = TRUE)

  # Verify essential artifacts (rsync downloads directly to local_model_dir)
  model_file <- fs::path(local_model_dir, "model_final.pth")
  config_file <- fs::path(local_model_dir, "config.yaml")

  if (!fs::file_exists(model_file) || !fs::file_exists(config_file)) {
    cli::cli_abort("Model artifacts incomplete - missing model_final.pth or config.yaml")
  }

  cli::cli_alert_success("Model downloaded to: {.path {local_model_dir}}")
  return(local_model_dir)
}
