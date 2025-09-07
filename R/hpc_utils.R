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
  cli::cli_alert_info("Connecting to: {target}")
  
  # Check if master connection already exists
  check_result <- processx::run("ssh", c("-O", "check", target), timeout = 5, error_on_status = FALSE)
  if (check_result$status == 0) {
    cli::cli_alert_success("SSH master connection already active for {target}")
    return(invisible(target))
  }
  
  # Start master connection (handles Duo interactively)
  cli::cli_alert_info("Starting SSH master connection (Duo authentication required)...")
  cli::cli_alert("Please respond to any authentication prompts below:")
  
  # Start master: -M=master, -N=no command, -f=background
  processx::run("ssh", c("-MNf", target), timeout = 60, error_on_status = TRUE)
  
  # Wait a moment for connection to establish
  Sys.sleep(2)
  
  # Verify master connection works
  processx::run("ssh", c(target, "echo 'Master connection ready'"), timeout = 10, error_on_status = TRUE)
  
  cli::cli_alert_success("SSH master connection established for {target}")
  cli::cli_alert_info("All subsequent SSH and rsync operations will use this connection without re-authentication")
  
  invisible(target)
}

#' Upload data/code and submit SLURM job
#' @keywords internal
hpc_sync_and_submit <- function(target, data_dir, hpc_base_dir, output_name, training_params) {
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  remote_base <- fs::path(hpc_base_dir, output_name, timestamp)
  
  # Create remote directories
  remote_dirs <- c(
    remote_base,
    fs::path(remote_base, "data"),
    fs::path(remote_base, "src"),
    fs::path(remote_base, "output")
  )
  
  cli::cli_alert_info("Creating remote directories...")
  mkdir_cmd <- paste("mkdir -p", paste(shQuote(remote_dirs), collapse = " "))
  processx::run("ssh", c(target, mkdir_cmd), timeout = 30, error_on_status = TRUE)
  
  # Fast rsync for data
  cli::cli_alert_info("Syncing data to HPC...")
  processx::run("rsync", c(
    "-avz", "--stats",
    paste0(data_dir, "/"),
    paste0(target, ":", remote_base, "/data/")
  ), timeout = Inf, spinner = TRUE, error_on_status = TRUE)
  
  # Fast rsync for code  
  cli::cli_alert_info("Syncing training code to HPC...")
  src_dir <- system.file("python", package = "petrographer")
  processx::run("rsync", c(
    "-avz", "--stats", "--exclude", "__pycache__/",
    paste0(src_dir, "/"),
    paste0(target, ":", remote_base, "/src/")
  ), timeout = Inf, spinner = TRUE, error_on_status = TRUE)

  # Generate SLURM script inline
  slurm_script <- paste0("#!/bin/bash\n",
    "#SBATCH --job-name=petrographer_train\n",
    "#SBATCH --time=04:00:00\n",
    "#SBATCH --cpus-per-task=4\n",
    "#SBATCH --mem=24gb\n",
    "#SBATCH --gpus=1\n",
    "#SBATCH --output=%x_%j.out\n",
    "#SBATCH --error=%x_%j.err\n",
    "module purge && module load detectron2\n",
    "cd ", shQuote(remote_base), "\n",
    "python src/train.py ", paste(training_params, collapse = " "))

  # Submit job via SSH
  cli::cli_alert_info("Submitting SLURM job...")
  submit_cmd <- paste0("sbatch <<'EOF'\n", slurm_script, "\nEOF")
  submit_result <- processx::run("ssh", c(target, submit_cmd), timeout = 30, error_on_status = TRUE)
  
  if (!grepl("Submitted batch job", submit_result$stdout)) {
    cli::cli_abort("Unexpected SLURM response: {submit_result$stdout}")
  }
  
  job_id <- gsub("Submitted batch job ([0-9]+).*", "\\1", submit_result$stdout)
  cli::cli_alert_success("Job submitted with ID: {job_id}")

  list(job_id = job_id, remote_base = remote_base)
}



#' Monitor SLURM job until completion
#' @keywords internal
hpc_monitor <- function(target, job_id, remote_base) {
  start_time <- Sys.time()
  cli::cli_alert_info("Monitoring job progress...")

  repeat {
    # Check if job exists in queue
    result <- processx::run("ssh", c(target, paste("squeue -j", job_id, "-h -o %T")), timeout = 30, error_on_status = FALSE)

    if (nchar(result$stdout) == 0) {
      # Job finished - check final status
      final <- processx::run("ssh", c(target, paste("sacct -j", job_id, "-n -o State | tail -n 1")), timeout = 30, error_on_status = FALSE)
      status <- trimws(final$stdout)
      if (status != "COMPLETED") {
        # Show error output
        err_result <- processx::run("ssh", c(target, 
          paste0("cat ", fs::path(remote_base, paste0("petrographer_train_", job_id, ".err")))), 
          timeout = 30, error_on_status = FALSE)
        if (nchar(err_result$stdout) > 0) {
          cli::cli_h3("Error output")
          cli::cli_code(err_result$stdout)
        }
        cli::cli_abort("Job {job_id} failed with status: {status}")
      }
      cli::cli_alert_success("Training completed successfully on HPC!")
      return(TRUE)
    }

    # Show periodic progress
    current_status <- trimws(result$stdout)
    cli::cli_alert_info("Job {job_id} status: {current_status}")

    if (current_status %in% c("FAILED", "CANCELLED", "TIMEOUT")) {
      cli::cli_abort("Job {job_id} failed with status: {current_status}")
    }

    Sys.sleep(30)
    if (difftime(Sys.time(), start_time, units = "hours") > 8) {
      cli::cli_abort("Job monitoring timeout after 8 hours")
    }
  }
}


#' Download trained model from HPC
#' @keywords internal
hpc_download <- function(target, remote_base, output_name, local_output_dir) {
  local_model_dir <- fs::path(local_output_dir, output_name)
  fs::dir_create(local_model_dir)

  cli::cli_alert_info("Downloading trained model from HPC...")
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
