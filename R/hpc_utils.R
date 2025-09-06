# ============================================================================
# HPC Helper Functions
# ============================================================================

#' Test SSH connection to HPC
#' @keywords internal
test_ssh_connection <- function(hpc_host = "hpg", hpc_user = NULL) {
  # Determine SSH target respecting optional user
  target <- if (!is.null(hpc_user) && !(hpc_host == "hpg" && is.null(hpc_user))) glue::glue("{hpc_user}@{hpc_host}") else hpc_host

  # Check if control master already exists
  check <- processx::run("ssh", c("-O", "check", target), error_on_status = FALSE)$status

  if (check != 0) {
    cli::cli_alert_info("SSH connection required. Please authenticate with Duo MFA...")

    # Open interactive SSH connection for Duo MFA
    # This will show the Duo prompt in the console
    # Keep system2 for interactive TTY-based Duo prompt
    system2("ssh", c("-tt", target, "echo 'Connection established'"), wait = TRUE)

    # Now start the control master in background
    status <- processx::run("ssh", c("-MNf", target), error_on_status = FALSE)$status
    Sys.sleep(1)

    # Verify connection
    check <- processx::run("ssh", c("-O", "check", target), error_on_status = FALSE)$status
    return(check == 0L)
  }

  return(TRUE)
}


#' Setup remote directory structure on HPC
#' @keywords internal
setup_remote_directories <- function(hpc_host, hpc_user = NULL, hpc_base_dir, output_name) {
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  remote_session_dir <- fs::path(hpc_base_dir, glue::glue("{output_name}_{timestamp}"))
  target <- if (!is.null(hpc_user)) glue::glue("{hpc_user}@{hpc_host}") else hpc_host

  remote_dirs <- c(
    remote_session_dir,
    fs::path(remote_session_dir, "data"),
    fs::path(remote_session_dir, "src")
  )

  mkdir_cmd <- glue::glue("mkdir -p {paste(shQuote(remote_dirs), collapse = ' ')}")
  res <- processx::run("ssh", c(target, shQuote(mkdir_cmd)), error_on_status = FALSE)
  if (!identical(res$status, 0L)) cli::cli_abort("Failed to create remote directories on HPC")

  return(remote_session_dir)
}



#' Sync local data to HPC using rsync
#' @keywords internal
sync_data_to_hpc <- function(local_data_dir, hpc_host, hpc_user = NULL, remote_session_dir) {
  # Don't override user if using SSH config
  if (hpc_host == "hpg" && is.null(hpc_user)) {
    target <- hpc_host  # SSH config handles the user
  } else {
    target <- if (!is.null(hpc_user)) glue::glue("{hpc_user}@{hpc_host}") else hpc_host
  }

  remote_data_dir <- fs::path(remote_session_dir, "data")

  # Sync the contents of local_data_dir to remote data dir
  # This preserves the train/ and val/ subdirectories
  res <- processx::run(
    "rsync",
    c("-avz", "--progress", paste0(local_data_dir, "/"), paste0(target, ":", remote_data_dir)),
    echo = TRUE, error_on_status = FALSE
  )
  if (!identical(res$status, 0L)) cli::cli_abort("Failed to sync data to HPC")
}



#' Sync training code to HPC
#' @keywords internal
sync_code_to_hpc <- function(hpc_host, hpc_user = NULL, remote_session_dir) {
  target <- if (!is.null(hpc_user)) glue::glue("{hpc_user}@{hpc_host}") else hpc_host

  train_script <- system.file("python", "train.py", package = "petrographer")
  res <- processx::run(
    "rsync",
    c("-avz", train_script, paste0(target, ":", fs::path(remote_session_dir, "src/"))),
    echo = TRUE, error_on_status = FALSE
  )
  if (!identical(res$status, 0L)) cli::cli_abort("Failed to sync training code to HPC")
}


#' Generate and submit SLURM job for training
#' @keywords internal
submit_slurm_job <- function(hpc_host, hpc_user = NULL, remote_session_dir, output_name,
                             max_iter, learning_rate, num_classes, eval_period, checkpoint_period) {
  slurm_script <- generate_slurm_script(remote_session_dir, output_name, max_iter, learning_rate, num_classes, eval_period, checkpoint_period)
  script_path <- fs::path(remote_session_dir, "train_job.sh")
  temp_script <- tempfile(fileext = ".sh")
  writeLines(slurm_script, temp_script)

  target <- if (!is.null(hpc_user)) glue::glue("{hpc_user}@{hpc_host}") else hpc_host
  res <- processx::run("rsync", c("-avz", temp_script, paste0(target, ":", script_path)), echo = TRUE, error_on_status = FALSE)
  if (!identical(res$status, 0L)) cli::cli_abort("Failed to transfer SLURM script to HPC")
  unlink(temp_script)

  submit_cmd <- glue::glue("cd {shQuote(remote_session_dir)} && sbatch train_job.sh")
  res <- processx::run("ssh", c(target, shQuote(submit_cmd)), error_on_status = FALSE)

  if (is.null(res$stdout) || !grepl("Submitted batch job", res$stdout)) {
    cli::cli_abort("Failed to submit SLURM job: {paste(c(res$stdout, res$stderr), collapse = '\\n')}")
  }
  job_id <- gsub("Submitted batch job ([0-9]+)", "\\1", strsplit(res$stdout, "\n")[[1]][1])
  return(job_id)
}


#' Generate SLURM script content
#' @keywords internal
generate_slurm_script <- function(remote_session_dir, output_name, max_iter, learning_rate, num_classes, eval_period, checkpoint_period) {
  data_dir <- fs::path(remote_session_dir, "data")
  output_dir <- fs::path(remote_session_dir, "output")

  script_lines <- glue::glue("
#!/bin/bash
#SBATCH --job-name=petrography_train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --gpus=1

module purge
module load detectron2

mkdir -p {shQuote(output_dir)}

cd {shQuote(remote_session_dir)}
python src/train.py \\
  --dataset-name {output_name}_train \\
  --annotation-json {fs::path(data_dir, 'train', '_annotations.coco.json')} \\
  --image-root {fs::path(data_dir, 'train')} \\
  --val-annotation-json {fs::path(data_dir, 'val', '_annotations.coco.json')} \\
  --val-image-root {fs::path(data_dir, 'val')} \\
  --output-dir {output_dir} \\
  --num-classes {num_classes} \\
  --device cuda \\
  --max-iter {max_iter} \\
  --learning-rate {learning_rate} \\
  --eval-period {eval_period} \\
  --checkpoint-period {checkpoint_period}

echo \"Training completed with exit code: $?\"
") |> stringr::str_split("\\n") |> unlist()

  return(script_lines)
}


#' Monitor SLURM job status until completion
#' @keywords internal
monitor_slurm_job <- function(hpc_host, hpc_user = NULL, job_id, monitor_interval = 30, remote_session_dir) {
  # Use hpc_host directly since SSH config handles user
  if (hpc_host == "hpg" && is.null(hpc_user)) {
    target <- hpc_host
  } else {
    target <- if (!is.null(hpc_user)) paste0(hpc_user, "@", hpc_host) else hpc_host
  }

  # Track last line count to show only new output
  last_line_count <- 0

  while (TRUE) {
    # Check if job is still in queue
    status_cmd <- glue::glue("squeue -j {job_id} -h -o %T")
    status_result <- tryCatch({
      processx::run("ssh", c(target, shQuote(status_cmd)), error_on_status = FALSE)
    }, error = function(e) NULL)

    # Check output file for progress
    output_file <- fs::path(remote_session_dir, glue::glue("petrography_train_{job_id}.out"))

    # Get line count and tail of output file
    output_sh <- glue::glue("if [ -f {output_file} ]; then wc -l < {output_file} && tail -n 10 {output_file}; fi")
    output_result <- tryCatch({
      processx::run("ssh", c(target, shQuote(output_sh)), error_on_status = FALSE)
    }, error = function(e) NULL)

    if (!is.null(output_result) && nzchar(output_result$stdout)) {
      out_lines <- strsplit(output_result$stdout, "\n")[[1]]
      current_line_count <- suppressWarnings(as.numeric(out_lines[1]))
      if (current_line_count > last_line_count) {
        cli::cli_h3("Training progress")
        cli::cli_code(paste(out_lines[-1], collapse = "\n"))
        last_line_count <- current_line_count
      }
    }

    if (is.null(status_result) || !nzchar(status_result$stdout)) {
      # Not in queue anymore, check final state
      final_cmd <- glue::glue("sacct -j {job_id} -n -o State | tail -n 1")
      final_res <- tryCatch({
        processx::run("ssh", c(target, shQuote(final_cmd)), error_on_status = FALSE)
      }, error = function(e) NULL)

      final_status <- if (!is.null(final_res) && nzchar(final_res$stdout)) trimws(strsplit(final_res$stdout, "\n")[[1]][1]) else "UNKNOWN"
      cli::cli_alert_info("Job {job_id} final status: {final_status}")

      # Show any error output if job failed
      if (final_status != "COMPLETED") {
        error_file <- fs::path(remote_session_dir, glue::glue("petrography_train_{job_id}.err"))
        error_sh <- glue::glue("if [ -f {error_file} ]; then tail -n 20 {error_file}; fi")
        error_res <- tryCatch({
          processx::run("ssh", c(target, shQuote(error_sh)), error_on_status = FALSE)
        }, error = function(e) NULL)

        if (!is.null(error_res) && nzchar(error_res$stdout)) {
          cli::cli_h3("Error output")
          cli::cli_code(paste(strsplit(error_res$stdout, "\n")[[1]], collapse = "\n"))
        }
      }

      return(final_status)
    }

    current_status <- trimws(strsplit(status_result$stdout, "\n")[[1]][1])
    cli::cli_alert_info("Job {job_id} status: {current_status}")

    if (current_status %in% c("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT")) {
      return(current_status)
    }

    Sys.sleep(monitor_interval)
  }
}


#' Download trained model from HPC to local machine
#' @keywords internal
download_trained_model <- function(hpc_host, hpc_user = NULL, remote_session_dir,
                                   output_name, local_output_dir) {
  target <- if (!is.null(hpc_user)) glue::glue("{hpc_user}@{hpc_host}") else hpc_host

  local_model_dir <- fs::path(local_output_dir, output_name)
  fs::dir_create(local_model_dir)

  remote_output_dir <- fs::path(remote_session_dir, "output")

  res <- processx::run(
    "rsync",
    c("-avz", "--progress", paste0(target, ":", remote_output_dir, "/"), paste0(local_model_dir, "/")),
    echo = TRUE, error_on_status = FALSE
  )
  if (!identical(res$status, 0L)) cli::cli_abort("Failed to download trained model from HPC")

  return(local_model_dir)
}


#' Clean up remote session directory
#' @keywords internal
cleanup_remote_session <- function(hpc_host, hpc_user = NULL, remote_session_dir) {
  target <- if (!is.null(hpc_user)) glue::glue("{hpc_user}@{hpc_host}") else hpc_host

  rm_cmd <- glue::glue("rm -rf {shQuote(remote_session_dir)}")
  res <- processx::run("ssh", c(target, shQuote(rm_cmd)), error_on_status = FALSE)
  if (!identical(res$status, 0L)) {
    cli::cli_alert_warning("Failed to clean up remote session directory: {remote_session_dir}")
  }
}
