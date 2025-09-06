# ============================================================================
# HPC Helper Functions
# ============================================================================

#' Test SSH connection to HPC
#' @keywords internal
test_ssh_connection <- function(hpc_host = "hpg", hpc_user = NULL) {
  # Check if control master already exists
  check <- system2("ssh", c("-O", "check", hpc_host),
                   stdout = FALSE, stderr = FALSE)

  if (check != 0) {
    cat("🔐 SSH connection required. Please authenticate with Duo MFA...\n")

    # Open interactive SSH connection for Duo MFA
    # This will show the Duo prompt in the console
    system2("ssh", c("-tt", hpc_host, "echo 'Connection established'"),
            wait = TRUE)  # MUST be wait=TRUE for interactive prompt

    # Now start the control master in background
    status <- system2("ssh", c("-MNf", hpc_host), wait = FALSE)
    Sys.sleep(1)

    # Verify connection
    check <- system2("ssh", c("-O", "check", hpc_host),
                     stdout = FALSE, stderr = FALSE)
    return(check == 0)
  }

  return(TRUE)
}


#' Setup remote directory structure on HPC
#' @keywords internal
setup_remote_directories <- function(hpc_host, hpc_user = NULL, hpc_base_dir, output_name) {
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  remote_session_dir <- file.path(hpc_base_dir, paste0(output_name, "_", timestamp), fsep = "/")
  target <- if (!is.null(hpc_user)) paste0(hpc_user, "@", hpc_host) else hpc_host

  remote_dirs <- c(
    remote_session_dir,
    file.path(remote_session_dir, "data", fsep = "/"),
    file.path(remote_session_dir, "src", fsep = "/")
  )

  mkdir_cmd <- paste("mkdir -p", paste(shQuote(remote_dirs), collapse = " "))
  ssh_cmd <- paste("ssh", target, shQuote(mkdir_cmd))

  result <- system(ssh_cmd, ignore.stderr = TRUE)
  if (result != 0) stop("❌ Failed to create remote directories on HPC")

  return(remote_session_dir)
}



#' Sync local data to HPC using rsync
#' @keywords internal
sync_data_to_hpc <- function(local_data_dir, hpc_host, hpc_user = NULL, remote_session_dir) {
  # Don't override user if using SSH config
  if (hpc_host == "hpg" && is.null(hpc_user)) {
    target <- hpc_host  # SSH config handles the user
  } else {
    target <- if (!is.null(hpc_user)) paste0(hpc_user, "@", hpc_host) else hpc_host
  }

  remote_data_dir <- file.path(remote_session_dir, "data", fsep = "/")

  # Sync the contents of local_data_dir to remote data dir
  # This preserves the train/ and val/ subdirectories
  rsync_cmd <- paste("rsync -avz --progress",
                     shQuote(paste0(local_data_dir, "/")),
                     shQuote(paste0(target, ":", remote_data_dir)))

  result <- system(rsync_cmd)
  if (result != 0) stop("❌ Failed to sync data to HPC")
}



#' Sync training code to HPC
#' @keywords internal
sync_code_to_hpc <- function(hpc_host, hpc_user = NULL, remote_session_dir) {
  target <- if (!is.null(hpc_user)) paste0(hpc_user, "@", hpc_host) else hpc_host

  rsync_cmd <- paste("rsync -avz", "src/train.py",
                     shQuote(paste0(target, ":", file.path(remote_session_dir, "src/", fsep = "/"))))

  result <- system(rsync_cmd)
  if (result != 0) stop("❌ Failed to sync training code to HPC")
}


#' Generate and submit SLURM job for training
#' @keywords internal
submit_slurm_job <- function(hpc_host, hpc_user = NULL, remote_session_dir, output_name,
                             max_iter, learning_rate, num_classes, eval_period, checkpoint_period) {
  slurm_script <- generate_slurm_script(remote_session_dir, output_name, max_iter, learning_rate, num_classes, eval_period, checkpoint_period)
  script_path <- file.path(remote_session_dir, "train_job.sh", fsep = "/")
  temp_script <- tempfile(fileext = ".sh")
  writeLines(slurm_script, temp_script)

  target <- if (!is.null(hpc_user)) paste0(hpc_user, "@", hpc_host) else hpc_host
  rsync_cmd <- paste("rsync -avz", shQuote(temp_script), shQuote(paste0(target, ":", script_path)))

  result <- system(rsync_cmd)
  if (result != 0) stop("❌ Failed to transfer SLURM script to HPC")
  unlink(temp_script)

  ssh_cmd <- paste("ssh", target, shQuote(paste("cd", shQuote(remote_session_dir), "&& sbatch train_job.sh")))
  result <- system(ssh_cmd, intern = TRUE)

  if (length(result) == 0 || !grepl("Submitted batch job", result[1])) {
    stop("❌ Failed to submit SLURM job: ", paste(result, collapse = "\n"))
  }

  job_id <- gsub("Submitted batch job ([0-9]+)", "\\1", result[1])
  return(job_id)
}


#' Generate SLURM script content
#' @keywords internal
generate_slurm_script <- function(remote_session_dir, output_name, max_iter, learning_rate, num_classes, eval_period, checkpoint_period) {
  data_dir <- file.path(remote_session_dir, "data", fsep = "/")
  output_dir <- file.path(remote_session_dir, "output", fsep = "/")

  script_lines <- c(
    "#!/bin/bash",
    "#SBATCH --job-name=petrography_train",
    "#SBATCH --output=%x_%j.out",
    "#SBATCH --error=%x_%j.err",
    "#SBATCH --time=02:00:00",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks=1",
    "#SBATCH --cpus-per-task=4",
    "#SBATCH --mem=24gb",
    "#SBATCH --gpus=1",
    "",
    "module purge",
    "module load detectron2",
    "",
    paste("mkdir -p", shQuote(output_dir)),
    "",
    paste("cd", shQuote(remote_session_dir)),
    paste("python src/train.py \\"),
    paste("  --dataset-name", paste0(output_name, "_train"), "\\"),
    paste("  --annotation-json", file.path(data_dir, "train", "_annotations.coco.json", fsep = "/"), "\\"),
    paste("  --image-root", file.path(data_dir, "train", fsep = "/"), "\\"),
    paste("  --val-annotation-json", file.path(data_dir, "val", "_annotations.coco.json", fsep = "/"), "\\"),
    paste("  --val-image-root", file.path(data_dir, "val", fsep = "/"), "\\"),
    paste("  --output-dir", output_dir, "\\"),
    paste("  --num-classes", num_classes, "\\"),
    paste("  --device", "cuda", "\\"),
    paste("  --max-iter", max_iter, "\\"),
    paste("  --learning-rate", learning_rate, "\\"),
    paste("  --eval-period", eval_period, "\\"),
    paste("  --checkpoint-period", checkpoint_period),
    "",
    'echo "Training completed with exit code: $?"'
  )

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
    ssh_cmd <- paste("ssh", target,
                     shQuote(paste("squeue -j", job_id, "-h -o %T")))

    status_result <- tryCatch({
      system(ssh_cmd, intern = TRUE, ignore.stderr = TRUE)
    }, error = function(e) character(0))

    # Check output file for progress
    output_file <- file.path(remote_session_dir, paste0("petrography_train_", job_id, ".out"))

    # Get line count and tail of output file
    output_cmd <- paste("ssh", target,
                        shQuote(paste("if [ -f", output_file, "]; then wc -l <", output_file,
                                      "&& tail -n 10", output_file, "; fi")))

    output_result <- tryCatch({
      system(output_cmd, intern = TRUE, ignore.stderr = TRUE)
    }, error = function(e) character(0))

    if (length(output_result) > 0) {
      current_line_count <- as.numeric(output_result[1])
      if (current_line_count > last_line_count) {
        cat("\n📊 Training progress:\n")
        cat(paste(output_result[-1], collapse = "\n"), "\n")
        last_line_count <- current_line_count
      }
    }

    if (length(status_result) == 0) {
      # Not in queue anymore, check final state
      ssh_cmd <- paste("ssh", target,
                       shQuote(paste("sacct -j", job_id, "-n -o State | tail -n 1")))

      final_status <- tryCatch({
        system(ssh_cmd, intern = TRUE, ignore.stderr = TRUE)
      }, error = function(e) "UNKNOWN")

      final_status <- if (length(final_status) > 0) trimws(final_status[1]) else "UNKNOWN"
      cat("🏁 Job", job_id, "final status:", final_status, "\n")

      # Show any error output if job failed
      if (final_status != "COMPLETED") {
        error_file <- file.path(remote_session_dir, paste0("petrography_train_", job_id, ".err"))
        error_cmd <- paste("ssh", target,
                           shQuote(paste("if [ -f", error_file, "]; then tail -n 20", error_file, "; fi")))

        error_output <- tryCatch({
          system(error_cmd, intern = TRUE, ignore.stderr = TRUE)
        }, error = function(e) character(0))

        if (length(error_output) > 0) {
          cat("\n❌ Error output:\n")
          cat(paste(error_output, collapse = "\n"), "\n")
        }
      }

      return(final_status)
    }

    current_status <- trimws(status_result[1])
    cat("⏳ Job", job_id, "status:", current_status, "\n")

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
  target <- if (!is.null(hpc_user)) paste0(hpc_user, "@", hpc_host) else hpc_host

  local_model_dir <- file.path(local_output_dir, output_name, fsep = "/")
  dir.create(local_model_dir, showWarnings = FALSE, recursive = TRUE)

  remote_output_dir <- file.path(remote_session_dir, "output", fsep = "/")

  rsync_cmd <- paste("rsync -avz --progress",
                     shQuote(paste0(target, ":", remote_output_dir, "/")),
                     shQuote(paste0(local_model_dir, "/")))

  result <- system(rsync_cmd)
  if (result != 0) stop("❌ Failed to download trained model from HPC")

  return(local_model_dir)
}


#' Clean up remote session directory
#' @keywords internal
cleanup_remote_session <- function(hpc_host, hpc_user = NULL, remote_session_dir) {
  target <- if (!is.null(hpc_user)) paste0(hpc_user, "@", hpc_host) else hpc_host

  ssh_cmd <- paste("ssh", target,
                   shQuote(paste("rm -rf", shQuote(remote_session_dir))))

  result <- system(ssh_cmd, ignore.stderr = TRUE)
  if (result != 0) {
    warning("⚠️ Failed to clean up remote session directory:", remote_session_dir)
  }
}