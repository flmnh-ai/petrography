# ============================================================================
# HPC Helper Functions
# Streamlined SSH-based remote training
# ============================================================================


#' Create SSH connection to HPC
#' @keywords internal
hpc_session <- function(hpc_host, hpc_user = NULL) {
  target <- if (!is.null(hpc_user) && nzchar(hpc_user)) paste0(hpc_user, "@", hpc_host) else hpc_host
  ssh::ssh_connect(target)
}


#' Upload data/code and submit SLURM job
#' @keywords internal
hpc_sync_and_submit <- function(session, data_dir, hpc_base_dir, output_name, training_params) {
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  remote_base <- fs::path(hpc_base_dir, output_name, timestamp)
  
  # Create remote directories
  remote_dirs <- c(
    remote_base,
    fs::path(remote_base, "data"),
    fs::path(remote_base, "src"), 
    fs::path(remote_base, "output")
  )
  ssh::ssh_exec_wait(session, paste("mkdir -p", paste(shQuote(remote_dirs), collapse = " ")))
  
  # Upload data and code
  cli::cli_alert_info("Syncing data to HPC...")
  ssh::scp_upload(session, data_dir, paste0(remote_base, "/data"), recursive = TRUE)
  
  cli::cli_alert_info("Syncing training code to HPC...")
  src_dir <- system.file("python", package = "petrographer")
  ssh::scp_upload(session, src_dir, paste0(remote_base, "/src"), recursive = TRUE)
  
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
  
  # Submit job
  cli::cli_alert_info("Submitting SLURM job...")
  result <- ssh::ssh_exec_internal(session, paste0("sbatch <<'EOF'\n", slurm_script, "\nEOF"))
  if (result$status != 0) cli::cli_abort("Failed to submit SLURM job: {result$stderr}")
  
  job_id <- gsub("Submitted batch job ([0-9]+).*", "\\1", result$stdout)
  cli::cli_alert_success("Job submitted with ID: {job_id}")
  
  list(job_id = job_id, remote_base = remote_base)
}



#' Monitor SLURM job until completion
#' @keywords internal
hpc_monitor <- function(session, job_id, remote_base) {
  start_time <- Sys.time()
  cli::cli_alert_info("Monitoring job progress...")
  
  repeat {
    # Check if job exists in queue  
    result <- ssh::ssh_exec_internal(session, paste("squeue -j", job_id, "-h -o %T"), error = FALSE)
    
    if (nchar(result$stdout) == 0) {
      # Job finished - check final status
      final <- ssh::ssh_exec_internal(session, paste("sacct -j", job_id, "-n -o State | tail -n 1"))
      status <- trimws(final$stdout)
      if (status != "COMPLETED") {
        # Show error output
        err_result <- ssh::ssh_exec_internal(session, 
          paste0("cat ", fs::path(remote_base, paste0("petrographer_train_", job_id, ".err"))))
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
hpc_download <- function(session, remote_base, output_name, local_output_dir) {
  local_model_dir <- fs::path(local_output_dir, output_name)
  fs::dir_create(local_model_dir)
  
  cli::cli_alert_info("Downloading trained model...")
  ssh::scp_download(session, paste0(remote_base, "/output"), local_model_dir, recursive = TRUE)
  
  # Verify essential artifacts
  model_file <- fs::path(local_model_dir, "output", "model_final.pth")
  config_file <- fs::path(local_model_dir, "output", "config.yaml")
  
  if (!fs::file_exists(model_file) || !fs::file_exists(config_file)) {
    cli::cli_abort("Model artifacts incomplete - missing model_final.pth or config.yaml")
  }
  
  # Move files up one level to match expected structure
  file.rename(fs::path(local_model_dir, "output"), fs::path(local_model_dir, "temp"))
  fs::file_move(fs::dir_ls(fs::path(local_model_dir, "temp")), local_model_dir)
  fs::dir_delete(fs::path(local_model_dir, "temp"))
  
  cli::cli_alert_success("Model downloaded to: {.path {local_model_dir}}")
  return(local_model_dir)
}


