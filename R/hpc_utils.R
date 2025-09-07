# ============================================================================
# HPC Helper Functions
# Simplified, safer, and more robust for remote training
# ============================================================================

# Consistent SLURM job naming
.SLURM_JOB_NAME <- "petrographer_train"


# Parse and print a concise rsync summary from --stats output
.rsync_summary <- function(stdout, label = NULL) {
  if (!nzchar(stdout)) return(invisible())
  lines <- strsplit(stdout, "\n", fixed = TRUE)[[1]]
  sent_line <- lines[grepl("^sent \\d+", lines)]
  files_line <- lines[grepl("^Number of regular files transferred:", lines)]
  if (length(sent_line)) cli::cli_alert_info(paste0(if (!is.null(label)) paste0(label, ": ") else "", trimws(sent_line[1])))
  if (length(files_line)) cli::cli_alert_info(trimws(files_line[1]))
}

.ssh_target <- function(hpc_host, hpc_user = NULL) {
  if (!is.null(hpc_user) && nzchar(hpc_user)) paste0(hpc_user, "@", hpc_host) else hpc_host
}

.safe_remote_path <- function(remote_path, hpc_base_dir) {
  if (is.null(remote_path) || !nzchar(remote_path)) return(FALSE)
  if (remote_path %in% c("/", "~", ".", "./", "../")) return(FALSE)
  if (!startsWith(remote_path, hpc_base_dir)) return(FALSE)
  TRUE
}


#' Test SSH connection to HPC
#' @keywords internal
test_ssh_connection <- function(hpc_host = "hpg", hpc_user = NULL) {
  target <- .ssh_target(hpc_host, hpc_user)
  check <- processx::run("ssh", c("-O", "check", target), timeout = 5, error_on_status = FALSE)$status
  if (!identical(check, 0L)) {
    cli::cli_alert_info("SSH connection required. Please authenticate with Duo MFA...")
    processx::run("ssh", c("-tt", target, "echo 'Connection established'"), error_on_status = FALSE)
    status <- processx::run("ssh", c("-MNf", target), timeout = 5, error_on_status = FALSE)$status
    Sys.sleep(1)
    check <- processx::run("ssh", c("-O", "check", target), timeout = 5, error_on_status = FALSE)$status
    return(identical(check, 0L))
  }
  TRUE
}

close_ssh_connection <- function(hpc_host = "hpg", hpc_user = NULL) {
  target <- .ssh_target(hpc_host, hpc_user)
  invisible(processx::run("ssh", c("-O", "exit", target), timeout = 5, error_on_status = FALSE))
}


#' Setup remote directory structure on HPC
#' @keywords internal
setup_remote_directories <- function(hpc_host, hpc_user = NULL, hpc_base_dir, output_name) {
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  remote_session_dir <- fs::path(hpc_base_dir, output_name, timestamp)
  target <- .ssh_target(hpc_host, hpc_user)

  remote_dirs <- c(
    remote_session_dir,
    fs::path(remote_session_dir, "data"),
    fs::path(remote_session_dir, "src"),
    fs::path(remote_session_dir, "output")
  )

  if (!.safe_remote_path(remote_session_dir, hpc_base_dir)) {
    cli::cli_abort("Unsafe remote session path: {remote_session_dir}")
  }

  mkdir_cmd <- glue::glue("mkdir -p {paste(shQuote(remote_dirs), collapse = ' ')}")
  # Important: do NOT shQuote the whole remote command when calling ssh via processx
  res <- processx::run("ssh", c(target, mkdir_cmd), timeout = 30, error_on_status = FALSE)
  if (!identical(res$status, 0L)) cli::cli_abort("Failed to create remote directories on HPC: {res$stderr}")

  remote_session_dir
}



#' Sync local data to HPC using rsync
#' @keywords internal
sync_data_to_hpc <- function(local_data_dir, hpc_host, hpc_user = NULL, remote_session_dir, rsync_mode = c("update", "mirror"), dry_run = FALSE) {
  target <- .ssh_target(hpc_host, hpc_user)
  remote_data_dir <- fs::path(remote_session_dir, "data")
  rsync_mode <- match.arg(rsync_mode)
  step_id <- cli::cli_progress_step("Syncing data to HPC...", msg_done = "Data synced", msg_failed = "Data sync failed")
  args <- c("-az", "--stats", if (dry_run) "-n" else NULL, if (rsync_mode == "mirror") "--delete" else NULL, paste0(local_data_dir, "/"), paste0(target, ":", remote_data_dir))
  res <- processx::run("rsync", args, timeout = Inf, error_on_status = FALSE)
  if (!identical(res$status, 0L)) { cli::cli_progress_done(step_id, result = "failed"); cli::cli_abort("Failed to sync data to HPC: {res$stderr}") }
  cli::cli_progress_done(step_id, result = "done")
  .rsync_summary(res$stdout, "Data sync")
}



#' Sync training code to HPC
#' @keywords internal
sync_code_to_hpc <- function(hpc_host, hpc_user = NULL, remote_session_dir, dry_run = FALSE) {
  target <- .ssh_target(hpc_host, hpc_user)
  src_dir <- system.file("python", package = "petrographer")
  if (!fs::dir_exists(src_dir)) cli::cli_abort("Packaged python directory not found")
  step_id <- cli::cli_progress_step("Syncing training code to HPC...", msg_done = "Code synced", msg_failed = "Code sync failed")
  args <- c("-az", "--stats", if (dry_run) "-n" else NULL, "--exclude", "__pycache__/", paste0(src_dir, "/"), paste0(target, ":", fs::path(remote_session_dir, "src/")))
  res <- processx::run("rsync", args, timeout = Inf, error_on_status = FALSE)
  if (!identical(res$status, 0L)) { cli::cli_progress_done(step_id, result = "failed"); cli::cli_abort("Failed to sync training code to HPC: {res$stderr}") }
  cli::cli_progress_done(step_id, result = "done")
  .rsync_summary(res$stdout, "Code sync")
}


#' Generate and submit SLURM job for training
#' @keywords internal
submit_slurm_job <- function(hpc_host, hpc_user = NULL, remote_session_dir, output_name,
                             max_iter, learning_rate, num_classes, eval_period, checkpoint_period) {
  target <- .ssh_target(hpc_host, hpc_user)

  # Bootstrap script
  bootstrap <- generate_bootstrap_script(remote_session_dir, output_name, max_iter, learning_rate, num_classes, eval_period, checkpoint_period)
  bootstrap_path <- fs::path(remote_session_dir, "bootstrap.sh")
  tmp_boot <- tempfile(fileext = ".sh")
  writeLines(bootstrap, tmp_boot)
  res <- processx::run("rsync", c("-avz", tmp_boot, paste0(target, ":", bootstrap_path)), error_on_status = FALSE)
  unlink(tmp_boot)
  if (!identical(res$status, 0L)) cli::cli_abort("Failed to upload bootstrap script: {res$stderr}")

  # SLURM wrapper
  slurm_script <- generate_slurm_wrapper(remote_session_dir)
  script_path <- fs::path(remote_session_dir, "train_job.sh")
  tmp_slurm <- tempfile(fileext = ".sh")
  writeLines(slurm_script, tmp_slurm)
  res <- processx::run("rsync", c("-avz", tmp_slurm, paste0(target, ":", script_path)), error_on_status = FALSE)
  unlink(tmp_slurm)
  if (!identical(res$status, 0L)) cli::cli_abort("Failed to upload SLURM script: {res$stderr}")

  # submit
  submit_cmd <- glue::glue("cd {shQuote(remote_session_dir)} && sbatch train_job.sh")
  res <- processx::run("ssh", c(target, submit_cmd), error_on_status = FALSE)
  if (is.null(res$stdout) || !grepl("Submitted batch job", res$stdout)) {
    cli::cli_abort("Failed to submit SLURM job: {paste(c(res$stdout, res$stderr), collapse = '\\n')}")
  }
  gsub("Submitted batch job ([0-9]+)", "\\1", strsplit(res$stdout, "\n")[[1]][1])
}


#' Generate SLURM script content
#' @keywords internal
generate_bootstrap_script <- function(remote_session_dir, output_name, max_iter, learning_rate, num_classes, eval_period, checkpoint_period) {
  data_dir <- fs::path(remote_session_dir, "data")
  output_dir <- fs::path(remote_session_dir, "output")
  module_lines <- c("module purge", "module load detectron2")
  c(
    "#!/bin/bash",
    module_lines,
    glue::glue("mkdir -p {shQuote(output_dir)}"),
    glue::glue("cd {shQuote(remote_session_dir)}"),
    "echo Python: $(which python) && python -V",
    "echo CUDA: $CUDA_VISIBLE_DEVICES",
    glue::glue("python src/train.py \\") ,
    glue::glue("  --dataset-name {output_name}_train \\") ,
    glue::glue("  --annotation-json {fs::path(data_dir, 'train', '_annotations.coco.json')} \\") ,
    glue::glue("  --image-root {fs::path(data_dir, 'train')} \\") ,
    glue::glue("  --val-annotation-json {fs::path(data_dir, 'val', '_annotations.coco.json')} \\") ,
    glue::glue("  --val-image-root {fs::path(data_dir, 'val')} \\") ,
    glue::glue("  --output-dir {output_dir} \\") ,
    glue::glue("  --num-classes {num_classes} \\") ,
    glue::glue("  --device cuda \\") ,
    glue::glue("  --max-iter {max_iter} \\") ,
    glue::glue("  --learning-rate {learning_rate} \\") ,
    glue::glue("  --eval-period {eval_period} \\") ,
    glue::glue("  --checkpoint-period {checkpoint_period}"),
    "echo Training exit code: $?"
  )
}

generate_slurm_wrapper <- function(remote_session_dir) {
  # Hardcoded SLURM parameters
  sb <- list(time = "04:00:00", cpus = 4, mem = "24gb", gpus = 1, partition = NULL)
  lines <- c(
    "#!/bin/bash",
    glue::glue("#SBATCH --job-name={.SLURM_JOB_NAME}"),
    "#SBATCH --output=%x_%j.out",
    "#SBATCH --error=%x_%j.err",
    glue::glue("#SBATCH --time={sb$time}"),
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks=1",
    glue::glue("#SBATCH --cpus-per-task={sb$cpus}"),
    glue::glue("#SBATCH --mem={sb$mem}"),
    glue::glue("#SBATCH --gpus={sb$gpus}")
  )
  if (!is.null(sb$partition)) lines <- c(lines, glue::glue("#SBATCH --partition={sb$partition}"))
  c(lines, glue::glue("cd {shQuote(remote_session_dir)} && bash bootstrap.sh"))
}


#' Monitor SLURM job status until completion
#' @keywords internal
monitor_slurm_job <- function(hpc_host, hpc_user = NULL, job_id, poll_interval = 30, remote_session_dir, max_duration = 8*3600, max_idle = 1800) {
  target <- .ssh_target(hpc_host, hpc_user)
  last_lines <- 0
  t_start <- Sys.time()
  t_last_output <- t_start
  out_file <- fs::path(remote_session_dir, glue::glue("{.SLURM_JOB_NAME}_{job_id}.out"))

  repeat {
    if (as.numeric(difftime(Sys.time(), t_start, units = "secs")) > max_duration) {
      cli::cli_abort("Monitoring timed out after {round(max_duration/3600,1)} hours")
    }

    status_cmd <- glue::glue("squeue -j {job_id} -h -o %T")
    status_res <- processx::run("ssh", c(target, status_cmd), error_on_status = FALSE)

    out_cmd <- glue::glue("if [ -f {out_file} ]; then wc -l < {out_file} && tail -n 10 {out_file}; fi")
    out_res <- processx::run("ssh", c(target, out_cmd), error_on_status = FALSE)
    if (nzchar(out_res$stdout)) {
      lines <- strsplit(out_res$stdout, "\n")[[1]]
      n <- suppressWarnings(as.numeric(lines[1]))
      if (!is.na(n) && n > last_lines) {
        cli::cli_h3("Training progress")
        cli::cli_code(paste(lines[-1], collapse = "\n"))
        last_lines <- n
        t_last_output <- Sys.time()
      }
    }

    if (as.numeric(difftime(Sys.time(), t_last_output, units = "secs")) > max_idle) {
      cli::cli_abort("No new output for {round(max_idle/60,1)} minutes; aborting monitor")
    }

    if (!nzchar(status_res$stdout)) {
      final_cmd <- glue::glue("sacct -j {job_id} -n -o State | tail -n 1")
      final_res <- processx::run("ssh", c(target, final_cmd), error_on_status = FALSE)
      final_status <- if (nzchar(final_res$stdout)) trimws(strsplit(final_res$stdout, "\n")[[1]][1]) else "UNKNOWN"
      cli::cli_alert_info("Job {job_id} final status: {final_status}")
      if (final_status != "COMPLETED") {
        err_file <- fs::path(remote_session_dir, glue::glue("{.SLURM_JOB_NAME}_{job_id}.err"))
        err_cmd <- glue::glue("if [ -f {err_file} ]; then tail -n 20 {err_file}; fi")
        err_res <- processx::run("ssh", c(target, err_cmd), error_on_status = FALSE)
        if (nzchar(err_res$stdout)) {
          cli::cli_h3("Error output")
          cli::cli_code(err_res$stdout)
        }
      }
      return(final_status)
    }

    current_status <- trimws(strsplit(status_res$stdout, "\n")[[1]][1])
    cli::cli_alert_info("Job {job_id} status: {current_status}")
    if (current_status %in% c("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT")) return(current_status)
    Sys.sleep(poll_interval)
  }
}


#' Download trained model from HPC to local machine
#' @keywords internal
download_trained_model <- function(hpc_host, hpc_user = NULL, remote_session_dir,
                                   output_name, local_output_dir) {
  target <- .ssh_target(hpc_host, hpc_user)
  local_model_dir <- fs::path(local_output_dir, output_name)
  fs::dir_create(local_model_dir)
  remote_output_dir <- fs::path(remote_session_dir, "output")
  step_id <- cli::cli_progress_step("Downloading trained model...", msg_done = "Download complete", msg_failed = "Download failed")
  res <- processx::run(
    "rsync",
    c("-az", "--stats", paste0(target, ":", remote_output_dir, "/"), paste0(local_model_dir, "/")),
    timeout = Inf, error_on_status = FALSE
  )
  if (!identical(res$status, 0L)) { cli::cli_progress_done(step_id, result = "failed"); cli::cli_abort("Failed to download trained model from HPC: {res$stderr}") }
  cli::cli_progress_done(step_id, result = "done")
  .rsync_summary(res$stdout, "Download")
  return(local_model_dir)
}



verify_local_artifacts <- function(local_model_dir) {
  pth <- fs::path(local_model_dir, "model_final.pth")
  cfg <- fs::path(local_model_dir, "config.yaml")
  fs::file_exists(pth) && fs::file_exists(cfg)
}
