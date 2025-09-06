# ============================================================================
# Model Training Functions
# ============================================================================

#' Train a new petrography detection model
#' @param data_dir Local directory containing train and val subdirectories with COCO annotations
#' @param output_name Name for the trained model (default: "petrography_model")
#' @param max_iter Maximum training iterations (default: 5000)
#' @param num_classes Number of object classes (default: 5)
#' @param device Device for local training: 'cpu', 'cuda', 'mps' (default: 'cpu')
#' @param eval_period Validation evaluation frequency in iterations (default: 500)
#' @param checkpoint_period Checkpoint saving frequency (0=final only, >0=every N iterations, default: 0)
#' @param hpc_host SSH hostname for HPC training (NULL for local training)
#' @param hpc_user Username for HPC (default: current user)
#' @param hpc_base_dir Remote base directory on HPC (default: "~/petrography_training")
#' @param local_output_dir Local directory to save trained model (default: "Detectron2_Models")
#' @param cleanup_remote Whether to cleanup remote files after download (default: TRUE)
#' @param monitor_interval How often to check job status in seconds (default: 30)
#' @return Path to trained model directory
#' @export
train_model <- function(data_dir,
                       output_name = "petrography_model",
                       max_iter = 5000,
                       learning_rate = 0.001,
                       num_classes = 5,
                       device = "cpu",
                       eval_period = 500,
                       checkpoint_period = 0,
                       hpc_host = NULL,
                       hpc_user = NULL,
                       hpc_base_dir = NULL,
                       local_output_dir = "Detectron2_Models",
                       cleanup_remote = TRUE,
                       monitor_interval = 30) {


  # Validate inputs
  if (!dir.exists(data_dir)) {
    stop("Data directory not found: ", data_dir)
  }

  train_dir <- file.path(data_dir, "train", fsep = "/")
  val_dir <- file.path(data_dir, "val", fsep = "/")

  if (!dir.exists(train_dir) || !dir.exists(val_dir)) {
    stop("Data directory must contain 'train' and 'val' subdirectories")
  }

  if (!file.exists(file.path(train_dir, "_annotations.coco.json", fsep = "/"))) {
    stop("Missing COCO annotations in train directory")
  }

  if (!file.exists(file.path(val_dir, "_annotations.coco.json", fsep = "/"))) {
    stop("Missing COCO annotations in val directory")
  }

  # Determine training mode
  if (is.null(hpc_host)) {
    return(train_model_local(data_dir, output_name, max_iter, learning_rate, num_classes, device, eval_period, checkpoint_period, local_output_dir))
  } else {
    return(train_model_hpc(data_dir, output_name, max_iter, learning_rate, num_classes, eval_period, checkpoint_period,
                          hpc_host, hpc_user, hpc_base_dir, local_output_dir,
                          cleanup_remote, monitor_interval))
  }
}

#' Train model locally using available hardware
#' @keywords internal
train_model_local <- function(data_dir, output_name, max_iter, learning_rate, num_classes, device, eval_period, checkpoint_period, local_output_dir) {

  output_dir <- file.path(local_output_dir, output_name, fsep = "/")
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  cat("🔬 Starting local training...\n")
  cat("- Data directory:", data_dir, "\n")
  cat("- Output directory:", output_dir, "\n")
  cat("- Max iterations:", max_iter, "\n")

  # Get the Python executable that reticulate is using
  python_exe <- reticulate::py_config()$python

  # Build python command using reticulate's Python
  python_cmd <- paste(
    shQuote(python_exe), "src/train.py",
    "--dataset-name", paste0(output_name, "_train"),
    "--annotation-json", file.path(data_dir, "train", "_annotations.coco.json", fsep = "/"),
    "--image-root", file.path(data_dir, "train", fsep = "/"),
    "--val-annotation-json", file.path(data_dir, "val", "_annotations.coco.json", fsep = "/"),
    "--val-image-root", file.path(data_dir, "val", fsep = "/"),
    "--output-dir", output_dir,
    "--num-classes", num_classes,
    "--device", device,
    "--max-iter", max_iter,
    "--learning-rate", learning_rate,
    "--eval-period", eval_period,
    "--checkpoint-period", checkpoint_period  # 0 gets converted to 999999 in train.py
  )

  # Execute training using reticulate's Python environment
  cat("🚀 Using Python:", python_exe, "\n")
  cat("🚀 Running training command:\n", python_cmd, "\n")
  result <- system(python_cmd, wait = TRUE)

  if (result != 0) {
    stop("Training failed with exit code: ", result)
  }

  cat("✅ Local training completed successfully!\n")
  cat("📁 Model saved to:", output_dir, "\n")

  return(output_dir)
}

#' Train model on HPC using SLURM
#' @keywords internal
train_model_hpc <- function(data_dir, output_name, max_iter, learning_rate, num_classes, eval_period, checkpoint_period,
                           hpc_host, hpc_user, hpc_base_dir, local_output_dir,
                           cleanup_remote, monitor_interval) {

  if (is.null(hpc_base_dir)) {
    stop("Missing `hpc_base_dir`: please specify the base path for training files on your HPC system.")
  }

  # Check SSH connectivity
  if (!test_ssh_connection(hpc_host, hpc_user)) {
    stop("Cannot connect to HPC host: ", hpc_host)
  }

  cat("🔗 Connected to HPC:", hpc_host, "\n")

  # Setup remote directories
  remote_session_dir <- setup_remote_directories(hpc_host, hpc_user, hpc_base_dir, output_name)

  # Sync data to HPC
  cat("📤 Syncing data to HPC...\n")
  sync_data_to_hpc(data_dir, hpc_host, hpc_user, remote_session_dir)

  # Sync code to HPC
  cat("📤 Syncing training code to HPC...\n")
  sync_code_to_hpc(hpc_host, hpc_user, remote_session_dir)

  # Generate and submit SLURM job
  cat("🎯 Generating and submitting SLURM job...\n")
  job_id <- submit_slurm_job(hpc_host, hpc_user, remote_session_dir, output_name,
                            max_iter, learning_rate, num_classes, eval_period, checkpoint_period)

  cat("🔄 Job submitted with ID:", job_id, "\n")
  cat("⏱️  Monitoring job progress...\n")

  # Monitor job using future/mirai
  future_result <- future({
    monitor_slurm_job(hpc_host, hpc_user, job_id, monitor_interval, remote_session_dir)
  })

  # Wait for completion
  job_status <- value(future_result)

  if (job_status != "COMPLETED") {
    stop("Training job failed with status: ", job_status)
  }

  cat("✅ Training completed successfully on HPC!\n")

  # Download trained model
  cat("📥 Downloading trained model...\n")
  local_model_dir <- download_trained_model(hpc_host, hpc_user, remote_session_dir,
                                           output_name, local_output_dir)

  # Cleanup remote files if requested
  if (cleanup_remote) {
    cat("🧹 Cleaning up remote files...\n")
    cleanup_remote_session(hpc_host, hpc_user, remote_session_dir)
  }

  cat("🎉 HPC training pipeline completed!\n")
  cat("📁 Model saved to:", local_model_dir, "\n")

  return(local_model_dir)
}