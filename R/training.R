# ============================================================================
# Model Training Functions
# ============================================================================

#' Train a new petrography detection model
#' @param data_dir Local directory containing train and val subdirectories with COCO annotations
#' @param output_name Name for the trained model (default: "petrography_model")
#' @param max_iter Maximum training iterations (default: 5000)
#' @param learning_rate Learning rate for training (default: 0.001)
#' @param num_classes Number of object classes (default: 5)
#' @param device Device for local training: 'cpu', 'cuda', 'mps' (default: 'cpu')
#' @param eval_period Validation evaluation frequency in iterations (default: 500)
#' @param checkpoint_period Checkpoint saving frequency (0=final only, >0=every N iterations, default: 0)
#' @param hpc_host SSH hostname for HPC training (NULL for local training)
#' @param hpc_user Username for HPC (default: current user)
#' @param hpc_base_dir Remote base directory on HPC (default: "~/petrography_training")
#' @param local_output_dir Local directory to save trained model (default: "Detectron2_Models")
#' @param rsync_mode Data sync mode: 'update' (default) or 'mirror' (adds --delete)
#' @param dry_run If TRUE, rsync runs with -n to preview changes
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
                       rsync_mode = c("update", "mirror"),
                       dry_run = FALSE) {
  
  cli::cli_h1("Model Training")
  training_mode <- if(is.null(hpc_host)) "Local" else paste0("HPC (", hpc_host, ")")
  cli::cli_dl(c(
    "Model name" = output_name,
    "Mode" = training_mode,
    "Max iterations" = max_iter,
    "Learning rate" = learning_rate,
    "Classes" = num_classes
  ))

  start_time <- Sys.time()

  # Validate inputs
  if (!fs::dir_exists(data_dir)) {
    cli::cli_abort("Data directory not found: {.path {data_dir}}")
  }

  # Normalize important paths early
  data_dir <- fs::path_abs(fs::path_norm(data_dir))
  local_output_dir <- fs::path_abs(fs::path_norm(local_output_dir))

  # Validate output name for safety and portability
  if (!grepl("^[A-Za-z0-9._-]{1,64}$", output_name)) {
    cli::cli_abort("Invalid output_name. Use only letters, numbers, ., _, - (max 64 chars)")
  }

  train_dir <- fs::path(data_dir, "train")
  val_dir <- fs::path(data_dir, "val")

  if (!fs::dir_exists(train_dir) || !fs::dir_exists(val_dir)) {
    cli::cli_abort("Data directory must contain 'train' and 'val' subdirectories")
  }

  if (!fs::file_exists(fs::path(train_dir, "_annotations.coco.json"))) {
    cli::cli_abort("Missing COCO annotations in train directory")
  }

  if (!fs::file_exists(fs::path(val_dir, "_annotations.coco.json"))) {
    cli::cli_abort("Missing COCO annotations in val directory")
  }

  # Determine training mode
  if (is.null(hpc_host)) {
    result <- train_model_local(data_dir, output_name, max_iter, learning_rate, num_classes, device, eval_period, checkpoint_period, local_output_dir)
  } else {
    rsync_mode <- match.arg(rsync_mode)
    result <- train_model_hpc(data_dir, output_name, max_iter, learning_rate, num_classes, eval_period, checkpoint_period,
                          hpc_host, hpc_user, hpc_base_dir, local_output_dir,
                          rsync_mode, dry_run)
  }
  
  duration <- difftime(Sys.time(), start_time, units = "mins")
  cli::cli_alert_success("Training completed in {round(duration, 1)} minutes")
  cli::cli_alert_info("Model saved to: {.path {result}}")
  
  return(result)
}

#' Train model locally using available hardware
#' @keywords internal
train_model_local <- function(data_dir, output_name, max_iter, learning_rate, num_classes, device, eval_period, checkpoint_period, local_output_dir) {

  output_dir <- fs::path(local_output_dir, output_name)
  fs::dir_create(output_dir)

  cli::cli_h2("Starting Local Training")
  cli::cli_dl(c(
    "Data directory" = data_dir,
    "Output directory" = output_dir,
    "Max iterations" = max_iter,
    "Device" = device
  ))

  # Get the Python executable that reticulate is using
  python_exe <- reticulate::py_config()$python

  # Resolve packaged training script (inst/python/train.py)
  train_script <- system.file("python", "train.py", package = "petrographer")

  # Build argument vector to avoid shell quoting issues (spaces in paths)
  args <- c(
    train_script,
    "--dataset-name", paste0(output_name, "_train"),
    "--annotation-json", fs::path(data_dir, "train", "_annotations.coco.json"),
    "--image-root", fs::path(data_dir, "train"),
    "--val-annotation-json", fs::path(data_dir, "val", "_annotations.coco.json"),
    "--val-image-root", fs::path(data_dir, "val"),
    "--output-dir", output_dir,
    "--num-classes", as.character(num_classes),
    "--device", device,
    "--max-iter", as.character(max_iter),
    "--learning-rate", as.character(learning_rate),
    "--eval-period", as.character(eval_period),
    "--checkpoint-period", as.character(checkpoint_period)
  )

  # Pretty command for display
  display_cmd <- paste(
    shQuote(python_exe),
    paste(vapply(args, shQuote, character(1)), collapse = " ")
  )

  # Execute training using reticulate's Python environment
  cli::cli_alert_info("Using Python: {.path {python_exe}}")
  cli::cli_alert_info("Running training command")
  cli::cli_code(display_cmd)

  res <- processx::run(python_exe, args = args, echo = TRUE, echo_cmd = FALSE, error_on_status = FALSE)

  if (!identical(res$status, 0L)) {
    cli::cli_abort("Training failed with exit code: {res$status}")
  }

  cli::cli_alert_success("Local training completed successfully!")
  cli::cli_alert_info("Model saved to: {.path {output_dir}}")

  return(output_dir)
}

#' Train model on HPC using SLURM
#' @keywords internal
train_model_hpc <- function(data_dir, output_name, max_iter, learning_rate, num_classes, eval_period, checkpoint_period,
                           hpc_host, hpc_user, hpc_base_dir, local_output_dir,
                           rsync_mode, dry_run) {

  if (is.null(hpc_base_dir)) {
    cli::cli_abort("Missing `hpc_base_dir`: please specify the base path for training files on your HPC system.")
  }

  # Build training parameters
  training_params <- c(
    "--dataset-name", paste0(output_name, "_train"),
    "--annotation-json", "data/train/_annotations.coco.json",
    "--image-root", "data/train",
    "--val-annotation-json", "data/val/_annotations.coco.json", 
    "--val-image-root", "data/val",
    "--output-dir", "output",
    "--num-classes", as.character(num_classes),
    "--max-iter", as.character(max_iter),
    "--learning-rate", as.character(learning_rate),
    "--eval-period", as.character(eval_period),
    "--checkpoint-period", as.character(checkpoint_period),
    "--device", "cuda"
  )

  # Execute HPC workflow
  cli::cli_alert_info("Connecting to HPC: {hpc_host}")
  session <- hpc_session(hpc_host, hpc_user)
  
  cli::cli_alert_info("Uploading data and submitting job...")
  job_info <- hpc_sync_and_submit(session, data_dir, hpc_base_dir, output_name, training_params)
  
  hpc_monitor(session, job_info$job_id, job_info$remote_base)
  
  result <- hpc_download(session, job_info$remote_base, output_name, local_output_dir)
  
  ssh::ssh_disconnect(session)
  cli::cli_alert_success("HPC training pipeline completed!")
  
  return(result)
}
