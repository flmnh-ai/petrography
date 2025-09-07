# ============================================================================
# Model Training Functions
# ============================================================================

#' Train a new petrography detection model
#' @param data_dir Local directory containing train and val subdirectories with COCO annotations
#' @param output_name Name for the trained model (required)
#' @param max_iter Maximum training iterations (default: 2000)
#' @param learning_rate Base learning rate before auto-scaling (default: 0.00025)
#' @param num_classes Number of object classes (required)
#' @param device Device for local training: 'cpu', 'cuda', 'mps' (default: 'cuda')
#' @param eval_period Validation evaluation frequency in iterations (default: 100)
#' @param checkpoint_period Checkpoint saving frequency (0=final only, >0=every N iterations, default: 0)
#' @param ims_per_batch Total images per batch across all devices. If NA (default),
#'   uses 2 images per GPU (global batch = 2 * gpus).
#' @param auto_scale_lr Automatically scale learning rate by number of GPUs
#'   (effective LR = learning_rate * gpus). Set FALSE to use learning_rate as-is.
#'   (default: TRUE)
#' @param num_workers DataLoader workers per process/GPU. If NULL (default), uses 4 on HPC
#'   (per GPU) and min(4, max(1, parallel::detectCores(logical = FALSE) - 1)) locally.
#' @param freeze_at Backbone freeze level: 0 (none), 2 (default small-data), 3 (freeze more)
#' @param optimizer Optimizer to use: 'SGD' or 'AdamW' (default: 'SGD')
#' @param weight_decay Weight decay (L2/AdamW). If NULL, default is 1e-4 for SGD and 0.05 for AdamW
#' @param backbone_multiplier LR multiplier applied to backbone params when unfrozen (default: 0.1)
#' @param scheduler LR scheduler: 'multistep' (default) or 'cosine'
#' @param warmup_frac Fraction of max_iter used for warmup (default: 0.1). Ignored if warmup_iters provided
#' @param warmup_iters Explicit warmup iters. If NULL, computed from warmup_frac
#' @param step_milestones_frac Milestones as fractions of max_iter for multistep (default: c(0.75, 0.9))
#' @param step_milestones Explicit milestone iters vector (overrides step_milestones_frac if provided)
#' @param gpus Number of GPUs for HPC training (default: 1, ignored for local training)
#' @param hpc_host SSH hostname for HPC training (default: PETROGRAPHER_HPC_HOST env var, or "" for local training)
#' @param hpc_user Username for HPC (default: NULL)
#' @param hpc_base_dir Remote base directory on HPC (default: PETROGRAPHER_HPC_BASE_DIR env var)
#' @param local_output_dir Local directory to save trained model (default: "Detectron2_Models")
#' @param rsync_mode Data sync mode: 'update' (default) or 'mirror' (adds --delete)
#' @return Path to trained model directory
#' @export
train_model <- function(data_dir,
                       output_name,
                       num_classes,
                       max_iter = 2000,
                       learning_rate = 0.0025,
                       device = "cuda",
                       eval_period = 100,
                       checkpoint_period = 0,
                       ims_per_batch = NA,
                       auto_scale_lr = TRUE,
                       num_workers = NULL,
                       freeze_at = 2,
                       optimizer = c("SGD", "AdamW"),
                       weight_decay = NULL,
                       backbone_multiplier = 0.1,
                       scheduler = c("multistep", "cosine"),
                       warmup_frac = 0.1,
                       warmup_iters = NULL,
                       step_milestones_frac = c(0.75, 0.9),
                       step_milestones = NULL,
                       classwise = FALSE,
                       gpus = 1,
                       hpc_host = Sys.getenv("PETROGRAPHER_HPC_HOST", ""),
                       hpc_user = NULL,
                       hpc_base_dir = Sys.getenv("PETROGRAPHER_HPC_BASE_DIR", ""),
                       local_output_dir = here::here("Detectron2_Models"),
                       rsync_mode = c("update", "mirror")) {

  cli::cli_h1("Model Training")
  training_mode <- if(is.null(hpc_host) || hpc_host == "") "Local" else paste0("HPC (", hpc_host, ")")
  cli::cli_h2("Training Configuration")
  # Resolve effective global batch (default 2 per GPU)
  effective_ims <- ims_per_batch
  if (is.na(effective_ims)) {
    effective_ims <- 2L * max(1L, as.integer(gpus))
  }
  if (!is.numeric(effective_ims) || length(effective_ims) != 1 || is.na(effective_ims) || effective_ims < 1) {
    cli::cli_abort("ims_per_batch must be a positive integer or NA for auto (2 per GPU)")
  }
  effective_ims <- as.integer(effective_ims)

  # Normalize choices
  optimizer <- match.arg(optimizer)
  scheduler <- match.arg(scheduler)

  # Compute effective LR for display if auto-scaling by number of GPUs
  eff_lr <- if (isTRUE(auto_scale_lr)) learning_rate * max(1L, as.integer(gpus)) else learning_rate

  details <- c(
    "Model name" = output_name,
    "Mode" = training_mode,
    "Data directory" = as.character(fs::path_abs(fs::path_norm(data_dir))),
    "Local output root" = as.character(fs::path_abs(fs::path_norm(local_output_dir))),
    "Device" = device,
    "Classes" = num_classes,
    "Max iterations" = max_iter,
    "Learning rate (base)" = learning_rate,
    "LR auto-scale" = if (isTRUE(auto_scale_lr)) glue::glue("ON -> {signif(eff_lr, 3)}") else "OFF",
    "Eval period" = eval_period,
    "Checkpoint period" = checkpoint_period,
    "Images per batch" = effective_ims,
    "Freeze level (FREEZE_AT)" = freeze_at,
    "Optimizer" = optimizer,
    "Weight decay" = if (is.null(weight_decay)) if (optimizer == "AdamW") 0.05 else 1e-4 else weight_decay,
    "Backbone LR multiplier" = backbone_multiplier,
    "Scheduler" = scheduler,
    "Classwise metrics" = if (isTRUE(classwise)) "ON" else "OFF"
  )
  if (!is.null(hpc_host) && hpc_host != "") {
    details <- c(details,
      "HPC host" = hpc_host
    )
    if (!is.null(hpc_user) && nzchar(hpc_user)) {
      details <- c(details, "HPC user" = hpc_user)
    }
  }
  cli::cli_dl(details)

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

  # If training on HPC, ensure IMS_PER_BATCH divisible by number of GPUs
  if (!is.null(hpc_host) && hpc_host != "") {
    if (effective_ims %% gpus != 0) {
      cli::cli_abort("ims_per_batch ({ims_per_batch}) must be divisible by gpus ({gpus}) for multi-GPU training")
    }
  }

  # Resolve num_workers: set to per-GPU images by default
  if (is.null(num_workers)) {
    per_gpu <- max(1L, as.integer(effective_ims / max(1L, as.integer(gpus))))
    num_workers <- per_gpu
  }
  if (!is.numeric(num_workers) || length(num_workers) != 1 || is.na(num_workers) || num_workers < 1) {
    cli::cli_abort("num_workers must be a positive integer or NULL")
  }
  num_workers <- as.integer(num_workers)

  # Warmup and milestones
  if (!is.null(warmup_iters)) {
    if (!is.numeric(warmup_iters) || length(warmup_iters) != 1 || warmup_iters < 0) {
      cli::cli_abort("warmup_iters must be a single non-negative number or NULL")
    }
    warmup <- as.integer(round(warmup_iters))
  } else {
    if (!is.numeric(warmup_frac) || warmup_frac < 0 || warmup_frac > 1) {
      cli::cli_abort("warmup_frac must be in [0,1]")
    }
    warmup <- as.integer(round(max_iter * warmup_frac))
  }

  steps <- NULL
  if (identical(scheduler, "multistep")) {
    if (!is.null(step_milestones)) {
      steps <- sort(unique(as.integer(step_milestones)))
    } else if (!is.null(step_milestones_frac)) {
      steps <- sort(unique(as.integer(round(max_iter * step_milestones_frac))))
    }
    # Drop any 0 or >= max_iter
    steps <- steps[steps > 0 & steps < max_iter]
  }

  # Determine training mode
  if (is.null(hpc_host) || hpc_host == "") {
    result <- train_model_local(
      data_dir, output_name, max_iter, eff_lr, num_classes, device, eval_period, checkpoint_period,
      effective_ims, num_workers, freeze_at, optimizer, if (is.null(weight_decay)) if (optimizer == "AdamW") 0.05 else 1e-4 else weight_decay,
      backbone_multiplier, scheduler, warmup, steps, classwise, local_output_dir
    )
  } else {
    result <- train_model_hpc(
      data_dir, output_name, max_iter, eff_lr, num_classes, eval_period, checkpoint_period,
      effective_ims, num_workers, freeze_at, optimizer, if (is.null(weight_decay)) if (optimizer == "AdamW") 0.05 else 1e-4 else weight_decay,
      backbone_multiplier, scheduler, warmup, steps, classwise, gpus, hpc_host, hpc_user, hpc_base_dir, local_output_dir
    )
  }

  duration_mins <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
  cli::cli_alert_success("Training completed in {duration_mins} minute{?s}")
  cli::cli_alert_info("Model saved to: {.path {result}}")

  return(result)
}

#' Train model locally using available hardware
#' @keywords internal
train_model_local <- function(data_dir, output_name, max_iter, learning_rate, num_classes, device, eval_period, checkpoint_period,
                              ims_per_batch, num_workers, freeze_at, optimizer, weight_decay, backbone_multiplier,
                              scheduler, warmup_iters, steps, classwise, local_output_dir) {

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
    "--num-workers", as.character(num_workers),
    "--device", device,
    "--max-iter", as.character(max_iter),
    "--learning-rate", as.character(learning_rate),
    "--eval-period", as.character(eval_period),
    "--checkpoint-period", as.character(checkpoint_period),
    "--ims-per-batch", as.character(ims_per_batch),
    "--freeze-at", as.character(freeze_at),
    "--optimizer", optimizer,
    "--weight-decay", as.character(weight_decay),
    "--backbone-multiplier", as.character(backbone_multiplier),
    "--scheduler", scheduler,
    "--warmup-iters", as.character(warmup_iters),
    if (!is.null(steps) && length(steps) > 0) c("--steps", paste(steps, collapse = ",")) else NULL,
    if (isTRUE(classwise)) "--classwise" else NULL,
    NULL
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
                           ims_per_batch, num_workers, freeze_at, optimizer, weight_decay, backbone_multiplier,
                           scheduler, warmup_iters, steps, classwise, gpus, hpc_host, hpc_user, hpc_base_dir, local_output_dir) {

  if (is.null(hpc_base_dir) || hpc_base_dir == "") {
    cli::cli_abort("Missing `hpc_base_dir`: please specify the base path for training files on your HPC system or set PETROGRAPHER_HPC_BASE_DIR environment variable.")
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
    "--num-workers", as.character(num_workers),
    "--max-iter", as.character(max_iter),
    "--learning-rate", as.character(learning_rate),
    "--eval-period", as.character(eval_period),
    "--checkpoint-period", as.character(checkpoint_period),
    "--ims-per-batch", as.character(ims_per_batch),
    "--freeze-at", as.character(freeze_at),
    "--optimizer", optimizer,
    "--weight-decay", as.character(weight_decay),
    "--backbone-multiplier", as.character(backbone_multiplier),
    "--scheduler", scheduler,
    "--warmup-iters", as.character(warmup_iters),
    if (!is.null(steps) && length(steps) > 0) c("--steps", paste(steps, collapse = ",")) else NULL,
    if (isTRUE(classwise)) "--classwise" else NULL,
    "--device", "cuda",
    "--num-gpus", as.character(gpus)
  )

  # Execute HPC workflow with SSH multiplexing
  target <- hpc_authenticate(hpc_host, hpc_user)

  cli::cli_alert_info("Uploading data and submitting job...")
  job_info <- hpc_sync_and_submit(target, data_dir, hpc_base_dir, output_name, training_params, gpus)

  hpc_monitor(target, job_info$job_id, job_info$remote_base)

  result <- hpc_download(target, job_info$remote_base, output_name, local_output_dir)

  cli::cli_alert_success("HPC training pipeline completed!")

  return(result)
}
