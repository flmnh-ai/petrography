# ============================================================================
# Model Loading and Management
# ============================================================================

#' Load petrography detection model
#' @param model_path Path to trained model weights (default: 'Detectron2_Models/model_final.pth')
#' @param config_path Path to model config (default: 'Detectron2_Models/config.yaml')
#' @param confidence Confidence threshold (default: 0.5)
#' @param device Device to use: 'cpu', 'cuda', 'mps' (default: 'cpu')
#' @return PetrographyModel object
#' @export
load_model <- function(model_path = NULL,
                       config_path = NULL,
                       confidence = 0.5,
                       device = "cpu") {

  cache <- get_model_cache_dir()
  default_model <- file.path(cache, "model_final.pth")
  default_config <- file.path(cache, "config.yaml")

  if (is.null(model_path)) model_path <- default_model
  if (is.null(config_path)) config_path <- default_config

  if (!file.exists(model_path) || !file.exists(config_path)) {
    message("Model files not found. Downloading...")
    download_model()
  }

  sahi_model <- sahi$AutoDetectionModel$from_pretrained(
    model_type = 'detectron2',
    model_path = model_path,
    config_path = config_path,
    confidence_threshold = confidence,
    device = device
  )

  model <- list(
    sahi_model = sahi_model,
    model_path = model_path,
    config_path = config_path,
    confidence = confidence,
    device = device
  )
  class(model) <- "PetrographyModel"
  return(model)
}


get_model_cache_dir <- function() {
  tools::R_user_dir("petrography", which = "cache")
}


download_model <- function(force = FALSE) {
  cache_dir <- get_model_cache_dir()
  dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

  model_url <- "https://www.dropbox.com/scl/fi/3ilo6msi7r1d9fmfn1zq2/model_final.pth?rlkey=6x2ielfy0fr7kijkysa0i3b3l&st=wbfz9k50&dl=1"
  config_url <- "https://www.dropbox.com/scl/fi/kjlggms8k1x4ghhjiph39/config.yaml?rlkey=8lqiu9eeh6xtjcoj2v7ksyb3k&st=haqn63up&dl=1"

  model_path <- file.path(cache_dir, "model_final.pth")
  config_path <- file.path(cache_dir, "config.yaml")

  if (!file.exists(model_path) || force) {
    message("Downloading model weights...")
    download.file(model_url, model_path, mode = "wb")
    message("Model weights saved to: ", model_path)
  } else {
    message("Model weights already present at: ", model_path)
  }

  if (!file.exists(config_path) || force) {
    message("Downloading model config...")
    download.file(config_url, config_path, mode = "wb")
    message("Model config saved to: ", config_path)
  } else {
    message("Model config already present at: ", config_path)
  }

  return(list(model_path = model_path, config_path = config_path))
}

clear_model_cache <- function() {
  cache_dir <- get_model_cache_dir()
  if (dir.exists(cache_dir)) {
    unlink(cache_dir, recursive = TRUE)
    message("Cleared model cache at: ", cache_dir)
  } else {
    message("No model cache found at: ", cache_dir)
  }
}



# ============================================================================
# Helper Functions for Direct Reticulate Calls
# ============================================================================

#' Calculate morphological properties from SAHI result using direct reticulate
#' @param result SAHI prediction result object
#' @param image_path Original image path
#' @return Tibble with morphological properties
calculate_morphology_from_result <- function(result, image_path) {

  # Extract masks from predictions
  predictions <- result$object_prediction_list

  if (length(predictions) == 0) {
    return(tibble())
  }

  # Convert predictions to masks and calculate properties
  morphology_list <- list()

  for (i in seq_along(predictions)) {
    pred <- predictions[[i]]

    # Get mask
    mask <- pred$mask$bool_mask

    # Use skimage to calculate region properties
    labeled_mask <- skimage$measure$label(mask)
    storage.mode(labeled_mask) <- 'integer'
    props <- skimage$measure$regionprops(labeled_mask)

    if (length(props) == 0) {
      stop("scikit-image could not extract region properties from mask - this indicates a processing error")
    }

    prop <- props[[1]]  # Should only be one region per prediction

    morphology_list[[i]] <- list(
      class_id = pred$category$id,
      class_name = pred$category$name,
      confidence = pred$score$value,
      area = prop$area,
      perimeter = prop$perimeter,
      centroid_x = prop$centroid[[1]],
      centroid_y = prop$centroid[[2]],
      eccentricity = prop$eccentricity,
      orientation = prop$orientation,
      major_axis_length = prop$major_axis_length,
      minor_axis_length = prop$minor_axis_length,
      circularity = (4 * pi * prop$area) / (prop$perimeter^2),
      aspect_ratio = prop$major_axis_length / prop$minor_axis_length,
      solidity = prop$solidity,
      extent = prop$extent
    )
  }

  # Convert to tibble
  morphology_list %>%
    map_dfr(as_tibble) %>%
    mutate(image_name = basename(image_path))
}

# ============================================================================
# Core Prediction Functions - Using Direct Reticulate Calls
# ============================================================================

#' Predict objects in a single image using loaded model
#' @param image_path Path to image file
#' @param model PetrographyModel object from load_model()
#' @param use_slicing Whether to use SAHI sliced inference (default: TRUE)
#' @param slice_size Size of slices for SAHI in pixels (default: 512)
#' @param overlap Overlap ratio between slices (default: 0.2)
#' @param output_dir Output directory (auto-generated if NULL)
#' @param save_visualization Whether to save prediction visualization (default: TRUE)
#' @return Tibble with detection results and morphological properties
#' @export
predict_image <- function(image_path, model, use_slicing = TRUE,
                         slice_size = 512, overlap = 0.2, output_dir = NULL,
                         save_visualization = TRUE) {

  # Validate inputs
  if (!file.exists(image_path)) {
    stop("Image file not found: ", image_path)
  }

  if (!inherits(model, "PetrographyModel")) {
    stop("model must be a PetrographyModel object from load_model()")
  }

  # Set up output directory
  if (is.null(output_dir)) {
    output_dir <- file.path("results", tools::file_path_sans_ext(basename(image_path)))
  }

  if (save_visualization) {
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  }

  # Run SAHI prediction
  if (use_slicing) {
    result <- sahi$predict$get_sliced_prediction(
      image = image_path,
      detection_model = model$sahi_model,
      slice_height = as.integer(slice_size),
      slice_width = as.integer(slice_size),
      overlap_height_ratio = overlap,
      overlap_width_ratio = overlap
    )
  } else {
    result <- sahi$predict$get_prediction(
      image = image_path,
      detection_model = model$sahi_model
    )
  }

  # Check if any objects detected
  if (length(result$object_prediction_list) == 0) {
    return(tibble())
  }

  # Save visualization if requested
  if (save_visualization) {
    image_name <- tools::file_path_sans_ext(basename(image_path))
    result$export_visuals(
      export_dir = output_dir,
      file_name = paste0(image_name, "_prediction"),
      hide_conf = TRUE,
      rect_th = 2L
    )
  }
  # Calculate morphological properties and return formatted tibble
  calculate_morphology_from_result(result, image_path) %>%
    clean_names() %>%
    enhance_results()
}

#' Predict objects in multiple images using SAHI native batch prediction
#' @param input_dir Directory containing images
#' @param model PetrographyModel object from load_model()
#' @param use_slicing Whether to use SAHI sliced inference (default: TRUE)
#' @param slice_size Size of slices for SAHI in pixels (default: 512)
#' @param overlap Overlap ratio between slices (default: 0.2)
#' @param output_dir Output directory (default: 'results/batch')
#' @param save_visualizations Whether to save prediction visualizations (default: TRUE)
#' @return Tibble with detection results for all images
#' @export
predict_batch <- function(input_dir, model, use_slicing = TRUE,
                         slice_size = 512, overlap = 0.2,
                         output_dir = "results/batch",
                         save_visualizations = TRUE) {

  # Validate inputs
  if (!dir.exists(input_dir)) {
    stop("Input directory not found: ", input_dir)
  }

  if (!inherits(model, "PetrographyModel")) {
    stop("model must be a PetrographyModel object from load_model()")
  }

  # Create output directory
  if (save_visualizations) {
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  }

  # Use SAHI's native batch prediction - much more efficient!
  result <- sahi$predict$predict(
    model_type = 'detectron2',
    model_path = model$model_path,
    model_config_path = model$config_path,
    model_confidence_threshold = model$confidence,
    model_device = model$device,
    source = input_dir,
    no_standard_prediction = use_slicing,  # If using slicing, disable standard
    no_sliced_prediction = !use_slicing,   # If not using slicing, disable sliced
    slice_height = as.integer(slice_size),
    slice_width = as.integer(slice_size),
    overlap_height_ratio = overlap,
    overlap_width_ratio = overlap,
    export_pickle = FALSE,
    export_crop = FALSE,
    export_visuals = save_visualizations,
    export_dir = if (save_visualizations) output_dir else NULL
  )

  # Extract all predictions from batch result
  all_predictions <- list()

  for (i in seq_along(result$object_prediction_list)) {
    pred_result <- result$object_prediction_list[[i]]
    image_path <- pred_result$image$file_name

    if (length(pred_result$object_prediction_list) > 0) {
      # Create a temporary result object for morphology calculation
      temp_result <- list(object_prediction_list = pred_result$object_prediction_list)

      # Calculate morphology for this image
      morph_data <- calculate_morphology_from_result(temp_result, image_path)
      all_predictions[[length(all_predictions) + 1]] <- morph_data
    }
  }

  # Combine all results
  if (length(all_predictions) > 0) {
    combined_results <- map_dfr(all_predictions, identity) %>%
      clean_names() %>%
      enhance_results()
    return(combined_results)
  } else {
    return(tibble())
  }
}

#' Evaluate model training
#' @param model_dir Directory containing trained model (default: 'Detectron2_Models')
#' @param device Device to use for evaluation (default: 'cpu')
#' @param output_dir Output directory for results (default: 'results/evaluation')
#' @return List with training data tibble and summary statistics
#' @export
evaluate_training <- function(model_dir = "Detectron2_Models", device = "cpu",
                             output_dir = "results/evaluation") {

  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # Look for training metrics
  metrics_file <- file.path(model_dir, "metrics.json")
  log_file <- file.path(model_dir, "log.txt")

  training_data <- tibble()

  if (file.exists(metrics_file)) {
    # Read metrics.json using pandas
    pd <- import("pandas")

    # Read JSON lines file
    df <- pd$read_json(metrics_file, lines = TRUE)

    # Convert to R tibble
    training_data <- py_to_r(df) %>%
      as_tibble() %>%
      clean_names()

    # Separate training and validation metrics
    training_metrics <- training_data %>%
      select(-contains("bbox")) %>%  # Remove bbox validation metrics for cleaner view
      filter(!is.na(iteration))

    validation_metrics <- training_data %>%
      select(iteration, contains("bbox")) %>%
      filter(!is.na(iteration), if_any(contains("bbox"), ~ !is.na(.)))

    # Save to CSV files
    write_csv(training_metrics, file.path(output_dir, "training_metrics.csv"))
    if (nrow(validation_metrics) > 0) {
      write_csv(validation_metrics, file.path(output_dir, "validation_metrics.csv"))
    }

    # Update training_data to include both
    training_data <- training_metrics

  } else if (file.exists(log_file)) {
    # Could add log parsing here if needed
    warning("Only log.txt found - metrics.json preferred for analysis")
  }

  # Generate enhanced summary
  summary <- list(
    total_iterations = if (nrow(training_data) > 0) max(training_data$iteration, na.rm = TRUE) else 0,
    metrics_available = nrow(training_data) > 0,
    validation_metrics_available = exists("validation_metrics") && nrow(validation_metrics) > 0,
    validation_evaluations = if (exists("validation_metrics")) nrow(validation_metrics) else 0
  )

  result <- list(
    training_data = training_data,
    summary = summary,
    output_dir = output_dir
  )

  # Add validation data if available
  if (exists("validation_metrics") && nrow(validation_metrics) > 0) {
    result$validation_data <- validation_metrics
  }

  return(result)
}

# ============================================================================
# Data Enhancement Functions
# ============================================================================

#' Clean column names to consistent snake_case
#' @param .data Data frame
#' @return Data frame with cleaned names
clean_names <- function(.data) {
  names(.data) <- names(.data) %>%
    str_replace_all("-", "_") %>%
    str_replace_all("\\s+", "_") %>%
    str_to_lower()
  .data
}

#' Add useful derived metrics that users would calculate anyway
#' @param .data Data frame with morphological data
#' @return Data frame with additional calculated metrics
enhance_results <- function(.data) {
  if (nrow(.data) == 0) return(.data)

  .data %>%
    mutate(
      # Essential derived metrics
      log_area = log10(area),
      orientation_deg = orientation * 180 / pi,

      # Useful categories
      size_category = case_when(
        area < quantile(area, 0.33, na.rm = TRUE) ~ "small",
        area < quantile(area, 0.67, na.rm = TRUE) ~ "medium",
        TRUE ~ "large"
      ),

      shape_category = case_when(
        circularity > 0.8 ~ "circular",
        aspect_ratio > 2 ~ "elongated",
        eccentricity > 0.8 ~ "eccentric",
        TRUE ~ "irregular"
      )
    )
}

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
    return(train_model_local(data_dir, output_name, max_iter, num_classes, device, eval_period, checkpoint_period, local_output_dir))
  } else {
    return(train_model_hpc(data_dir, output_name, max_iter, num_classes, eval_period, checkpoint_period,
                          hpc_host, hpc_user, hpc_base_dir, local_output_dir,
                          cleanup_remote, monitor_interval))
  }
}

#' Train model locally using available hardware
#' @keywords internal
train_model_local <- function(data_dir, output_name, max_iter, num_classes, device, eval_period, checkpoint_period, local_output_dir) {

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
train_model_hpc <- function(data_dir, output_name, max_iter, num_classes, eval_period, checkpoint_period,
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
                            max_iter, num_classes, eval_period, checkpoint_period)

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
                             max_iter, num_classes, eval_period, checkpoint_period) {
  slurm_script <- generate_slurm_script(remote_session_dir, output_name, max_iter, num_classes, eval_period, checkpoint_period)
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
generate_slurm_script <- function(remote_session_dir, output_name, max_iter, num_classes, eval_period, checkpoint_period) {
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


# ============================================================================
# Summary Functions
# ============================================================================

#' Summarize detections by image
#' @param .data Data frame with detections
#' @return Summary tibble with per-image statistics
#' @export
summarize_by_image <- function(.data) {
  .data %>%
    group_by(image_name) %>%
    summarise(
      n_objects = n(),
      total_area = sum(area, na.rm = TRUE),
      mean_area = mean(area, na.rm = TRUE),
      median_area = median(area, na.rm = TRUE),
      mean_circularity = mean(circularity, na.rm = TRUE),
      mean_eccentricity = mean(eccentricity, na.rm = TRUE),
      area_cv = sd(area, na.rm = TRUE) / mean(area, na.rm = TRUE),
      .groups = "drop"
    )
}

#' Get overall population statistics
#' @param .data Data frame with detections
#' @return Named list of population-level statistics
#' @export
get_population_stats <- function(.data) {
  if (nrow(.data) == 0) {
    return(list(
      total_objects = 0,
      unique_images = 0,
      mean_objects_per_image = 0
    ))
  }

  list(
    total_objects = nrow(.data),
    unique_images = length(unique(.data$image_name)),
    mean_objects_per_image = nrow(.data) / length(unique(.data$image_name)),
    total_area = sum(.data$area, na.rm = TRUE),
    mean_area = mean(.data$area, na.rm = TRUE),
    median_area = median(.data$area, na.rm = TRUE),
    area_range = range(.data$area, na.rm = TRUE),
    mean_circularity = mean(.data$circularity, na.rm = TRUE),
    mean_eccentricity = mean(.data$eccentricity, na.rm = TRUE)
  )
}
