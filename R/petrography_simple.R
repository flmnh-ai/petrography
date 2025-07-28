# ============================================================================
# Required Libraries for HPC Training
# ============================================================================

#' Check and load required libraries for training functionality
#' @keywords internal
check_training_dependencies <- function() {
  required_packages <- c("future", "reticulate", "tidyverse")

  missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

  if (length(missing_packages) > 0) {
    stop("Missing required packages for training: ", paste(missing_packages, collapse = ", "),
         "\nInstall with: install.packages(c('", paste(missing_packages, collapse = "', '"), "'))")
  }

  # Load future for async operations
  if (!require(future, quietly = TRUE)) {
    stop("Failed to load future package")
  }

  return(invisible(TRUE))
}

# ============================================================================
# Model Loading and Management
# ============================================================================

#' Load petrography detection model
#' @param model_path Path to trained model weights (default: 'Detectron2_Models/model_final.pth')
#' @param config_path Path to model config (default: 'Detectron2_Models/config.yaml')
#' @param confidence Confidence threshold (default: 0.5)
#' @param device Device to use: 'cpu', 'cuda', 'mps' (default: 'mps')
#' @return PetrographyModel object
#' @export
load_model <- function(model_path = "Detectron2_Models/model_final.pth",
                       config_path = "Detectron2_Models/config.yaml",
                       confidence = 0.5,
                       device = "cpu") {

  # Load SAHI model
  sahi_model <- sahi$AutoDetectionModel$from_pretrained(
    model_type = 'detectron2',
    model_path = model_path,
    config_path = config_path,
    confidence_threshold = confidence,
    device = device
  )

  # Create custom wrapper object
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

    # Save to CSV
    write_csv(training_data, file.path(output_dir, "training_metrics.csv"))

  } else if (file.exists(log_file)) {
    # Could add log parsing here if needed
    warning("Only log.txt found - metrics.json preferred for analysis")
  }

  # Generate simple summary
  summary <- list(
    total_iterations = if (nrow(training_data) > 0) max(training_data$iteration, na.rm = TRUE) else 0,
    metrics_available = nrow(training_data) > 0
  )

  list(
    training_data = training_data,
    summary = summary,
    output_dir = output_dir
  )
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
                       hpc_host = NULL,
                       hpc_user = Sys.getenv("USER"),
                       hpc_base_dir = "~/petrography_training",
                       local_output_dir = "Detectron2_Models",
                       cleanup_remote = TRUE,
                       monitor_interval = 30) {

  # Check dependencies
  check_training_dependencies()

  # Validate inputs
  if (!dir.exists(data_dir)) {
    stop("Data directory not found: ", data_dir)
  }

  train_dir <- file.path(data_dir, "train")
  val_dir <- file.path(data_dir, "val")

  if (!dir.exists(train_dir) || !dir.exists(val_dir)) {
    stop("Data directory must contain 'train' and 'val' subdirectories")
  }

  if (!file.exists(file.path(train_dir, "_annotations.coco.json"))) {
    stop("Missing COCO annotations in train directory")
  }

  if (!file.exists(file.path(val_dir, "_annotations.coco.json"))) {
    stop("Missing COCO annotations in val directory")
  }

  # Determine training mode
  if (is.null(hpc_host)) {
    return(train_model_local(data_dir, output_name, max_iter, num_classes, device, local_output_dir))
  } else {
    return(train_model_hpc(data_dir, output_name, max_iter, num_classes,
                          hpc_host, hpc_user, hpc_base_dir, local_output_dir,
                          cleanup_remote, monitor_interval))
  }
}

#' Train model locally using available hardware
#' @keywords internal
train_model_local <- function(data_dir, output_name, max_iter, num_classes, device, local_output_dir) {

  output_dir <- file.path(local_output_dir, output_name)
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
    "--annotation-json", file.path(data_dir, "train", "_annotations.coco.json"),
    "--image-root", file.path(data_dir, "train"),
    "--val-annotation-json", file.path(data_dir, "val", "_annotations.coco.json"),
    "--val-image-root", file.path(data_dir, "val"),
    "--output-dir", output_dir,
    "--num-classes", num_classes,
    "--opts SOLVER.MAX_ITER", max_iter
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
train_model_hpc <- function(data_dir, output_name, max_iter, num_classes,
                           hpc_host, hpc_user, hpc_base_dir, local_output_dir,
                           cleanup_remote, monitor_interval) {

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
                            max_iter, num_classes)

  cat("🔄 Job submitted with ID:", job_id, "\n")
  cat("⏱️  Monitoring job progress...\n")

  # Monitor job using future/mirai
  future_result <- future({
    monitor_slurm_job(hpc_host, hpc_user, job_id, monitor_interval)
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
test_ssh_connection <- function(hpc_host, hpc_user) {
  tryCatch({
    result <- system(paste("ssh -o BatchMode=yes -o ConnectTimeout=5",
                          paste0(hpc_user, "@", hpc_host), "echo 'connection_test'"),
                    intern = TRUE, ignore.stderr = TRUE)
    return(length(result) > 0 && result[1] == "connection_test")
  }, error = function(e) {
    return(FALSE)
  })
}

#' Setup remote directory structure on HPC
#' @keywords internal
setup_remote_directories <- function(hpc_host, hpc_user, hpc_base_dir, output_name) {
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  remote_session_dir <- file.path(hpc_base_dir, paste0(output_name, "_", timestamp))

  # Create remote directories
  ssh_cmd <- paste("ssh", paste0(hpc_user, "@", hpc_host),
                   shQuote(paste("mkdir -p", remote_session_dir,
                                file.path(remote_session_dir, "data"),
                                file.path(remote_session_dir, "src"))))

  result <- system(ssh_cmd, ignore.stderr = TRUE)
  if (result != 0) {
    stop("Failed to create remote directories")
  }

  return(remote_session_dir)
}

#' Sync local data to HPC using rsync
#' @keywords internal
sync_data_to_hpc <- function(local_data_dir, hpc_host, hpc_user, remote_session_dir) {
  remote_data_dir <- file.path(remote_session_dir, "data", basename(local_data_dir))

  rsync_cmd <- paste("rsync -avz --progress",
                     paste0(local_data_dir, "/"),
                     paste0(hpc_user, "@", hpc_host, ":", remote_data_dir, "/"))

  result <- system(rsync_cmd)
  if (result != 0) {
    stop("Failed to sync data to HPC")
  }
}

#' Sync training code to HPC
#' @keywords internal
sync_code_to_hpc <- function(hpc_host, hpc_user, remote_session_dir) {

  # Copy Python training script
  rsync_cmd <- paste("rsync -avz", "src/train.py",
                     paste0(hpc_user, "@", hpc_host, ":", file.path(remote_session_dir, "src/")))

  result <- system(rsync_cmd)
  if (result != 0) {
    stop("Failed to sync training code to HPC")
  }
}

#' Generate and submit SLURM job for training
#' @keywords internal
submit_slurm_job <- function(hpc_host, hpc_user, remote_session_dir, output_name,
                            max_iter, num_classes) {

  # Generate SLURM script based on template
  slurm_script <- generate_slurm_script(remote_session_dir, output_name, max_iter, num_classes)

  # Write script to remote location
  script_path <- file.path(remote_session_dir, "train_job.sh")

  # Transfer script to HPC
  temp_script <- tempfile(fileext = ".sh")
  writeLines(slurm_script, temp_script)

  rsync_cmd <- paste("rsync -avz", temp_script,
                     paste0(hpc_user, "@", hpc_host, ":", script_path))

  result <- system(rsync_cmd)
  if (result != 0) {
    stop("Failed to transfer SLURM script to HPC")
  }

  unlink(temp_script)

  # Submit job and capture job ID
  ssh_cmd <- paste("ssh", paste0(hpc_user, "@", hpc_host),
                   shQuote(paste("cd", remote_session_dir, "&& sbatch train_job.sh")))

  result <- system(ssh_cmd, intern = TRUE)

  # Extract job ID from sbatch output
  if (length(result) == 0 || !grepl("Submitted batch job", result[1])) {
    stop("Failed to submit SLURM job: ", paste(result, collapse = "\n"))
  }

  job_id <- gsub("Submitted batch job ([0-9]+)", "\\1", result[1])
  return(job_id)
}

#' Generate SLURM script content
#' @keywords internal
generate_slurm_script <- function(remote_session_dir, output_name, max_iter, num_classes) {

  data_dir <- file.path(remote_session_dir, "data")
  output_dir <- file.path(remote_session_dir, "output")

  script_lines <- c(
    "#!/bin/bash",
    "#SBATCH --job-name=petrography_train",
    "#SBATCH --output=%x_%j.out",
    "#SBATCH --error=%x_%j.err",
    "#SBATCH --mail-type=END,FAIL",
    "#SBATCH --mail-user=$USER@ufl.edu",
    "#SBATCH --time=04:00:00",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks=1",
    "#SBATCH --cpus-per-task=4",
    "#SBATCH --mem=32gb",
    "#SBATCH --partition=gpu",
    "#SBATCH --gpus=1",
    "",
    "# Load modules",
    "module load conda",
    "module load cuda/11.8",
    "",
    "# Activate conda environment",
    "conda activate detectron2",
    "",
    "# Create output directory",
    paste("mkdir -p", output_dir),
    "",
    "# Run training",
    paste("cd", remote_session_dir),
    paste("python src/train.py \\"),
    paste("  --dataset-name", paste0(output_name, "_train"), "\\"),
    paste("  --annotation-json", file.path(data_dir, "train", "_annotations.coco.json"), "\\"),
    paste("  --image-root", file.path(data_dir, "train"), "\\"),
    paste("  --val-annotation-json", file.path(data_dir, "val", "_annotations.coco.json"), "\\"),
    paste("  --val-image-root", file.path(data_dir, "val"), "\\"),
    paste("  --output-dir", output_dir, "\\"),
    paste("  --num-classes", num_classes, "\\"),
    paste("  --opts SOLVER.MAX_ITER", max_iter),
    "",
    "echo \"Training completed with exit code: $?\""
  )

  return(script_lines)
}

#' Monitor SLURM job status until completion
#' @keywords internal
monitor_slurm_job <- function(hpc_host, hpc_user, job_id, monitor_interval) {

  while (TRUE) {
    # Check job status
    ssh_cmd <- paste("ssh", paste0(hpc_user, "@", hpc_host),
                     shQuote(paste("squeue -j", job_id, "-h -o %T")))

    status_result <- tryCatch({
      system(ssh_cmd, intern = TRUE, ignore.stderr = TRUE)
    }, error = function(e) {
      character(0)
    })

    if (length(status_result) == 0) {
      # Job not found in queue, check if it completed
      ssh_cmd <- paste("ssh", paste0(hpc_user, "@", hpc_host),
                       shQuote(paste("sacct -j", job_id, "-n -o State | tail -n 1")))

      final_status <- tryCatch({
        system(ssh_cmd, intern = TRUE, ignore.stderr = TRUE)
      }, error = function(e) {
        "UNKNOWN"
      })

      if (length(final_status) > 0) {
        final_status <- trimws(final_status[1])
        cat("🏁 Job", job_id, "final status:", final_status, "\n")
        return(final_status)
      } else {
        return("UNKNOWN")
      }
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
download_trained_model <- function(hpc_host, hpc_user, remote_session_dir,
                                  output_name, local_output_dir) {

  local_model_dir <- file.path(local_output_dir, output_name)
  dir.create(local_model_dir, showWarnings = FALSE, recursive = TRUE)

  remote_output_dir <- file.path(remote_session_dir, "output")

  # Download all files from remote output directory
  rsync_cmd <- paste("rsync -avz --progress",
                     paste0(hpc_user, "@", hpc_host, ":", remote_output_dir, "/"),
                     paste0(local_model_dir, "/"))

  result <- system(rsync_cmd)
  if (result != 0) {
    stop("Failed to download trained model from HPC")
  }

  return(local_model_dir)
}

#' Clean up remote session directory
#' @keywords internal
cleanup_remote_session <- function(hpc_host, hpc_user, remote_session_dir) {

  ssh_cmd <- paste("ssh", paste0(hpc_user, "@", hpc_host),
                   shQuote(paste("rm -rf", remote_session_dir)))

  result <- system(ssh_cmd, ignore.stderr = TRUE)
  if (result != 0) {
    warning("Failed to cleanup remote session directory: ", remote_session_dir)
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
