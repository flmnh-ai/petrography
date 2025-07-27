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
