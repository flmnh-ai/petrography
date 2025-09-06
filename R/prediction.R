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