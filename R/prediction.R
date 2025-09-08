# ============================================================================
# Core Prediction Functions - Using Direct Reticulate Calls
# ============================================================================

#' Predict objects in a single image
#' @param image_path Path to image file
#' @param model PetrographyModel object from load_model()
#' @param use_slicing Whether to use SAHI sliced inference (default: TRUE)
#' @param slice_size Size of slices for SAHI in pixels (default: 512)
#' @param overlap Overlap ratio between slices (default: 0.2)
#' @param output_dir Output directory (auto-generated if NULL)
#' @param save_visualizations Whether to save prediction visualization (default: TRUE)
#' @return Tibble with detection results and morphological properties
#' @export
predict_image <- function(image_path, model, use_slicing = TRUE,
                         slice_size = 512, overlap = 0.2, output_dir = NULL,
                         save_visualizations = TRUE) {

  # Validate inputs
  if (!fs::file_exists(image_path)) {
    cli::cli_abort("Image file not found: {.path {image_path}}")
  }

  if (!inherits(model, "PetrographyModel")) {
    cli::cli_abort("model must be a PetrographyModel object from load_model()")
  }

  # Set up output directory
  if (is.null(output_dir)) {
    output_dir <- fs::path("results", tools::file_path_sans_ext(fs::path_file(image_path)))
  }

  if (save_visualizations) {
    fs::dir_create(output_dir)
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
    return(tibble::tibble())
  }

  # Save visualization if requested
  if (save_visualizations) {
    image_name <- tools::file_path_sans_ext(basename(image_path))
    result$export_visuals(
      export_dir = output_dir,
      file_name = paste0(image_name, "_prediction"),
      hide_conf = TRUE,
      rect_th = 2L
    )
  }
  # Calculate morphological properties and return formatted tibble
  calculate_morphology_from_result(result, image_path) |>
    clean_names() |>
    enhance_results()
}

#' Predict objects in multiple images (directory)
#' @param input_dir Directory containing images
#' @param model PetrographyModel object from load_model()
#' @param use_slicing Whether to use SAHI sliced inference (default: TRUE)
#' @param slice_size Size of slices for SAHI in pixels (default: 512)
#' @param overlap Overlap ratio between slices (default: 0.2)
#' @param output_dir Output directory (default: 'results/batch')
#' @param save_visualizations Whether to save prediction visualizations (default: TRUE)
#' @return Tibble with detection results for all images
#' @export
predict_images <- function(input_dir, model, use_slicing = TRUE,
                          slice_size = 512, overlap = 0.2,
                          output_dir = "results/batch",
                          save_visualizations = TRUE) {

  # Validate inputs
  if (!fs::dir_exists(input_dir)) {
    cli::cli_abort("Input directory not found: {.path {input_dir}}")
  }

  if (!inherits(model, "PetrographyModel")) {
    cli::cli_abort("model must be a PetrographyModel object from load_model()")
  }

  # Create output directory
  if (save_visualizations) {
    fs::dir_create(output_dir)
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
    combined_results <- purrr::map_dfr(all_predictions, identity) |>
      clean_names() |>
      enhance_results()
    return(combined_results)
  } else {
    return(tibble::tibble())
  }
}

#' Evaluate model training
#' Reads Detectron2 metrics.json and exports:
#' - training_metrics.csv: losses, lr, etc.
#' - validation_metrics.csv: aggregate COCO bbox and segm AP metrics
#' - validation_classwise.csv: per-class AP metrics (when logged by evaluator)
#' @param model_dir Directory containing trained model (default: 'Detectron2_Models')
#' @param output_dir Output directory for results (default: 'results/evaluation')
#' @return List with parsed tibbles and summary statistics
#' @export
evaluate_training <- function(model_dir = "Detectron2_Models",
                             output_dir = "results/evaluation") {
  
  cli::cli_h2("Training Evaluation")
  cli::cli_alert_info("Loading training metrics from: {.path {model_dir}}")

  # Create output directory
  fs::dir_create(output_dir)

  # Look for training metrics
  metrics_file <- fs::path(model_dir, "metrics.json")
  log_file <- fs::path(model_dir, "log.txt")

  parsed <- list(training = tibble::tibble(), validation = tibble::tibble(), classwise = tibble::tibble())
  if (fs::file_exists(metrics_file)) {
    parsed <- parse_metrics(metrics_file)
    # Save to CSV files
    readr::write_csv(parsed$training, fs::path(output_dir, "training_metrics.csv"))
    if (nrow(parsed$validation) > 0) readr::write_csv(parsed$validation, fs::path(output_dir, "validation_metrics.csv"))
    if (nrow(parsed$classwise) > 0) readr::write_csv(parsed$classwise, fs::path(output_dir, "validation_classwise.csv"))
  } else if (fs::file_exists(log_file)) {
    # Could add log parsing here if needed
    warning("Only log.txt found - metrics.json preferred for analysis")
  }

  # Generate enhanced summary
  summary <- list(
    total_iterations = if (nrow(parsed$training) > 0) max(parsed$training$iteration, na.rm = TRUE) else 0,
    metrics_available = nrow(parsed$training) > 0,
    validation_metrics_available = nrow(parsed$validation) > 0,
    validation_segm_available = any(grepl("segm", names(parsed$validation))),
    classwise_available = nrow(parsed$classwise) > 0,
    validation_evaluations = nrow(parsed$validation)
  )

  result <- list(
    training_data = parsed$training,
    summary = summary,
    output_dir = output_dir
  )

  # Add validation data if available
  if (nrow(parsed$validation) > 0) result$validation_data <- parsed$validation
  if (nrow(parsed$classwise) > 0) result$validation_classwise <- parsed$classwise
  
  # Print summary
  cli::cli_dl(c(
    "Training iterations" = summary$total_iterations,
    "Validation evaluations" = summary$validation_evaluations,
    "Segm metrics available" = if (isTRUE(summary$validation_segm_available)) "yes" else "no",
    "Classwise metrics available" = if (isTRUE(summary$classwise_available)) "yes" else "no",
    "Training records" = nrow(training_data),
    "Output directory" = output_dir
  ))
  
  if (nrow(training_data) > 0) {
    final_metrics <- tail(training_data, 1)
    if ("total_loss" %in% names(final_metrics)) {
      cli::cli_alert_info("Final training loss: {round(final_metrics$total_loss, 4)}")
    }
  }

  cli::cli_alert_success("Training evaluation completed")
  return(result)
}
