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
    return(tibble::tibble())
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
  morphology_list |>
    purrr::map_dfr(tibble::as_tibble) |>
    dplyr::mutate(image_name = basename(image_path))
}

# ============================================================================
# Data Enhancement Functions
# ============================================================================

#' Clean column names to consistent snake_case
#' @param .data Data frame
#' @return Data frame with cleaned names
clean_names <- function(.data) {
  names(.data) <- names(.data) |>
    stringr::str_replace_all("-", "_") |>
    stringr::str_replace_all("\\s+", "_") |>
    stringr::str_to_lower()
  .data
}

#' Add useful derived metrics that users would calculate anyway
#' @param .data Data frame with morphological data
#' @return Data frame with additional calculated metrics
enhance_results <- function(.data) {
  if (nrow(.data) == 0) return(.data)

  .data |>
    dplyr::mutate(
      # Essential derived metrics
      log_area = log10(area),
      orientation_deg = orientation * 180 / pi,

      # Useful categories
      size_category = dplyr::case_when(
        area < stats::quantile(area, 0.33, na.rm = TRUE) ~ "small",
        area < stats::quantile(area, 0.67, na.rm = TRUE) ~ "medium",
        TRUE ~ "large"
      ),

      shape_category = dplyr::case_when(
        circularity > 0.8 ~ "circular",
        aspect_ratio > 2 ~ "elongated",
        eccentricity > 0.8 ~ "eccentric",
        TRUE ~ "irregular"
      )
    )
}
