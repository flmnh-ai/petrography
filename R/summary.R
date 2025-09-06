# ============================================================================
# Summary Functions
# ============================================================================

#' Summarize detections by image
#' @param .data Data frame with detections
#' @return Summary tibble with per-image statistics
#' @export
summarize_by_image <- function(.data) {
  .data |>
    dplyr::group_by(image_name) |>
    dplyr::summarise(
      n_objects = dplyr::n(),
      total_area = sum(area, na.rm = TRUE),
      mean_area = stats::mean(area, na.rm = TRUE),
      median_area = stats::median(area, na.rm = TRUE),
      mean_circularity = stats::mean(circularity, na.rm = TRUE),
      mean_eccentricity = stats::mean(eccentricity, na.rm = TRUE),
      area_cv = stats::sd(area, na.rm = TRUE) / stats::mean(area, na.rm = TRUE),
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
    mean_area = stats::mean(.data$area, na.rm = TRUE),
    median_area = stats::median(.data$area, na.rm = TRUE),
    area_range = range(.data$area, na.rm = TRUE),
    mean_circularity = stats::mean(.data$circularity, na.rm = TRUE),
    mean_eccentricity = stats::mean(.data$eccentricity, na.rm = TRUE)
  )
}
