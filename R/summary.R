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