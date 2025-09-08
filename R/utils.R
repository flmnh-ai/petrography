# Small data helpers

clean_names <- function(.data) {
  names(.data) <- names(.data) |>
    stringr::str_replace_all("-", "_") |>
    stringr::str_replace_all("\\s+", "_") |>
    stringr::str_to_lower()
  .data
}

enhance_results <- function(.data) {
  if (nrow(.data) == 0) return(.data)
  .data |>
    dplyr::mutate(
      log_area = log10(area),
      orientation_deg = orientation * 180 / pi,
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

