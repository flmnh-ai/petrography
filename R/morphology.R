# Morphology helpers: SAHI result -> tibble with properties

#' Calculate morphological properties from SAHI result
#'
#' Internal helper converting SAHI object predictions into a tibble of
#' morphological properties using scikit-image via reticulate.
#'
#' @param result SAHI prediction result object.
#' @param image_path Original image path (used to populate `image_name`).
#' @return A tibble with morphological properties per object.
#' @keywords internal
calculate_morphology_from_result <- function(result, image_path) {
  predictions <- result$object_prediction_list
  if (length(predictions) == 0) return(tibble::tibble())

  morphology_list <- vector("list", length(predictions))
  for (i in seq_along(predictions)) {
    pred <- predictions[[i]]
    mask <- pred$mask$bool_mask
    labeled_mask <- skimage$measure$label(mask)
    storage.mode(labeled_mask) <- 'integer'
    props <- skimage$measure$regionprops(labeled_mask)
    if (length(props) == 0) stop("scikit-image could not extract region properties from mask")
    prop <- props[[1]]
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

  morphology_list |>
    purrr::map_dfr(tibble::as_tibble) |>
    dplyr::mutate(image_name = basename(image_path))
}
