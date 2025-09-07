# ============================================================================
# Dataset validation and summary helpers
# ============================================================================

#' Validate a COCO-style dataset directory
#' @param data_dir Directory containing 'train' and 'val' subdirectories
#' @return A list with validation flags and counts
#' @export
validate_coco_dataset <- function(data_dir) {
  data_dir <- fs::path_abs(fs::path_norm(data_dir))
  train_dir <- fs::path(data_dir, "train")
  val_dir <- fs::path(data_dir, "val")
  train_ok <- fs::dir_exists(train_dir)
  val_ok <- fs::dir_exists(val_dir)
  train_anno <- fs::file_exists(fs::path(train_dir, "_annotations.coco.json"))
  val_anno <- fs::file_exists(fs::path(val_dir, "_annotations.coco.json"))
  train_images <- if (train_ok) length(fs::dir_ls(train_dir, regexp = "\\.(jpg|jpeg|png)$", ignore_case = TRUE)) else 0
  val_images <- if (val_ok) length(fs::dir_ls(val_dir, regexp = "\\.(jpg|jpeg|png)$", ignore_case = TRUE)) else 0

  list(
    data_dir = data_dir,
    train_dir_exists = train_ok,
    val_dir_exists = val_ok,
    train_annotations = train_anno,
    val_annotations = val_anno,
    train_images = train_images,
    val_images = val_images,
    valid = train_ok && val_ok && train_anno && val_anno && (train_images + val_images) > 0
  )
}

#' Summarize a dataset directory
#' @param data_dir Directory containing 'train' and 'val'
#' @return A tibble with counts for train and val
#' @export
summarize_dataset <- function(data_dir) {
  data_dir <- fs::path_abs(fs::path_norm(data_dir))
  dirs <- c("train", "val")
  tibble::tibble(
    split = dirs,
    images = vapply(dirs, function(d) {
      p <- fs::path(data_dir, d)
      if (!fs::dir_exists(p)) return(0L)
      length(fs::dir_ls(p, regexp = "\\.(jpg|jpeg|png)$", ignore_case = TRUE))
    }, integer(1)),
    annotations = vapply(dirs, function(d) {
      fs::file_exists(fs::path(data_dir, d, "_annotations.coco.json"))
    }, logical(1))
  )
}

