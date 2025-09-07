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
  train_images <- if (train_ok) length(fs::dir_ls(train_dir, regexp = "(?i)\\.(jpg|jpeg|png)$")) else 0
  val_images <- if (val_ok) length(fs::dir_ls(val_dir, regexp = "(?i)\\.(jpg|jpeg|png)$")) else 0

  res <- list(
    data_dir = data_dir,
    train_dir_exists = train_ok,
    val_dir_exists = val_ok,
    train_annotations = train_anno,
    val_annotations = val_anno,
    train_images = train_images,
    val_images = val_images,
    valid = train_ok && val_ok && train_anno && val_anno && (train_images + val_images) > 0
  )
  class(res) <- c("CocoValidation", class(res))
  if (!res$valid) {
    # Print a concise summary then abort
    print(res)
    cli::cli_abort("Dataset validation failed")
  }
  res
}

#' Validate dataset and compute total size
#' @param data_dir Directory containing the dataset (expects 'train' and 'val')
#' @return A list with validation flags, counts, and size metrics
#' @export
validate_dataset <- function(data_dir) {
  cli::cli_h2("Dataset Validation")
  
  # Reuse coco validation (prints only when autoprinted; aborts if invalid)
  val <- validate_coco_dataset(data_dir)

  # Compute total size (bytes and MB)
  files <- fs::dir_ls(val$data_dir, recurse = TRUE, type = "file")
  total_bytes <- sum(fs::file_size(files), na.rm = TRUE)
  total_mb <- as.numeric(total_bytes) / (1024^2)

  out <- val
  out$size_bytes <- as.numeric(total_bytes)
  out$size_mb <- round(total_mb, 1)
  class(out) <- c("DatasetValidation", class(out))
  
  # Print validation summary
  cli::cli_dl(c(
    "Data directory" = out$data_dir,
    "Train images" = out$train_images,
    "Val images" = out$val_images,
    "Total size" = paste0(out$size_mb, " MB")
  ))
  
  cli::cli_alert_success("Dataset validation passed")
  out
}

#' Summarize a dataset directory
#' @param data_dir Directory containing 'train' and 'val'
#' @return A tibble with counts for train and val
#' @export
summarize_dataset <- function(data_dir) {
  data_dir <- fs::path_abs(fs::path_norm(data_dir))
  dirs <- c("train", "val")
  out <- tibble::tibble(
    split = dirs,
    images = vapply(dirs, function(d) {
      p <- fs::path(data_dir, d)
      if (!fs::dir_exists(p)) return(0L)
      length(fs::dir_ls(p, regexp = "(?i)\\.(jpg|jpeg|png)$"))
    }, integer(1)),
    annotations = vapply(dirs, function(d) {
      fs::file_exists(fs::path(data_dir, d, "_annotations.coco.json"))
    }, logical(1))
  )
  class(out) <- c("DatasetSummary", class(out))
  out
}
