# ============================================================================
# S3 print methods for friendly, default CLI output
# ============================================================================

#' @export
print.DatasetSummary <- function(x, ...) {
  cli::cli_h2("Dataset Summary")
  tot <- sum(x$images, na.rm = TRUE)
  cli::cli_dl(c(
    "Total images" = tot,
    "Splits" = paste(x$split, collapse = ", ")
  ))
  # Then show the underlying tibble
  NextMethod("print")
}

#' @export
print.DatasetValidation <- function(x, ...) {
  cli::cli_h2("Dataset Validation")
  cli::cli_dl(c(
    "Data directory" = x$data_dir,
    "Train images" = x$train_images,
    "Val images" = x$val_images,
    "Train annotations" = if (isTRUE(x$train_annotations)) "✓" else "✗",
    "Val annotations" = if (isTRUE(x$val_annotations)) "✓" else "✗",
    "Total size" = glue::glue("{x$size_mb} MB")
  ))
  if (isTRUE(x$valid)) cli::cli_alert_success("Dataset valid") else cli::cli_alert_danger("Dataset invalid")
  invisible(x)
}
