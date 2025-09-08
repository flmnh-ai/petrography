# Metrics parsing

#' Parse Detectron2 metrics.json into tibbles
#'
#' Reads a Detectron2 `metrics.json` (JSONL) and returns separate tibbles for
#' training rows, aggregate validation metrics (bbox + segm), and per-class AP
#' when available.
#'
#' @param metrics_file Path to `metrics.json`.
#' @return A list with elements `training`, `validation`, and `classwise` (tibbles).
#' @keywords internal
parse_metrics <- function(metrics_file) {
  if (!fs::file_exists(metrics_file)) return(list(training = tibble::tibble(), validation = tibble::tibble(), classwise = tibble::tibble()))
  con <- file(metrics_file, open = "r"); on.exit(close(con), add = TRUE)
  df <- tryCatch(jsonlite::stream_in(con, verbose = FALSE), error = function(e) NULL)
  if (is.null(df)) return(list(training = tibble::tibble(), validation = tibble::tibble(), classwise = tibble::tibble()))
  d <- tibble::as_tibble(df) |> clean_names()

  validation_rows <- d |>
    dplyr::filter(dplyr::if_any(dplyr::contains("bbox"), ~ !is.na(.)) | dplyr::if_any(dplyr::contains("segm"), ~ !is.na(.)))

  training <- d |>
    dplyr::filter(!dplyr::row_number() %in% dplyr::row_number(validation_rows))

  validation <- validation_rows |>
    dplyr::select(iteration, dplyr::contains("bbox"), dplyr::contains("segm"))

  classwise_cols <- grep("^ap_", names(d), value = TRUE)
  classwise <- tibble::tibble()
  if (length(classwise_cols) > 0 && nrow(validation_rows) > 0) {
    classwise <- validation_rows |>
      dplyr::select(iteration, dplyr::all_of(classwise_cols))
  }
  list(training = training, validation = validation, classwise = classwise)
}
