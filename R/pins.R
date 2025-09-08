# Minimal pins integration for publishing and retrieving models

#' Get a configured pins board for petrographer models
#' @param config Optional board object override (rarely needed)
#' @return A pins board object
#' @export
pg_board <- function(config = NULL) {
  if (!requireNamespace("pins", quietly = TRUE)) {
    cli::cli_abort("pins is required. Install with install.packages('pins')")
  }

  if (!is.null(config) && inherits(config, "pins_board")) return(config)

  pins_path <- Sys.getenv("PETRO_PINS_PATH", "")
  s3_bucket <- Sys.getenv("PETRO_S3_BUCKET", "")
  s3_prefix <- Sys.getenv("PETRO_S3_PREFIX", "petrographer")

  if (nzchar(s3_bucket)) {
    return(pins::board_s3(bucket = s3_bucket, prefix = s3_prefix, versioned = TRUE))
  }
  if (nzchar(pins_path)) {
    return(pins::board_folder(pins_path, versioned = TRUE))
  }
  pins::board_local(versioned = TRUE)
}

#' Publish (pin) a trained model directory to a board
#' @param model_dir Directory containing model_final.pth and config.yaml
#' @param name Pin name to publish under
#' @param board A pins board (default: pg_board())
#' @param metadata Optional named list to store with the pin
#' @param include_metrics Whether to include metrics.json if present (default TRUE)
#' @return pins metadata (list)
#' @export
publish_model <- function(model_dir, name, board = pg_board(), metadata = list(), include_metrics = TRUE) {
  if (!requireNamespace("pins", quietly = TRUE)) {
    cli::cli_abort("pins is required. Install with install.packages('pins')")
  }

  model_dir <- fs::path_abs(fs::path_norm(model_dir))
  model_file <- fs::path(model_dir, "model_final.pth")
  config_file <- fs::path(model_dir, "config.yaml")
  if (!fs::file_exists(model_file)) cli::cli_abort("Missing file: {.path {model_file}}")
  if (!fs::file_exists(config_file)) cli::cli_abort("Missing file: {.path {config_file}}")

  files <- c(model_file, config_file)
  metrics_file <- fs::path(model_dir, "metrics.json")
  if (isTRUE(include_metrics) && fs::file_exists(metrics_file)) files <- c(files, metrics_file)

  md <- c(list(model_dir = as.character(model_dir), published = Sys.time()), metadata)

  # Write a self-describing metadata JSON alongside artifacts for portability
  meta_json <- fs::path(model_dir, "petrographer_metadata.json")
  pg_meta <- list(
    name = name,
    published = as.character(Sys.time()),
    model_dir = as.character(model_dir),
    artifacts = basename(files),
    metadata = metadata
  )
  try({
    jsonlite::write_json(pg_meta, meta_json, auto_unbox = TRUE, pretty = TRUE)
    files <- c(files, meta_json)
  }, silent = TRUE)
  cli::cli_alert_info("Publishing model as pin: {.strong {name}}")
  pins::pin_upload(board, files, name = name, metadata = md)
  pins::pin_meta(board, name)
}

#' Retrieve a model pin and return resolved file paths
#' @param name Pin name
#' @param version Optional version id (NULL = latest)
#' @param board A pins board (default: pg_board())
#' @return List with fields model_path, config_path, pin_meta
#' @export
get_model <- function(name, version = NULL, board = pg_board()) {
  if (!requireNamespace("pins", quietly = TRUE)) {
    cli::cli_abort("pins is required. Install with install.packages('pins')")
  }

  available <- tryCatch(pins::pin_list(board), error = function(e) character())
  if (!(name %in% available)) {
    cli::cli_abort("Pin '{name}' not found on board. Available: {paste(available, collapse = ', ')}")
  }

  files <- if (is.null(version)) pins::pin_download(board, name) else pins::pin_download(board, name, version = version)
  model_path <- files[grepl("model_final\\.pth$", files)]
  config_path <- files[grepl("config\\.yaml$", files)]
  if (!length(model_path)) cli::cli_abort("model_final.pth not found in pin '{name}'")
  if (!length(config_path)) cli::cli_abort("config.yaml not found in pin '{name}'")

  list(
    model_path = model_path[[1]],
    config_path = config_path[[1]],
    pin_meta = pins::pin_meta(board, name, version = version)
  )
}
