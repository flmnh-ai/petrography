# ============================================================================
# Model Loading and Management
# ============================================================================

#' Load petrography detection model
#' @param model_path Path to trained model weights (default: 'Detectron2_Models/model_final.pth')
#' @param config_path Path to model config (default: 'Detectron2_Models/config.yaml')
#' @param confidence Confidence threshold (default: 0.5)
#' @param device Device to use: 'cpu', 'cuda', 'mps' (default: 'cpu')
#' @return PetrographyModel object
#' @export
load_model <- function(model_path = NULL,
                       config_path = NULL,
                       confidence = 0.5,
                       device = "cpu") {

  cache <- get_model_cache_dir()
  default_model <- fs::path(cache, "model_final.pth")
  default_config <- fs::path(cache, "config.yaml")

  if (is.null(model_path)) model_path <- default_model
  if (is.null(config_path)) config_path <- default_config

  if (!fs::file_exists(model_path) || !fs::file_exists(config_path)) {
    cli::cli_alert_info("Model files not found. Downloading...")
    download_model()
  }

  sahi_model <- sahi$AutoDetectionModel$from_pretrained(
    model_type = 'detectron2',
    model_path = model_path,
    config_path = config_path,
    confidence_threshold = confidence,
    device = device
  )

  model <- list(
    sahi_model = sahi_model,
    model_path = model_path,
    config_path = config_path,
    confidence = confidence,
    device = device
  )
  class(model) <- "PetrographyModel"
  return(model)
}


get_model_cache_dir <- function() {
  tools::R_user_dir("petrographer", which = "cache")
}


download_model <- function(force = FALSE) {
  cache_dir <- get_model_cache_dir()
  fs::dir_create(cache_dir)

  model_url <- "https://www.dropbox.com/scl/fi/3ilo6msi7r1d9fmfn1zq2/model_final.pth?rlkey=6x2ielfy0fr7kijkysa0i3b3l&st=wbfz9k50&dl=1"
  config_url <- "https://www.dropbox.com/scl/fi/kjlggms8k1x4ghhjiph39/config.yaml?rlkey=8lqiu9eeh6xtjcoj2v7ksyb3k&st=haqn63up&dl=1"

  model_path <- fs::path(cache_dir, "model_final.pth")
  config_path <- fs::path(cache_dir, "config.yaml")

  if (!fs::file_exists(model_path) || force) {
    cli::cli_alert_info("Downloading model weights...")
    download.file(model_url, model_path, mode = "wb")
    cli::cli_alert_success("Model weights saved to: {.path {model_path}}")
  } else {
    cli::cli_alert_info("Model weights already present at: {.path {model_path}}")
  }

  if (!fs::file_exists(config_path) || force) {
    cli::cli_alert_info("Downloading model config...")
    download.file(config_url, config_path, mode = "wb")
    cli::cli_alert_success("Model config saved to: {.path {config_path}}")
  } else {
    cli::cli_alert_info("Model config already present at: {.path {config_path}}")
  }

  return(list(model_path = model_path, config_path = config_path))
}

clear_model_cache <- function() {
  cache_dir <- get_model_cache_dir()
  if (fs::dir_exists(cache_dir)) {
    fs::dir_delete(cache_dir)
    cli::cli_alert_success("Cleared model cache at: {.path {cache_dir}}")
  } else {
    cli::cli_alert_info("No model cache found at: {.path {cache_dir}}")
  }
}
