# ============================================================================
# Health checks: Python and HPC
# ============================================================================

#' Check Python environment and required modules
#' @return Invisibly returns a list with python config and module availability
#' @export
check_python <- function() {
  cfg <- reticulate::py_config()
  cli::cli_h2("Python Environment")
  cli::cli_dl(c(
    "Python" = cfg$python,
    "Version" = as.character(cfg$version),
    "Virtualenv" = cfg$virtualenv
  ))

  mods <- c("detectron2", "sahi", "skimage")
  status <- vapply(mods, function(m) reticulate::py_module_available(m), logical(1))
  for (i in seq_along(mods)) {
    if (status[i]) cli::cli_alert_success(paste(mods[i], "available")) else cli::cli_alert_warning(paste(mods[i], "missing"))
  }
  invisible(list(config = cfg, modules = setNames(as.list(status), mods)))
}

#' Check HPC connectivity and tools
#' @param hpc_host Hostname (e.g., 'hpg')
#' @param hpc_user Optional username
#' @return TRUE if checks pass (invisibly)
#' @export
check_hpc <- function(hpc_host = "hpg", hpc_user = NULL) {
  cli::cli_h2("HPC Preflight")
  preflight_hpc(hpc_host, hpc_user)
  if (!test_ssh_connection(hpc_host, hpc_user)) cli::cli_abort("Interactive SSH authentication failed")
  cli::cli_alert_success("HPC connectivity OK: {hpc_host}")
  invisible(TRUE)
}

