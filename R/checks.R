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
  cli::cli_h2("HPC Connectivity")
  ok <- test_ssh_connection(hpc_host, hpc_user)
  if (isTRUE(ok)) {
    cli::cli_alert_success("SSH reachable and authenticated: {hpc_host}")
  } else {
    cli::cli_alert_danger("SSH check failed for: {hpc_host}")
  }
  # Best-effort info for sbatch (non-fatal)
  target <- if (!is.null(hpc_user) && nzchar(hpc_user)) paste0(hpc_user, "@", hpc_host) else hpc_host
  res <- tryCatch({ processx::run("ssh", c(target, "bash", "-lc", "command -v sbatch >/dev/null && echo 'sbatch available' || echo 'sbatch not found'"), error_on_status = FALSE) }, error = function(e) NULL)
  if (!is.null(res) && nzchar(res$stdout)) cli::cli_alert_info(trimws(res$stdout))
  invisible(isTRUE(ok))
}
