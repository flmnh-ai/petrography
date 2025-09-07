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
  
  # Test SSH connection using new ssh package approach
  ok <- tryCatch({
    session <- hpc_session(hpc_host, hpc_user)
    result <- ssh::ssh_exec_internal(session, "echo 'connection test'")
    ssh::ssh_disconnect(session)
    result$status == 0
  }, error = function(e) {
    cli::cli_alert_warning("SSH connection error: {e$message}")
    FALSE
  })
  
  if (isTRUE(ok)) {
    cli::cli_alert_success("SSH reachable and authenticated: {hpc_host}")
  } else {
    cli::cli_alert_danger("SSH check failed for: {hpc_host}")
  }
  
  # Best-effort info for sbatch (non-fatal)
  if (isTRUE(ok)) {
    tryCatch({
      session <- hpc_session(hpc_host, hpc_user)
      res <- ssh::ssh_exec_internal(session, "command -v sbatch >/dev/null && echo 'sbatch available' || echo 'sbatch not found'")
      ssh::ssh_disconnect(session)
      if (res$status == 0 && nzchar(res$stdout)) {
        cli::cli_alert_info(trimws(res$stdout))
      }
    }, error = function(e) {
      cli::cli_alert_info("Could not check sbatch availability")
    })
  }
  
  invisible(isTRUE(ok))
}
