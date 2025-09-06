# Package initialization and Python module bindings

# Maintain global bindings for Python modules used via reticulate
utils::globalVariables(c("sahi", "skimage"))

.onLoad <- function(libname, pkgname) {
  # Lazy import Python modules when needed
  sahi <<- reticulate::import("sahi", delay_load = TRUE)
  skimage <<- reticulate::import("skimage", delay_load = TRUE)
}

