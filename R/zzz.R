utils::globalVariables(c("sahi", "skimage"))

.onLoad <- function(libname, pkgname) {
  # Declare Python requirements (Reticulate >= 1.41) without initializing Python
  if (utils::packageVersion("reticulate") >= "1.41") {
    reticulate::py_require(c(
      "sahi", "torch", "torchvision", "opencv-python", "scikit-image"
    ))
    reticulate::py_require(
      "detectron2@git+https://github.com/facebookresearch/detectron2.git@4fa166c043ca45359cd7080523b7122e7e0f9d91"
    )
  }

  # Delay-load Python modules (keeps package load fast + CRAN-safe)
  sahi <<- reticulate::import("sahi", delay_load = TRUE)
  skimage <<- reticulate::import("skimage", delay_load = TRUE)
}
