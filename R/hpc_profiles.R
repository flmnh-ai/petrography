# ============================================================================
# HPC Profiles (minimal, optional)
# ============================================================================

hpc_profile_config <- function(name = "hpg") {
  switch(
    name,
    hpg = list(
      modules = c("module purge", "module load detectron2"),
      sbatch  = list(time = "04:00:00", cpus = 4, mem = "24gb", gpus = 1, partition = NULL)
    ),
    # default fallback
    list(
      modules = c("module purge", "module load detectron2"),
      sbatch  = list(time = "04:00:00", cpus = 4, mem = "24gb", gpus = 1, partition = NULL)
    )
  )
}

