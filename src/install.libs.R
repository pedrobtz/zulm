# Installs the probe DLL and every variant shared object (design §13.1),
# and records the native-build hooks for zulm_info() (design §13.4).
files <- Sys.glob(paste0("*", SHLIB_EXT))
dest <- file.path(R_PACKAGE_DIR, paste0("libs", R_ARCH))
dir.create(dest, recursive = TRUE, showWarnings = FALSE)
file.copy(files, dest, overwrite = TRUE)
if (file.exists("symbols.rds")) {
  file.copy("symbols.rds", dest, overwrite = TRUE)
}
hooks <- data.frame(
  ZULM_CFLAGS = Sys.getenv("ZULM_CFLAGS"),
  ZULM_CXXFLAGS = Sys.getenv("ZULM_CXXFLAGS")
)
write.dcf(hooks, file.path(dest, "zulm-build.dcf"))
