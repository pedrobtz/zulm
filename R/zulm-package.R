#' @keywords internal
"_PACKAGE"

# zujson parses safetensors headers (design §7.4) and zurand draws every
# random number (design §7.2). The imports below keep both declared
# namespaces in use until the stages that call them land.
#' @importFrom zujson json_parse_raw
#' @importFrom zurand rng_key
#' @useDynLib zulm, .registration = TRUE
NULL
