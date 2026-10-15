## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

## Test environments

`R CMD check --as-cran` through the reusable workflows of
https://github.com/pedrobtz/r-actions, on every push to the default
branch:

* Ubuntu (ubuntu-latest), R release with GCC and with clang, and R
  oldrel-1.
* Ubuntu with GCC 16 and with clang 23 (`-std=gnu23`), as CRAN's
  r-devel Debian flavours compile.
* macOS on arm64 (macos-latest) and on x86-64 (macos-15-intel), R
  release.
* Windows (windows-latest), R release with Rtools.
* Without the suggested packages installed.
* AddressSanitizer and UndefinedBehaviorSanitizer with GCC and clang,
  valgrind, `gctorture`, rchk, LTO, GCC's `-fanalyzer`, and the
  rcnst, rlibro and vnu checks of CRAN's special flavours.

## Compiled code

The package bundles the CPU subset of the 'ggml' tensor library (MIT;
authors and licences in `inst/COPYRIGHTS`). It is built once per
instruction-set level (baseline, SSE4.2 and AVX2 on x86-64, NEON on
arm64), with the targets chosen by pragmas in a force-included header,
not by compiler flags; the package loads the build the CPU supports.
No `-f`, `-m` or `-O` flag appears in `Makevars`. ggml's assertions are
routed to R errors, and no `abort`, `exit` or stdio symbol is linked.

## Reverse dependencies

There are no reverse dependencies on CRAN. The `zulm-runtime` branch of
zukzeit (https://github.com/pedrobtz/zukzeit/tree/zulm-runtime) runs its
Toto 2.0 4M parity fixtures with zulm as the tensor runtime, against
zulm's development version, on every CI leg.
