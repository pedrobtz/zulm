# zulm: Tensor Graphs, Training and Inference on 'ggml'

Builds computation graphs of tensors in R, runs them on the CPU,
differentiates them and trains their parameters, through the 'ggml'
tensor library (<https://github.com/ggml-org/ggml>) bundled in its CPU
subset. Graphs are built once per input shape and run many times; the
same operations train small and medium neural networks with the AdamW
(Loshchilov and Hutter (2019)
[doi:10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
) and SGD optimisers and run pretrained transformer checkpoints read
from 'GGUF' or 'safetensors' files. Results reproduce bit for bit at a
fixed seed whatever the thread count. A versioned C interface lets other
packages build models on top.

## See also

Useful links:

- <https://github.com/pedrobtz/zulm>

- <https://pedrobtz.github.io/zulm/>

- Report bugs at <https://github.com/pedrobtz/zulm/issues>

## Author

**Maintainer**: Pedro Baltazar <pedrobtz@gmail.com> \[copyright holder\]

Authors:

- Pedro Baltazar <pedrobtz@gmail.com> \[copyright holder\]

Other contributors:

- The ggml authors (Bundled 'ggml' library; see inst/COPYRIGHTS)
  \[copyright holder\]

- Mozilla Foundation ('llamafile' matrix kernels in the bundled 'ggml';
  see inst/COPYRIGHTS) \[copyright holder\]

- Jeffrey Quesnelle ('YaRN' rotary scaling in the bundled 'ggml'; see
  inst/COPYRIGHTS) \[copyright holder\]

- Bowen Peng ('YaRN' rotary scaling in the bundled 'ggml'; see
  inst/COPYRIGHTS) \[copyright holder\]

- The Apache Software Foundation (Arrow C data interface definitions;
  see inst/COPYRIGHTS) \[copyright holder\]
