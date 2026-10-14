"""GGUF interop with upstream's gguf package (design §11, Stage 2).

    python tools/interop/gguf-check.py read <dir>    # check zulm's files
    python tools/interop/gguf-check.py write <dir>   # write files for zulm

A zulm tensor of R dim c(d1, ..., dk) is ggml's ne = {d1, ..., dk}, which
NumPy sees as shape (dk, ..., d1) holding the same bytes.
"""
import sys

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

SHAPES = {"v": (7,), "m": (3, 4), "t": (2, 3, 4), "q": (2, 1, 3, 2)}


def expected(r_dim):
    n = int(np.prod(r_dim))
    x = np.arange(n, dtype=np.float64) / 8 - 3
    return x.reshape(tuple(reversed(r_dim))).astype(np.float32)


def as_f32(t):
    data = np.asarray(t.data)
    if t.tensor_type == GGMLQuantizationType.F32:
        return data.astype(np.float32)
    if t.tensor_type == GGMLQuantizationType.F16:
        return data.view(np.float16).astype(np.float32)
    if t.tensor_type == GGMLQuantizationType.BF16:
        bits = data.view(np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32)
    raise SystemExit(f"unexpected tensor type {t.tensor_type}")


def read(dir_):
    types = {"f32": GGMLQuantizationType.F32, "f16": GGMLQuantizationType.F16,
             "bf16": GGMLQuantizationType.BF16}
    for name, qtype in types.items():
        r = GGUFReader(f"{dir_}/zulm-{name}.gguf")
        got = {t.name: t for t in r.tensors}
        assert list(got) == list(SHAPES), (name, list(got))
        for tname, r_dim in SHAPES.items():
            t = got[tname]
            assert t.tensor_type == qtype, (name, tname, t.tensor_type)
            assert [int(x) for x in t.shape] == list(r_dim), (name, tname, t.shape)
            want = expected(r_dim)
            have = as_f32(t).reshape(want.shape)
            assert np.array_equal(have, want), (name, tname)
        source = r.fields["source"]
        assert bytes(source.parts[source.data[0]]).decode() == "zulm"
        width = r.fields["width"]
        assert int(width.parts[width.data[0]][0]) == 64
        print(f"ok  zulm-{name}.gguf: {len(SHAPES)} tensors and metadata as written")


def write(dir_):
    for name, qtype in (("f32", None), ("f16", GGMLQuantizationType.F16),
                        ("bf16", GGMLQuantizationType.BF16)):
        w = GGUFWriter(f"{dir_}/py-{name}.gguf", "zulm-interop")
        w.add_string("source", "gguf-py")
        w.add_uint32("width", 64)
        for tname, r_dim in SHAPES.items():
            x = expected(r_dim)
            if qtype is None:
                w.add_tensor(tname, x)
            elif qtype == GGMLQuantizationType.F16:
                w.add_tensor(tname, x.astype(np.float16))
            else:
                bits = (x.view(np.uint32) >> 16).astype(np.uint16)
                w.add_tensor(tname, bits, raw_shape=x.shape, raw_dtype=qtype)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        print(f"wrote py-{name}.gguf")


if __name__ == "__main__":
    {"read": read, "write": write}[sys.argv[1]](sys.argv[2])
