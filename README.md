## mcc

`mcc` is a tiny CLI that compiles and runs a local CUDA C (`.cu`) program on a cloud GPU using [Modal](https://modal.com). Install the package published as `modal-cuda`, keep using the `mcc` command, and the tool will ship your source to a disposable container, invoke `nvcc` inside the GPU runtime you pick, stream stdout/stderr back to your terminal, and clean up the artifacts once the run finishes.

### Highlights

- Runs CUDA samples on-demand without managing local GPU drivers.
- Lets you choose GPU tier (`T4`, `A100`, `H100`, etc.) and container image per run.
- Accepts extra `nvcc` arguments for fine-grained builds.
- Wraps the entire workflow (upload -> compile -> execute -> teardown) in a single command.

## Requirements

- Python 3.12+
- A Modal account with CLI access (`modal token new`).
- Network access that allows Modal to pull the chosen CUDA container image.

## Installation

Pick whichever path matches your workflow:

```bash
# one-off run without installing anything
uvx --from modal-cuda mcc ...

# install once as a reusable uv tool, then call `mcc ...`
uv tool install modal-cuda

# add the dependency to a project and expose `mcc`
uv add modal-cuda

# editable install from the local source checkout
uv sync && uv pip install -e .
```

### Installation examples

```bash
# 1) Quick try-out with uvx
uvx --from modal-cuda mcc sample.cu --gpu A100

# 2) Install once as a tool and reuse
uv tool install modal-cuda
mcc sample.cu --nvcc-arg=-arch=sm_90

# 3) Project dependency + local run
uv add modal-cuda
uv run mcc kernels/reduction.cu --gpu L40S
```

## Usage

```bash
mcc path/to/program.cu [--gpu GPU] [--image IMAGE] [--timeout SECONDS] [--nvcc-arg FLAG ...]
```

| Flag | Description |
| --- | --- |
| `input` | Path to a `.cu` file. The file must exist and be non-empty. |
| `--app` | Name of the Modal app; defaults to the source filename. Useful to group runs in the Modal dashboard. |
| `--gpu` | GPU type to request (`T4`, `L4`, `A10`, `A100`, `A100-40GB`, `A100-80GB`, `L40S`, `H100`, `H200`, `B200`). Defaults to `T4`. |
| `--image` | Container image reference passed to `modal.Image.from_registry`. Defaults to `nvidia/cuda:12.4.1-devel-ubuntu22.04`. |
| `--timeout` | Execution timeout (seconds) enforced by Modal. Defaults to 600. |
| `--nvcc-arg` | Additional flag forwarded to `nvcc`. Repeat for multiple flags (e.g., `--nvcc-arg -arch=sm_90`). |

Example session:

```bash
# Run the provided sample kernel on an A100 GPU with verbose PTX output
mcc sample.cu --gpu A100 --nvcc-arg=-Xptxas --nvcc-arg=-v
```

Modal streams stdout/stderr as the compiler and executable run. If either command exits non‑zero, `mcc` surfaces the failure message and returns a non-zero status to your shell.

## Development

1. Install dependencies with `uv sync`.
2. Run `python -m mcc.main path/to/file.cu` to skip the console entry point while iterating.
3. Use `uv run python -m pip install -e .` to get an editable install for local testing.


## Troubleshooting

- **`modal.ClientException: Unauthorized`** -> Run `modal token new` and try again.
- **`nvcc failed to compile...`** -> Fix compiler errors shown in the streamed stderr or adjust `--nvcc-arg` flags.
- **Long image pull times** -> Prefer the default CUDA image or build a custom Modal image that layers your dependencies once and reuse it via `--image`.

## License

MIT © ExpressGradient
