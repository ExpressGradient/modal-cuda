import argparse
from pathlib import Path

import modal


def main():
    parser = argparse.ArgumentParser(
        prog="mcc", description="Compile and Run CUDA C scripts with Modal"
    )
    parser.add_argument("input", help="Input CUDA C file", type=Path)
    parser.add_argument(
        "--app",
        help="App name to run the CUDA script; defaults to the input file name",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        help="Choose the GPU to run the script",
        type=str,
        default="T4",
        choices=[
            "T4",
            "L4",
            "A10",
            "A100",
            "A100-40GB",
            "A100-80GB",
            "L40S",
            "H100",
            "H200",
            "B200",
        ],
    )
    parser.add_argument(
        "--image",
        type=str,
        default="nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04",
        help="Container image to use for Modal (registry reference)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60 * 10,
        help="Execution timeout in seconds for the CUDA run (default: 600)",
    )
    parser.add_argument(
        "--nvcc-arg",
        dest="nvcc_args",
        action="append",
        default=None,
        metavar="FLAG",
        help="Extra argument to pass to nvcc; repeat for multiple flags",
    )

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"File {args.input} not found")

    if args.input.suffix.lower() != ".cu":
        parser.error("Input file must have a .cu extension")

    code = args.input.read_text()
    if not code.strip():
        parser.error(f"File {args.input} is empty")

    app_name = args.app or args.input.name
    cuda_image = modal.Image.from_registry(args.image, add_python="3.12")
    app = modal.App(app_name, image=cuda_image)

    @app.function(gpu=args.gpu, timeout=args.timeout, serialized=True)
    def compile_and_run_cuda(cuda_source: str, nvcc_args: list[str] | None = None):
        import subprocess
        import uuid
        from pathlib import Path

        run_id = uuid.uuid4().hex
        source_path = Path(f"/tmp/{run_id}.cu")
        binary_path = Path(f"/tmp/{run_id}")
        source_path.write_text(cuda_source)

        def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
            print(f"$ {' '.join(cmd)}")

            completed = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if completed.stdout:
                print(completed.stdout)

            if completed.stderr:
                print(completed.stderr)

            return completed

        try:
            extra_args = nvcc_args or []

            compile_result = _run_command(
                ["nvcc", str(source_path), "-o", str(binary_path), *extra_args]
            )

            if compile_result.returncode != 0:
                raise RuntimeError("nvcc failed to compile the CUDA source file")

            execution_result = _run_command([str(binary_path)])

            if execution_result.returncode != 0:
                raise RuntimeError(
                    "Executable exited with a non-zero status while running the CUDA program"
                )

            return execution_result.stdout

        finally:
            for path in (source_path, binary_path):
                if path.exists():
                    path.unlink()

    with modal.enable_output():
        with app.run():
            compile_and_run_cuda.remote(code, args.nvcc_args or [])


if __name__ == "__main__":
    main()
