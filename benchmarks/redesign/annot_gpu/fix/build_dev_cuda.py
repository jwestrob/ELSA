"""Build an unreleased aksha-cuda12 dev wheel from the branch with release/build.py's build_gpu, unchanged.

Run inside the maintainer manylinux2014 container (devtoolset-10, CUDA 12.5 at /opt/cuda) with the v4
build venv's Python. Host headers and the private ABI come from the installed aksha-runtime 0.1.1 of
the side-by-side environment. No auditwheel repair (policy None): a local linux_x86_64 wheel for the
side-by-side environment only; never published.

Usage: build_dev_cuda.py BRANCH_CHECKOUT RUNTIME_SITE_PACKAGES OUTPUT_DIR GPU_VERSION
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    branch, site, output, gpu_version = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
                                          sys.argv[4])
    sys.path.insert(0, str(branch / "release"))
    from build import build_gpu  # noqa: E402  (branch recipe, unchanged)

    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(branch / "native", output / "native")
    logs = output / "build-logs"
    logs.mkdir()
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME", "USER", "TMPDIR", "LANG"}}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", LC_ALL="C", TMPDIR="/tmp",
               PYTHONPATH=str(site))  # Cython resolves astra_pyhmmer .pxd files from the runtime site
    abi = subprocess.check_output([sys.executable, str(site / "plan7_gpu/_abi.py")],
                                  env=env, text=True).strip()
    assert len(abi) == 64, abi
    prep = {"source_lock": {"gpu_version": gpu_version, "runtime_version": "0.1.1"}}
    (output / "dev-build.json").write_text(json.dumps({
        "branch": str(branch), "runtime_site": str(site), "private_abi": abi,
        "gpu_version": gpu_version, "policy": None, "published": False,
        "git_head": subprocess.run(["git", "-C", str(branch), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
    }, indent=1))
    wheel = build_gpu(output, prep, site, abi, env, logs, Path("/opt/cuda"), None)
    print("WHEEL", wheel)


if __name__ == "__main__":
    main()
