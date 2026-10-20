"""Mount an explicit Node distribution without changing the benchmark image."""
from pathlib import Path


def node_runtime_mounts(node_directory: Path, platform: str) -> list[tuple[Path, str, bool]]:
    """Optionally make an ARM Node executable inside an amd64 task image.

    The Linux VM is native ARM on Apple Silicon. Giving just Node its native
    loader/libraries avoids the reproduced Rosetta child-reaping stall while
    leaving the image's amd64 shell, compiler and verifier unchanged. No loader
    wrapper, LD_LIBRARY_PATH, signal injection or process retry is needed.
    Ordinary Node distributions and native ARM images keep their existing setup.
    """
    node = Path(node_directory).resolve()
    mounts = [(node, "/opt/ac-node", True)]
    libraries = node / "compat" / "aarch64-linux-gnu"
    if platform == "linux/amd64" and libraries.is_dir():
        loader = libraries / "ld-linux-aarch64.so.1"
        if not loader.is_file():
            raise FileNotFoundError(f"ARM Node compatibility libraries lack loader: {loader}")
        mounts.extend([(libraries, "/lib/aarch64-linux-gnu", True),
                       (loader, "/lib/ld-linux-aarch64.so.1", True)])
    return mounts
