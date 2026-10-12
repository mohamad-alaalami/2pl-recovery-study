"""C++ compiler setup for PyTensor (PyMC's computational backend).

PyTensor compiles its C implementations with g++. Without a compiler it silently falls back to a
pure-Python mode that is far too slow for this study, so real fits refuse to run without one
(require_c_backend).

- Linux / macOS: PyTensor finds g++ on PATH; configure() changes nothing.
- Windows: MSVC is not supported. Install a MinGW-w64 g++ and set RECOVERY2PL_MINGW_BIN to its
  bin directory (default location checked: %LOCALAPPDATA%/winlibs/mingw64/bin). That directory is
  added to PATH and to PyTensor's cxx flag for THIS process only: the compiled modules need the
  MinGW runtime DLLs at load time, so a global ~/.pytensorrc would break PyTensor in processes
  without that PATH entry.

configure() must run before pytensor/pymc are imported; recovery2pl.model does this at import.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "RECOVERY2PL_MINGW_BIN"
_STATE = {"configured": False, "bin": None}


def mingw_bin() -> Path | None:
    candidates = [os.environ.get(ENV_VAR),
                  str(Path(os.environ.get("LOCALAPPDATA", "")) / "winlibs" / "mingw64" / "bin")]
    for c in candidates:
        if c and (Path(c) / "g++.exe").exists():
            return Path(c)
    return None


def configure() -> str:
    """Call before importing pytensor/pymc. Returns the backend that will be used."""
    if _STATE["configured"]:
        return backend()
    _STATE["configured"] = True
    if os.name != "nt" or "cxx=" in os.environ.get("PYTENSOR_FLAGS", ""):
        return backend()
    bin_dir = mingw_bin()
    if bin_dir is None:
        return backend()
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(bin_dir))
    flags = os.environ.get("PYTENSOR_FLAGS", "")
    gpp = (bin_dir / "g++.exe").as_posix()
    os.environ["PYTENSOR_FLAGS"] = f"{flags},cxx={gpp}" if flags else f"cxx={gpp}"
    _STATE["bin"] = str(bin_dir)
    return backend()


def backend() -> str:
    """'c' if PyTensor has a C++ compiler configured, else 'python'. Imports pytensor, so call it
    only after configure()."""
    import pytensor
    return "c" if pytensor.config.cxx else "python"


def require_c_backend() -> str:
    """Raise unless PyTensor compiles to C: a pure-Python fit is unusably slow for this study."""
    if backend() != "c":
        raise RuntimeError("PyTensor has no C++ compiler and would fall back to the pure-Python "
                           "backend (orders of magnitude slower). Install g++ (Windows: MinGW-w64, "
                           f"and set {ENV_VAR} to its bin directory) and retry.")
    return "c"
