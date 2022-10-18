"""
Large parts of this code was taken from [this
setup.py](https://github.com/WardBrian/stan_py_example/blob/main/setup.py) by
Brian Ward.

This setup.py is used only for the building of the Stan models; the rest of the
metadata is in `setup.cfg`.
"""

import os
import platform
from pathlib import Path
from shutil import copy, copytree, rmtree
from typing import Tuple

import cmdstanpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel

MODEL_DIR = "cmpbayes/stan"

MODELS = [
    "bimodal_nonnegative",
]

CMDSTAN_VERSION = "2.29.2"
BINARIES_DIR = "bin"
BINARIES = ["diagnose", "print", "stanc", "stansummary"]
MATH_LIB = "stan/lib/stan_math/lib"
TBB_DIRS = ["tbb", "tbb_2020.3"]


def prune_cmdstan(cmdstan_dir: os.PathLike) -> None:
    """
    Keep only the cmdstan executables and tbb files (minimum required to run
    cmdstanpy commands on a pre-compiled model).
    """
    original_dir = Path(cmdstan_dir).resolve()
    parent_dir = original_dir.parent
    temp_dir = parent_dir / "temp"
    if temp_dir.is_dir():
        rmtree(temp_dir)
    temp_dir.mkdir()

    print("Copying ", original_dir, " to ", temp_dir, " for pruning")
    copytree(original_dir / BINARIES_DIR, temp_dir / BINARIES_DIR)
    copy(original_dir / "makefile", temp_dir / "makefile")
    for f in (temp_dir / BINARIES_DIR).iterdir():
        if f.is_dir():
            rmtree(f)
        elif f.is_file() and f.stem not in BINARIES:
            os.remove(f)
    for tbb_dir in TBB_DIRS:
        copytree(original_dir / MATH_LIB / tbb_dir,
                 temp_dir / MATH_LIB / tbb_dir)

    rmtree(original_dir)
    temp_dir.rename(original_dir)


def repackage_cmdstan() -> bool:
    return os.environ.get("STAN_PY_EXAMPLE_REPACKAGE_CMDSTAN", "").lower() in [
        "true",
        "1",
    ]


def maybe_install_cmdstan_toolchain() -> None:
    """
    Install C++ compilers required to build stan models on Windows machines.
    """

    try:
        cmdstanpy.utils.cxx_toolchain_path()
    except Exception:
        from cmdstanpy.install_cxx_toolchain import run_rtools_install

        run_rtools_install({"version": None, "dir": None, "verbose": True})
        cmdstanpy.utils.cxx_toolchain_path()


def install_cmdstan_deps(cmdstan_dir: Path) -> None:
    from multiprocessing import cpu_count

    if repackage_cmdstan():
        if platform.platform().startswith("Win"):
            maybe_install_cmdstan_toolchain()
        print("Installing cmdstan to", cmdstan_dir)
        if os.path.isdir(cmdstan_dir):
            print("Removing existing dir", cmdstan_dir)
            rmtree(cmdstan_dir)

        if not cmdstanpy.install_cmdstan(
                version=CMDSTAN_VERSION,
                dir=cmdstan_dir.parent,
                overwrite=True,
                verbose=True,
                cores=cpu_count(),
        ):
            raise RuntimeError(
                "CmdStan failed to install in repackaged directory")
    else:
        try:
            cmdstanpy.cmdstan_path()
        except ValueError as e:
            raise SystemExit(
                "CmdStan not installed, but the package is building from source"
            ) from e


def build_models(target_dir: str) -> None:
    cmdstan_dir = (Path(target_dir) / f"cmdstan-{CMDSTAN_VERSION}").resolve()
    install_cmdstan_deps(cmdstan_dir)
    for model_name in MODELS:
        model = cmdstanpy.CmdStanModel(
            stan_file=os.path.join("src", MODEL_DIR, f"{model_name}.stan"),
            stanc_options={"O1": True},
        )
        copy(model.exe_file, os.path.join(target_dir, f"{model_name}.exe"))

    if repackage_cmdstan():
        prune_cmdstan(cmdstan_dir)


class BuildModels(build_ext):
    """
    Custom build command to pre-compile Stan models.
    """

    def run(self) -> None:
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib, MODEL_DIR)
            self.mkpath(target_dir)
            build_models(target_dir)


setup(
    # Mark this as platform-specific.
    ext_modules=[Extension("stan_py_example.stan", [])],
    # Override the build command.
    cmdclass={
        "build_ext": BuildModels,
    })
