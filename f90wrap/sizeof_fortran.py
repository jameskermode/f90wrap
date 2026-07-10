import os
import shlex
import subprocess
import tempfile
from pathlib import Path


def resolve_sizeof_fortran_t(explicit_size=None, compiler=None):
    if explicit_size is not None:
        if explicit_size <= 0:
            raise ValueError("--sizeof-fortran-t must be positive")
        return explicit_size
    if compiler:
        return probe_sizeof_fortran_t(compiler)
    from f90wrap.sizeof_fortran_t import sizeof_fortran_t as installed_sizeof_fortran_t
    return installed_sizeof_fortran_t()


def probe_sizeof_fortran_t(compiler):
    command = shlex.split(compiler)
    if not command:
        raise ValueError("Fortran compiler command is empty")

    source = Path(__file__).with_name("sizeoffortran.f90")
    with tempfile.TemporaryDirectory(prefix="f90wrap-sizeof-") as tmp:
        tmp_path = Path(tmp)
        driver = tmp_path / "probe.f90"
        executable = tmp_path / ("probe.exe" if os.name == "nt" else "probe")
        driver.write_text(
            "program probe\n"
            "  implicit none\n"
            "  integer :: size_out\n"
            "  call sizeof_fortran_t(size_out)\n"
            "  print *, size_out\n"
            "end program probe\n"
        )
        _run(command + [str(source), str(driver), "-o", str(executable)])
        result = _run([str(executable)])

    try:
        size = int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"Fortran handle-size probe returned {result.stdout.strip()!r}"
        ) from exc
    if size <= 0:
        raise RuntimeError(f"Fortran handle-size probe returned {size}")
    return size


def _run(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot run {' '.join(command)}: {exc}") from exc
    if result.returncode != 0:
        output = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{' '.join(command)} failed: {output}")
    return result
