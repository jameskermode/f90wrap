from pathlib import Path
import glob

wrapper_files = glob.glob("f90wrap_*.f90")
if not wrapper_files:
    raise SystemExit("No f90wrap_*.f90 files found")

wrapper = ""
for f in wrapper_files:
    wrapper += Path(f).read_text(encoding="utf-8")

expected = [
    "integer(selected_int_kind(9)), intent(out) :: i",
    "real(selected_real_kind(13,300)), intent(out) :: a",
]

missing = [line for line in expected if line not in wrapper]
if missing:
    raise SystemExit(
        "Missing expected wrapper lines:\n"
        + "\n".join(missing)
        + "\n\nGenerated wrapper:\n"
        + wrapper
    )

