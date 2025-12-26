from pathlib import Path

wrapper = Path("f90wrap_toplevel.f90").read_text(encoding="utf-8")

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

