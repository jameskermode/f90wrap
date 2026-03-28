from pathlib import Path

import global_mod


def main():
    wrapper = Path("f90wrap_global_mod.f90").read_text(encoding="utf-8")
    expected = "use global_mod, only: print_number"
    if expected not in wrapper:
        raise SystemExit(
            "Missing selective import in generated wrapper:\n"
            f"{expected}\n\nGenerated wrapper:\n{wrapper}"
        )

    global_mod.global_mod.print_number(7)
    print("Issue #363 test passed: generated wrapper uses selective import")


if __name__ == "__main__":
    main()
