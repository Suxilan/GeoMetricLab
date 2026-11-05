"""Install reversible C++ runtime hooks into the executing Conda environment."""

from pathlib import Path
import sys


def main() -> None:
    prefix = Path(sys.prefix)
    if sys.platform != "linux" or not (prefix / "conda-meta").is_dir():
        print("Conda C++ runtime hooks are only needed for Linux Conda environments.")
        return

    templates = Path(__file__).resolve().parent
    for action in ("activate", "deactivate"):
        source = templates / f"{action}.sh"
        target = prefix / "etc" / "conda" / f"{action}.d" / "gmlab-runtime.sh"
        content = source.read_text()
        if target.is_file() and target.read_text() == content:
            print(f"Already configured: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        print(f"Configured: {target}")
    print("Deactivate/reactivate this environment before checking Torch and FAISS.")


if __name__ == "__main__":
    main()
