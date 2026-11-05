"""Read-only inventory, integrity verification."""

import argparse
import json
from pathlib import Path

from .store import ArtifactStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "verify"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--object", type=Path)
    args = parser.parse_args()
    store = ArtifactStore(args.root)
    if args.action == "verify":
        if args.object is None:
            parser.error("verify requires --object")
        result = store.verify(args.object)
    else:
        result = store.inventory()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
