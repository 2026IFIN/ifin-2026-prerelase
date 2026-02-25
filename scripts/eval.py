from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config
from runner import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IFIN evaluation entry point")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = evaluate(config, checkpoint_path=args.checkpoint)
    print(result)


if __name__ == "__main__":
    main()
