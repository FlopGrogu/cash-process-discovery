#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from process_discovery_cash.cli.validate_splitminer_java import main

if __name__ == "__main__":
    main()
