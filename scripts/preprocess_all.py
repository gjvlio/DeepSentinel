"""
preprocess_all.py — Forwarding proxy to scripts/preprocessing/preprocess_all.py.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.preprocessing.preprocess_all import main

if __name__ == "__main__":
    main()
