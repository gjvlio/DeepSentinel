"""
evaluate.py — Forwarding proxy to scripts/evaluation/evaluate.py.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluation.evaluate import main

if __name__ == "__main__":
    main()
