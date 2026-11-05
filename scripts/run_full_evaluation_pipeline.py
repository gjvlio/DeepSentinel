"""
scripts/run_full_evaluation_pipeline.py
======================================
Root proxy for scripts/evaluation/run_full_evaluation_pipeline.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "scripts" / "evaluation" / "run_full_evaluation_pipeline.py"

if __name__ == "__main__":
    import runpy
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")

