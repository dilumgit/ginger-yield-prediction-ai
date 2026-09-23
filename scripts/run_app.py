"""Launch script for the Streamlit research prototype.

Usage:
    python scripts/run_app.py
"""

import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "src" / "ui" / "app.py"


def main():
    print(f"Launching Ginger Yield AI Streamlit Prototype from: {APP_PATH.resolve()}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStreamlit application stopped by user.")


if __name__ == "__main__":
    main()
