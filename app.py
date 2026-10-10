"""Root entry point forwarding to the main Streamlit dashboard at src/ui/app.py."""

import runpy
import sys
from pathlib import Path

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute the core Streamlit application
APP_PATH = ROOT_DIR / "src" / "ui" / "app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")
