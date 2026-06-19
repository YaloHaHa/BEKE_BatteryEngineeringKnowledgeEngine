import sys
from pathlib import Path

# Add repo root to sys.path so pytest can resolve `src.*` imports
sys.path.insert(0, str(Path(__file__).parent))
