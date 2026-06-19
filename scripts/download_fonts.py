"""Download Orbitron + Source Code Pro fonts for the BEKE cyberpunk theme.

Run once from the repo root:
    python download_fonts.py

Downloads .ttf files into static/ — referenced by .streamlit/config.toml.
"""

import os
import urllib.request

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Google Fonts GitHub repo — static TTF files
BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl"

FONTS = {
    # Orbitron — futuristic display font
    "Orbitron-Regular.ttf":   f"{BASE}/orbitron/static/Orbitron-Regular.ttf",
    "Orbitron-Medium.ttf":    f"{BASE}/orbitron/static/Orbitron-Medium.ttf",
    "Orbitron-SemiBold.ttf":  f"{BASE}/orbitron/static/Orbitron-SemiBold.ttf",
    "Orbitron-Bold.ttf":      f"{BASE}/orbitron/static/Orbitron-Bold.ttf",
    "Orbitron-ExtraBold.ttf": f"{BASE}/orbitron/static/Orbitron-ExtraBold.ttf",
    # Source Code Pro — terminal monospace
    "SourceCodePro-Regular.ttf": f"{BASE}/sourcecodepro/static/SourceCodePro-Regular.ttf",
    "SourceCodePro-Bold.ttf":    f"{BASE}/sourcecodepro/static/SourceCodePro-Bold.ttf",
}

def main():
    for filename, url in FONTS.items():
        dest = os.path.join(STATIC_DIR, filename)
        if os.path.exists(dest):
            print(f"  skip {filename} (already exists)")
            continue
        print(f"  downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            size_kb = os.path.getsize(dest) / 1024
            if size_kb < 5:
                os.remove(dest)
                print(f"  WARNING: {filename} too small ({size_kb:.0f}KB) — may be a 404 page")
            else:
                print(f"  OK ({size_kb:.0f}KB)")
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\nDone. Fonts saved to static/")
    print("Run:  streamlit run app.py")

if __name__ == "__main__":
    main()
