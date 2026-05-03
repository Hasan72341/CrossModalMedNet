import os
import sys

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
folder = "cyclegan_brain"
tex_path = os.path.join(DOCS_DIR, folder, f"{folder}_report.tex")

content = "DEBUG TEST CONTENT"

try:
    with open(tex_path, "w") as f:
        f.write(content)
    print(f"Successfully wrote to {tex_path}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
