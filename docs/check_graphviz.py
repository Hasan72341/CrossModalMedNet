import shutil
import os

with open("graphviz_check.txt", "w") as f:
    has_dot = shutil.which("dot") is not None
    f.write(f"dot: {has_dot}\n")
