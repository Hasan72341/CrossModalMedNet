import shutil
import subprocess

with open("tools_output.txt", "w") as f:
    f.write(f"pdflatex: {shutil.which('pdflatex')}\n")
    f.write(f"pandoc: {shutil.which('pandoc')}\n")
    try:
        res = subprocess.run(["pdflatex", "--version"], capture_output=True, text=True)
        f.write(f"pdflatex output: {res.stdout.splitlines()[0]}\n")
    except Exception as e:
        f.write(f"pdflatex error: {e}\n")

    try:
        res = subprocess.run(["pandoc", "--version"], capture_output=True, text=True)
        f.write(f"pandoc output: {res.stdout.splitlines()[0]}\n")
    except Exception as e:
        f.write(f"pandoc error: {e}\n")
