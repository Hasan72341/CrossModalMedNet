import subprocess
import os

def compile_report(tex_file):
    tectonic_bin = os.environ.get("TECTONIC_BIN", "tectonic")
    cmd = [tectonic_bin, tex_file]
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error during compilation of {tex_file}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    compile_report("cyclegan_brain/cyclegan_brain_report.tex")
