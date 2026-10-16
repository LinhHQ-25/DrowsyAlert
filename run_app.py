import os
import subprocess
import sys


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    python_bin = sys.executable
    venv_python = os.path.join(project_dir, "venv", "Scripts", "python.exe")
    if os.path.exists(venv_python) and os.path.abspath(python_bin) != os.path.abspath(venv_python):
        python_bin = venv_python

    cmd = [python_bin, "-m", "streamlit", "run", "app.py"]
    subprocess.run(cmd)
