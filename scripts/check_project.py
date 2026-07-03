import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name, command):
    print(f"\n== {name} ==", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    python = sys.executable
    steps = [
        ("compile python files", [python, "-m", "compileall", "-q", "."]),
        ("list project entries", [python, "main.py", "list"]),
        ("run synthetic tests", [python, "-m", "unittest", "discover", "-s", "tests"]),
    ]

    for name, command in steps:
        run_step(name, command)

    print("\nProject check passed.", flush=True)


if __name__ == "__main__":
    main()
