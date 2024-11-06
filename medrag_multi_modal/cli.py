import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="MedRAG Multi-Modal CLI")
    parser.add_argument(
        "command", choices=["run", "evaluate"], help="Command to execute"
    )
    args = parser.parse_args()

    if args.command == "run":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

    elif args.command == "evaluate":
        subprocess.run(
            [sys.executable, "-m", "pytest", "-s", os.path.join("tests", "evals.py")]
        )


if __name__ == "__main__":
    main()
