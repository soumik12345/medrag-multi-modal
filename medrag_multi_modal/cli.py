import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="MedRAG Multi-Modal CLI")
    parser.add_argument("command", choices=["run"], help="Command to execute")
    args = parser.parse_args()

    if args.command == "run":
        # Assuming your Streamlit app is in app.py
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
