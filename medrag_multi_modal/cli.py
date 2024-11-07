import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="MedRAG Multi-Modal CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run the Streamlit application")
    run_parser.add_argument(
        "--port", type=int, default=8501, help="Port to run Streamlit on"
    )

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation tests")
    eval_parser.add_argument(
        "--test-file",
        default=os.path.join("tests", "evals", "test_assistant_mmlu_anatomy.py"),
        help="Path to test file",
    )
    eval_parser.add_argument(
        "--test-case",
        type=str,
        help="Only run tests which match the given substring expression",
    )

    args = parser.parse_args()

    if args.command == "run":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.port",
                str(args.port),
            ]
        )

    elif args.command == "evaluate":
        test_file = (
            args.test_file + "::" + args.test_case if args.test_case else args.test_file
        )
        cmd = [sys.executable, "-m", "pytest", "-s", test_file]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
