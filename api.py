
import argparse
import json
import subprocess
import sys

from .service import init_db
from .executor import train_csv

def main():
    parser = argparse.ArgumentParser(prog="asterion")
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # Existing training command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model from a CSV dataset.",
    )
    train_parser.add_argument("--csv", required=True)
    train_parser.add_argument(
        "--min-f1",
        type=float,
        default=None,
    )

    # New monitoring command
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Start the telemetry monitoring agent.",
    )
    monitor_parser.add_argument(
        "--central-url",
        default="http://127.0.0.1:8765",
    )
    monitor_parser.add_argument(
        "--agent-id",
        default="host-01",
    )
    monitor_parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
    )
    monitor_parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of collection cycles; 0 runs continuously.",
    )

    args = parser.parse_args()
    init_db()

    if args.command == "train":
        result = train_csv(args.csv, args.min_f1)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.command == "monitor":
        if args.interval < 1:
            parser.error("--interval must be at least 1 second.")

        if args.cycles < 0:
            parser.error("--cycles cannot be negative.")

        agent_args = [
            sys.executable,
            "-m",
            "Asterion.init",
            "--central-url",
            args.central_url,
            "--agent-id",
            args.agent_id,
            "--interval",
            str(args.interval),
            "--cycles",
            str(args.cycles),
        ]

        try:
            subprocess.run(agent_args, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Agent exited with status {exc.returncode}."
            ) from exc
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")


if __name__ == "__main__":
    main()
