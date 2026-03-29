from __future__ import annotations

import argparse

from pet_researcher.autonomous import AutonomousSettings, run_autonomous_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the autonomous Oxford-IIIT Pet research loop.")
    parser.add_argument("--max-experiments", type=int, default=8)
    parser.add_argument("--target-val-acc", type=float, default=0.95)
    parser.add_argument("--top-k-finalists", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="./runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = AutonomousSettings(
        max_experiments=args.max_experiments,
        target_val_acc=args.target_val_acc,
        top_k_finalists=args.top_k_finalists,
        output_dir=args.output_dir,
    )
    result = run_autonomous_loop(settings)
    print(result["state"])
    print(result["final_report"])


if __name__ == "__main__":
    main()
