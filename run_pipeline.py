import argparse

from automl_agent.core.graph import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Agentic AutoML on a CSV/Excel dataset.")
    parser.add_argument("--data", required=True, help="Path to CSV/Excel dataset.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--name", default="dataset", help="Dataset name for tracking.")
    parser.add_argument("--retries", type=int, default=1, help="Maximum critic retries.")
    args = parser.parse_args()

    result = run_pipeline(args.data, args.target, args.name, max_retries=args.retries)
    print("\n".join(result.logs))
    if result.error:
        raise SystemExit(result.error)
    print("\nFinal metrics:", result.evaluation_result.metrics)
    print("Model:", result.ml_config.architecture)
    print("Critic:", result.critic_feedback.model_dump() if result.critic_feedback else {})


if __name__ == "__main__":
    main()

