import argparse
import csv
import os

import yaml

from pmlb_inference import load_model, run_inference


def list_regression_datasets(datasets_dir):
    dataset_names = []
    for name in sorted(os.listdir(datasets_dir)):
        metadata_path = os.path.join(datasets_dir, name, "metadata.yaml")
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path, "r") as handle:
            metadata = yaml.safe_load(handle) or {}
        if metadata.get("task") == "regression":
            dataset_names.append(name)
    return dataset_names


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", default="pmlb/datasets")
    parser.add_argument("--model_path", default="model.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_csv", default="pmlb_results.csv")
    parser.add_argument("--max_rows", type=int, default=200)
    parser.add_argument("--max_input_points", type=int, default=200)
    parser.add_argument("--n_trees_to_refine", type=int, default=100)
    parser.add_argument("--dataset_limit", type=int, default=None)
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main():
    args = build_parser().parse_args()
    dataset_names = list_regression_datasets(args.datasets_dir)
    if args.dataset_limit is not None:
        dataset_names = dataset_names[: args.dataset_limit]

    model = load_model(args.model_path, device=args.device)
    fieldnames = [
        "dataset",
        "status",
        "rows",
        "n_features",
        "refinement_type",
        "expr",
        "r2",
        "rmse",
        "complexity",
        "seconds",
        "error",
    ]

    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_name in dataset_names:
            row = {
                "dataset": dataset_name,
                "status": "ok",
                "rows": "",
                "n_features": "",
                "refinement_type": "",
                "expr": "",
                "r2": "",
                "rmse": "",
                "complexity": "",
                "seconds": "",
                "error": "",
            }
            try:
                result = run_inference(
                    dataset_name=dataset_name,
                    model=model,
                    datasets_dir=args.datasets_dir,
                    max_rows=args.max_rows,
                    max_input_points=args.max_input_points,
                    n_trees_to_refine=args.n_trees_to_refine,
                    rescale=args.rescale,
                )
                row.update(
                    {
                        "rows": result["rows"],
                        "n_features": result["n_features"],
                        "refinement_type": result["refinement_type"],
                        "expr": result["expr"],
                        "r2": result["r2"],
                        "rmse": result["rmse"],
                        "complexity": result["complexity"],
                        "seconds": result["seconds"],
                    }
                )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
            writer.writerow(row)
            handle.flush()
            print(f"{row['dataset']}: {row['status']}")


if __name__ == "__main__":
    main()
