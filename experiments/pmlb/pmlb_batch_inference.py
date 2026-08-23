import argparse
import csv
import os

import pandas as pd

from experiments.pmlb.pmlb_inference import (
    list_regression_datasets,
    load_model,
    run_inference,
)


RESULT_FIELDS = (
    "dataset",
    "status",
    "n_features",
    "refinement_type",
    "r2",
    "rmse",
    "complexity",
    "seconds",
    "error",
    "noise_strength",
    "expr",
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", default="pmlb/datasets")
    parser.add_argument("--model_path", default="model.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--sample_rows", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_input_points", type=int, default=200)
    parser.add_argument("--max_number_bags", type=int, default=100)
    parser.add_argument("--n_trees_to_refine", type=int, default=10)
    parser.add_argument("--dataset_limit", type=int, default=None)
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise_strength", type=float, default=0.0)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=29910)
    return parser


def format_noise_strength_for_filename(noise_strength):
    return f"{noise_strength:g}"


def default_output_csv(noise_strength):
    noise_token = format_noise_strength_for_filename(noise_strength)
    return (
        "experiments/pmlb/results/"
        f"pmlb_batch_inference_noise_{noise_token}.csv"
    )


def normalize_noise_strength(value):
    if value in (None, ""):
        return 0.0
    return float(value)


def load_existing_results(output_path):
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                completed.add(
                    (
                        row["dataset"],
                        normalize_noise_strength(row.get("noise_strength")),
                    )
                )
    return completed


def main():
    args = build_parser().parse_args()
    if args.noise_strength < 0:
        raise ValueError("noise_strength must be non-negative.")
    if args.sample_rows is not None:
        args.max_rows = args.sample_rows
        args.max_input_points = args.sample_rows

    dataset_names = list_regression_datasets(args.datasets_dir)
    dataset_indices = {name: index for index, name in enumerate(dataset_names)}
    if args.dataset_limit is not None:
        dataset_names = dataset_names[: args.dataset_limit]

    model = load_model(args.model_path, device=args.device)

    output_path = args.output_csv or default_output_csv(args.noise_strength)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    completed = load_existing_results(output_path)
    write_mode = "a" if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else "w"

    with open(output_path, write_mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if write_mode == "w":
            writer.writeheader()

        for dataset_name in dataset_names:
            completed_key = (dataset_name, args.noise_strength)
            if completed_key in completed:
                print(
                    f"{dataset_name}: skipped "
                    f"(already completed for noise_strength={args.noise_strength:g})"
                )
                continue
            dataset_path = os.path.join(
                args.datasets_dir, dataset_name, f"{dataset_name}.tsv.gz"
            )
            n_features = pd.read_csv(dataset_path, sep="\t", nrows=0).shape[1] - 1

            row = {field: "" for field in RESULT_FIELDS}
            row.update(
                {
                    "dataset": dataset_name,
                    "status": "ok",
                    "noise_strength": args.noise_strength,
                }
            )
            if n_features > 10:
                row["status"] = "skip"
                row["n_features"] = n_features
                writer.writerow(row)
                handle.flush()
                print(f"{dataset_name}: skipped (n_features={n_features} > 10)")
                continue
            try:
                result = run_inference(
                    dataset_name=dataset_name,
                    model=model,
                    datasets_dir=args.datasets_dir,
                    max_rows=args.max_rows,
                    max_input_points=args.max_input_points,
                    max_number_bags=args.max_number_bags,
                    n_trees_to_refine=args.n_trees_to_refine,
                    rescale=args.rescale,
                    noise_strength=args.noise_strength,
                    noise_seed=args.noise_seed,
                    dataset_index=dataset_indices[dataset_name],
                    random_state=args.random_state,
                )
                row.update({field: result[field] for field in RESULT_FIELDS if field in result})
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
            writer.writerow(row)
            handle.flush()
            print(f"{row['dataset']}: {row['status']}")


if __name__ == "__main__":
    main()
