import argparse
import csv
import os

from experiments.pmlb.pmlb_inference import load_model, load_pmlb_dataset, run_inference


RESULT_FIELDS = (
    "dataset",
    "status",
    "n_features",
    "refinement_type",
    "r2",
    "rmse",
    "complexity",
    "expr",
    "seconds",
    "error",
    "rows",
)


def is_regression_dataset(datasets_dir, dataset_name):
    metadata_path = os.path.join(datasets_dir, dataset_name, "metadata.yaml")
    if not os.path.exists(metadata_path):
        return False
    with open(metadata_path, "r") as handle:
        for line in handle:
            if line.strip() == "task: regression":
                return True
    return False


def list_regression_datasets(datasets_dir):
    return [
        name
        for name in sorted(os.listdir(datasets_dir))
        if is_regression_dataset(datasets_dir, name)
    ]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", default="pmlb/datasets")
    parser.add_argument("--model_path", default="model.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_csv", default="experiments/pmlb/results/pmlb_results.csv")
    parser.add_argument("--sample_rows", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=200)
    parser.add_argument("--max_input_points", type=int, default=200)
    parser.add_argument("--n_trees_to_refine", type=int, default=100)
    parser.add_argument("--dataset_limit", type=int, default=None)
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction, default=True)
    return parser


def load_existing_results(output_path):
    """Load existing results to skip completed datasets."""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                completed.add(row["dataset"])
    return completed


def main():
    args = build_parser().parse_args()
    if args.sample_rows is not None:
        args.max_rows = args.sample_rows
        args.max_input_points = args.sample_rows

    dataset_names = list_regression_datasets(args.datasets_dir)
    if args.dataset_limit is not None:
        dataset_names = dataset_names[: args.dataset_limit]

    model = load_model(args.model_path, device=args.device)

    output_path = args.output_csv
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load existing results to skip completed datasets
    completed = load_existing_results(output_path)

    # Determine write mode: append if file exists, otherwise write new
    write_mode = "a" if completed else "w"

    with open(output_path, write_mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if write_mode == "w":
            writer.writeheader()

        for dataset_name in dataset_names:
            if dataset_name in completed:
                print(f"{dataset_name}: skipped (already completed)")
                continue
            _, _, X, _ = load_pmlb_dataset(
                dataset_name=dataset_name,
                datasets_dir=args.datasets_dir,
                max_rows=args.max_rows,
            )
            if X.shape[1] > 10:
                print(f"{dataset_name}: skipped (n_features={X.shape[1]} > 10)")
                continue

            row = {field: "" for field in RESULT_FIELDS}
            row.update({"dataset": dataset_name, "status": "ok"})
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
                row.update({field: result[field] for field in RESULT_FIELDS if field in result})
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
            writer.writerow(row)
            handle.flush()
            print(f"{row['dataset']}: {row['status']}")


if __name__ == "__main__":
    main()
