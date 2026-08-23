import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import symbolicregression.model
from symbolicregression.metrics import compute_metrics


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


def format_expr(tree):
    expr = tree.infix()
    replacements = {
        " add ": " + ",
        " sub ": " - ",
        " mul ": " * ",
        " pow ": " ** ",
    }
    for old, new in replacements.items():
        expr = expr.replace(old, new)
    return expr


def load_pmlb_dataset(dataset_name, datasets_dir="pmlb/datasets", max_rows=200):
    dataset_path = f"{datasets_dir}/{dataset_name}/{dataset_name}.tsv.gz"
    df = pd.read_csv(dataset_path, sep="\t")
    if max_rows is not None:
        df = df.iloc[:max_rows]
    df = df.copy()
    X = df.drop(columns=["target"]).to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=float)
    return dataset_path, df, X, y


def apply_target_noise(y, noise_strength=0.0, noise_seed=0):
    if noise_strength < 0:
        raise ValueError("noise_strength must be non-negative.")
    if noise_strength == 0:
        return y
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(0, noise_strength, size=y.shape)
    return y * (1 + noise)


def resolve_device(device):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the current environment.")
        if ":" in device:
            index = int(device.split(":", 1)[1])
        else:
            index = 0
            device = "cuda:0"
        torch.cuda.set_device(index)
    return torch.device(device)


def load_model(model_path="model.pt", device="cpu"):
    torch_device = resolve_device(device)
    return torch.load(
        model_path,
        map_location=torch_device,
        weights_only=False,
    )


def run_inference(
    dataset_name,
    model,
    datasets_dir="pmlb/datasets",
    max_rows=None,
    max_input_points=200,
    max_number_bags=100,
    n_trees_to_refine=10,
    rescale=True,
    noise_strength=0.0,
    noise_seed=0,
    dataset_index=0,
    random_state=29910,
):
    dataset_path, df, X, y = load_pmlb_dataset(
        dataset_name=dataset_name,
        datasets_dir=datasets_dir,
        max_rows=max_rows,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    y_train_fit = apply_target_noise(
        y_train,
        noise_strength=noise_strength,
        noise_seed=noise_seed + dataset_index,
    )

    start = time.time()
    est = symbolicregression.model.SymbolicTransformerRegressor(
        model=model,
        max_input_points=max_input_points,
        max_number_bags=max_number_bags,
        n_trees_to_refine=n_trees_to_refine,
        rescale=rescale,
    )
    est.fit(X_train, y_train_fit)
    tree_info = est.retrieve_tree(with_infos=True)
    y_pred = est.predict(X_test, refinement_type=tree_info["refinement_type"])
    metrics = compute_metrics(
        {
            "true": [y_test],
            "predicted": [y_pred],
            "predicted_tree": [tree_info["predicted_tree_standardized"]],
        },
        metrics="r2,_rmse,_complexity",
    )
    elapsed = time.time() - start

    return {
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "rows": len(df),
        "n_features": X.shape[1],
        "refinement_type": tree_info["refinement_type"],
        "expr": format_expr(tree_info["relabed_predicted_tree"]),
        "r2": metrics["r2"][0],
        "rmse": metrics["_rmse"][0],
        "complexity": metrics["_complexity"][0],
        "seconds": elapsed,
        "noise_strength": noise_strength,
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="feynman_III_10_19")
    parser.add_argument("--datasets_dir", default="pmlb/datasets")
    parser.add_argument("--model_path", default="model.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_input_points", type=int, default=200)
    parser.add_argument("--max_number_bags", type=int, default=100)
    parser.add_argument("--n_trees_to_refine", type=int, default=10)
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise_strength", type=float, default=0.0)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=29910)
    return parser


def main():
    args = build_parser().parse_args()
    model = load_model(args.model_path, device=args.device)
    dataset_names = list_regression_datasets(args.datasets_dir)
    result = run_inference(
        dataset_name=args.dataset_name,
        model=model,
        datasets_dir=args.datasets_dir,
        max_rows=args.max_rows,
        max_input_points=args.max_input_points,
        max_number_bags=args.max_number_bags,
        n_trees_to_refine=args.n_trees_to_refine,
        rescale=args.rescale,
        noise_strength=args.noise_strength,
        noise_seed=args.noise_seed,
        dataset_index=dataset_names.index(args.dataset_name),
        random_state=args.random_state,
    )

    print(f"dataset={result['dataset']}")
    print(f"dataset_path={result['dataset_path']}")
    print(f"rows={result['rows']}")
    print(f"n_features={result['n_features']}")
    print(f"refinement_type={result['refinement_type']}")
    print(f"expr={result['expr']}")
    print(f"r2={result['r2']}")
    print(f"rmse={result['rmse']}")
    print(f"complexity={result['complexity']}")
    print(f"seconds={result['seconds']}")
    print(f"noise_strength={result['noise_strength']}")


if __name__ == "__main__":
    main()
