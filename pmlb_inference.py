import argparse
import time

import pandas as pd
import torch

import symbolicregression.model
from symbolicregression.metrics import compute_metrics


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
    max_rows=200,
    max_input_points=200,
    n_trees_to_refine=100,
    rescale=True,
):
    dataset_path, df, X, y = load_pmlb_dataset(
        dataset_name=dataset_name,
        datasets_dir=datasets_dir,
        max_rows=max_rows,
    )

    start = time.time()
    est = symbolicregression.model.SymbolicTransformerRegressor(
        model=model,
        max_input_points=max_input_points,
        n_trees_to_refine=n_trees_to_refine,
        rescale=rescale,
    )
    est.fit(X, y)
    tree_info = est.retrieve_tree(with_infos=True)
    y_pred = est.predict(X, refinement_type=tree_info["refinement_type"])
    metrics = compute_metrics(
        {
            "true": [y],
            "predicted": [y_pred],
            "predicted_tree": [tree_info["predicted_tree"]],
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
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="feynman_III_10_19")
    parser.add_argument("--datasets_dir", default="pmlb/datasets")
    parser.add_argument("--model_path", default="model.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_rows", type=int, default=200)
    parser.add_argument("--max_input_points", type=int, default=200)
    parser.add_argument("--n_trees_to_refine", type=int, default=100)
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main():
    args = build_parser().parse_args()
    model = load_model(args.model_path, device=args.device)
    result = run_inference(
        dataset_name=args.dataset_name,
        model=model,
        datasets_dir=args.datasets_dir,
        max_rows=args.max_rows,
        max_input_points=args.max_input_points,
        n_trees_to_refine=args.n_trees_to_refine,
        rescale=args.rescale,
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


if __name__ == "__main__":
    main()
