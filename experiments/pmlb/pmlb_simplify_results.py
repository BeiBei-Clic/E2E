import argparse
import csv
import os

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from experiments.pmlb.pmlb_inference import load_pmlb_dataset
from symbolicregression.envs.simplifiers import simplify as simplify_with_timeout
from symbolicregression.metrics import compute_metrics


NOISE_PATTERN = "pmlb_batch_inference_noise_"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csvs",
        nargs="+",
        default=(
            "experiments/pmlb/E2E_results/pmlb_results.csv",
            "experiments/pmlb/E2E_results/pmlb_batch_inference_noise_0.001.csv",
            "experiments/pmlb/E2E_results/pmlb_batch_inference_noise_0.01.csv",
            "experiments/pmlb/E2E_results/pmlb_batch_inference_noise_0.1.csv",
        ),
    )
    parser.add_argument("--datasets_dir", default="pmlb/datasets")
    parser.add_argument("--max_rows", type=int, default=200)
    return parser


def infer_noise_strength(input_csv):
    basename = os.path.basename(input_csv)
    if basename == "pmlb_results.csv":
        return 0.0
    if basename.startswith(NOISE_PATTERN) and basename.endswith(".csv"):
        return float(basename[len(NOISE_PATTERN) : -4])
    raise ValueError(f"无法从文件名推断 noise_strength: {input_csv}")


def default_output_csv(input_csv):
    base, ext = os.path.splitext(input_csv)
    return f"{base}_simplified{ext}"


def sympy_local_dict(max_symbols):
    local_dict = {
        "e": sp.E,
        "pi": sp.pi,
        "euler_gamma": sp.EulerGamma,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "arctan": sp.atan,
        "step": sp.Heaviside,
        "sign": sp.sign,
        "inv": lambda x: 1 / x,
        "Abs": sp.Abs,
        "sqrt": sp.sqrt,
        "exp": sp.exp,
        "log": sp.log,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "abs": sp.Abs,
    }
    for i in range(max_symbols):
        local_dict[f"x_{i}"] = sp.Symbol(f"x_{i}", real=True, integer=False)
    return local_dict


def sympy_to_prefix(expr):
    if isinstance(expr, sp.Symbol):
        return [str(expr)]
    if isinstance(expr, sp.Integer):
        return [str(expr)]
    if isinstance(expr, sp.Float):
        return [str(expr)]
    if isinstance(expr, sp.Rational):
        return ["mul", str(expr.p), "pow", str(expr.q), "-1"]
    if expr == sp.EulerGamma:
        return ["euler_gamma"]
    if expr == sp.E:
        return ["e"]
    if expr == sp.pi:
        return ["pi"]
    if expr == sp.nan:
        return ["nan"]
    if expr == sp.oo:
        return ["oo"]
    if expr == -sp.oo:
        return ["-oo"]
    if expr == sp.zoo:
        return ["zoo"]

    sympy_operators = {
        sp.Add: "add",
        sp.Mul: "mul",
        sp.Mod: "mod",
        sp.Pow: "pow",
        sp.Abs: "abs",
        sp.sign: "sign",
        sp.Heaviside: "step",
        sp.exp: "exp",
        sp.log: "log",
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",
        sp.asin: "arcsin",
        sp.acos: "arccos",
        sp.atan: "arctan",
    }
    for op_type, op_name in sympy_operators.items():
        if isinstance(expr, op_type):
            prefix = []
            for i, arg in enumerate(expr.args):
                if i == 0 or i < len(expr.args) - 1:
                    prefix.append(op_name)
                prefix.extend(sympy_to_prefix(arg))
            return prefix
    raise ValueError(f"不支持的 SymPy 表达式类型: {type(expr)}")


def expr_to_numpy(expr, x):
    symbols = [sp.Symbol(f"x_{i}", real=True, integer=False) for i in range(x.shape[1])]
    fn = sp.lambdify(symbols, expr, modules="numpy")
    values = fn(*[x[:, i] for i in range(x.shape[1])])
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(x.shape[0], float(values), dtype=float)
    return values


def simplify_one_csv(input_csv, datasets_dir, max_rows):
    output_csv = default_output_csv(input_csv)
    with open(input_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames)
        if "expr_simplified" not in fieldnames:
            expr_index = fieldnames.index("expr") + 1
            fieldnames.insert(expr_index, "expr_simplified")
        if "complexity_original" not in fieldnames:
            complexity_index = fieldnames.index("complexity") + 1
            fieldnames.insert(complexity_index, "complexity_original")
        if "complexity_simplified" not in fieldnames:
            complexity_original_index = fieldnames.index("complexity_original") + 1
            fieldnames.insert(complexity_original_index, "complexity_simplified")

        rows = list(reader)

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    noise_strength = infer_noise_strength(input_csv)
    dataset_cache = {}

    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            if row["status"] != "ok" or row["expr"] == "":
                row["expr_simplified"] = ""
                if "complexity_original" in row:
                    row["complexity_original"] = ""
                if "complexity_simplified" in row:
                    row["complexity_simplified"] = ""
                writer.writerow(row)
                continue

            dataset_name = row["dataset"]
            if dataset_name not in dataset_cache:
                _, df, x, y = load_pmlb_dataset(
                    dataset_name=dataset_name,
                    datasets_dir=datasets_dir,
                    max_rows=max_rows,
                )
                dataset_cache[dataset_name] = (df, x, y)

            df, x, y = dataset_cache[dataset_name]
            local_dict = sympy_local_dict(x.shape[1])
            expr = parse_expr(row["expr"], local_dict=local_dict, evaluate=True)
            expr_simplified = simplify_with_timeout(expr, seconds=10)
            original_prefix = ",".join(sympy_to_prefix(expr))
            simplified_prefix = ",".join(sympy_to_prefix(expr_simplified))
            complexity_original = len(original_prefix.split(","))
            complexity_simplified = len(simplified_prefix.split(","))
            y_true = y
            y_pred = expr_to_numpy(expr_simplified, x)
            metrics = compute_metrics(
                {
                    "true": [y_true],
                    "predicted": [y_pred],
                },
                metrics="r2,_rmse",
            )
            if np.isnan(metrics["r2"][0]):
                row["expr_simplified"] = str(expr_simplified)
                row["complexity_original"] = complexity_original
                row["complexity_simplified"] = complexity_simplified
                row["complexity"] = complexity_original
                if "rows" in row:
                    row["rows"] = len(df)
                if "noise_strength" in row:
                    row["noise_strength"] = noise_strength
                writer.writerow(row)
                continue

            row["expr_simplified"] = str(expr_simplified)
            row["r2"] = metrics["r2"][0]
            row["rmse"] = metrics["_rmse"][0]
            row["complexity_original"] = complexity_original
            row["complexity_simplified"] = complexity_simplified
            row["complexity"] = complexity_simplified
            if "rows" in row:
                row["rows"] = len(df)
            if "noise_strength" in row:
                row["noise_strength"] = noise_strength
            writer.writerow(row)

    print(f"saved simplified results to {output_csv}")


def main():
    args = build_parser().parse_args()
    for input_csv in args.input_csvs:
        simplify_one_csv(
            input_csv=input_csv,
            datasets_dir=args.datasets_dir,
            max_rows=args.max_rows,
        )


if __name__ == "__main__":
    main()
