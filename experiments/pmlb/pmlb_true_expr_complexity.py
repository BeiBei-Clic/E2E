import argparse
import csv
import os
import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from experiments.pmlb.pmlb_batch_inference import list_regression_datasets
from symbolicregression.envs.simplifiers import Simplifier


SUMMARY_FIELDS = (
    "category",
    "formula_count",
    "dataset_count",
    "average_complexity",
)

DETAIL_FIELDS = (
    "dataset",
    "category",
    "status",
    "raw_formula",
    "normalized_expr",
    "complexity",
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", default="pmlb/datasets")
    parser.add_argument(
        "--output_csv",
        default="experiments/pmlb/results/pmlb_true_expr_complexity_summary.csv",
    )
    parser.add_argument(
        "--detail_csv",
        default="experiments/pmlb/results/pmlb_true_expr_complexity_details.csv",
    )
    return parser


def category_from_dataset_name(dataset_name):
    if dataset_name.startswith("feynman_"):
        return "Feynman"
    if dataset_name.startswith("strogatz_"):
        return "Strogatz"
    if dataset_name.startswith("first_principles_"):
        return "First-principles"
    raise ValueError(f"Unsupported dataset category: {dataset_name}")


def normalize_formula_text(dataset_name, formula_text):
    normalized = formula_text.strip()
    normalized = normalized.replace("{", "(").replace("}", ")")
    normalized = normalized.replace("^", "**")
    normalized = normalized.replace("'", "")
    normalized = normalized.replace("\\Delta V(0)", "DV")
    normalized = normalized.replace("\\DeltaV(0)", "DV")
    normalized = normalized.replace("\\propto", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if dataset_name == "first_principles_tully_fisher":
        normalized = "DV**2.5"
    return normalized


def build_local_dict(formula_text):
    local_dict = {
        "ln": sp.log,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "cot": sp.cot,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "arctan": sp.atan,
        "sqrt": sp.sqrt,
        "exp": sp.exp,
        "log": sp.log,
        "Abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
    }
    for identifier in sorted(set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula_text))):
        if identifier in local_dict:
            continue
        local_dict[identifier] = sp.Symbol(identifier, real=True)
    return local_dict


def main():
    args = build_parser().parse_args()
    dataset_names = [
        dataset_name
        for dataset_name in list_regression_datasets(args.datasets_dir)
        if dataset_name.startswith("feynman_")
        or dataset_name.startswith("strogatz_")
        or dataset_name.startswith("first_principles_")
    ]
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    simplifier = Simplifier.__new__(Simplifier)
    detail_rows = []
    summary_rows = []
    grouped = {
        "Feynman": {"formula_count": 0, "dataset_count": 0, "complexity_sum": 0},
        "Strogatz": {"formula_count": 0, "dataset_count": 0, "complexity_sum": 0},
        "First-principles": {
            "formula_count": 0,
            "dataset_count": 0,
            "complexity_sum": 0,
        },
    }

    for dataset_name in dataset_names:
        category = category_from_dataset_name(dataset_name)
        grouped[category]["dataset_count"] += 1
        metadata_path = os.path.join(args.datasets_dir, dataset_name, "metadata.yaml")
        with open(metadata_path, "r") as handle:
            metadata_lines = handle.read().splitlines()

        description_lines = []
        in_description = False
        for line in metadata_lines:
            if line.startswith("description: |"):
                in_description = True
                continue
            if in_description and line and not line.startswith(" "):
                break
            if in_description:
                description_lines.append(line.rstrip())

        formula_lhs = ""
        formula_rhs = ""
        formula_operator = ""
        for line in description_lines:
            stripped = line.strip()
            equal_matches = re.findall(
                r"([A-Za-z][A-Za-z0-9_]*(?:\([^)]+\))?'?(?:\s*\^\s*[-+]?\d+(?:\.\d+)?)?)\s*=\s*([^,\n]+)",
                stripped,
            )
            if equal_matches:
                formula_lhs, formula_rhs = equal_matches[0]
                formula_operator = "="
                break
            if "\\propto" in stripped:
                proportional_match = re.search(
                    r"([A-Za-z][A-Za-z0-9_\\ ()]*)\s*\\propto\s*([^,\n]+)",
                    stripped,
                )
                if proportional_match is None:
                    raise ValueError(f"Failed to parse proportional formula in {metadata_path}")
                formula_lhs = proportional_match.group(1).strip()
                formula_rhs = proportional_match.group(2).strip()
                formula_operator = "\\propto"
                break

        if formula_rhs == "":
            detail_rows.append(
                {
                    "dataset": dataset_name,
                    "category": category,
                    "status": "missing_formula",
                    "raw_formula": "",
                    "normalized_expr": "",
                    "complexity": "",
                }
            )
            continue

        normalized_lhs = normalize_formula_text(dataset_name, formula_lhs)
        normalized_rhs = normalize_formula_text(dataset_name, formula_rhs)
        raw_formula = f"{formula_lhs} {formula_operator} {formula_rhs}"
        parse_local_dict = build_local_dict(f"{normalized_lhs} {normalized_rhs}")
        lhs_expr = parse_expr(
            normalized_lhs,
            local_dict=parse_local_dict,
            transformations=transformations,
            evaluate=True,
        )
        rhs_expr = parse_expr(
            normalized_rhs,
            local_dict=parse_local_dict,
            transformations=transformations,
            evaluate=True,
        )

        final_expr = rhs_expr
        lhs_symbols = sorted(lhs_expr.free_symbols, key=str)
        if len(lhs_symbols) == 1 and lhs_expr != lhs_symbols[0]:
            solutions = sp.solve(sp.Eq(lhs_expr, rhs_expr), lhs_symbols[0])
            if len(solutions) > 0:
                candidate_rows = []
                for solution in solutions:
                    candidate_prefix = simplifier.sympy_to_prefix(solution)
                    candidate_rows.append(
                        (
                            str(solution).startswith("-"),
                            len(candidate_prefix),
                            str(solution),
                            solution,
                        )
                    )
                candidate_rows.sort()
                final_expr = candidate_rows[0][3]

        complexity = len(simplifier.sympy_to_prefix(final_expr))
        grouped[category]["formula_count"] += 1
        grouped[category]["complexity_sum"] += complexity
        detail_rows.append(
            {
                "dataset": dataset_name,
                "category": category,
                "status": "ok",
                "raw_formula": raw_formula,
                "normalized_expr": str(final_expr),
                "complexity": complexity,
            }
        )

    for category in ("Feynman", "Strogatz", "First-principles"):
        category_stats = grouped[category]
        average_complexity = ""
        if category_stats["formula_count"] > 0:
            average_complexity = (
                category_stats["complexity_sum"] / category_stats["formula_count"]
            )
        summary_rows.append(
            {
                "category": category,
                "formula_count": category_stats["formula_count"],
                "dataset_count": category_stats["dataset_count"],
                "average_complexity": average_complexity,
            }
        )

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    detail_dir = os.path.dirname(args.detail_csv)
    if detail_dir:
        os.makedirs(detail_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(args.detail_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(detail_rows)

    print("category           formula_count/dataset_count   average_complexity")
    for row in summary_rows:
        average_display = ""
        if row["average_complexity"] != "":
            average_display = f"{row['average_complexity']:.6f}"
        print(
            f"{row['category']:<18} "
            f"{row['formula_count']}/{row['dataset_count']:<20} "
            f"{average_display}"
        )

    missing_datasets = [
        row["dataset"] for row in detail_rows if row["status"] == "missing_formula"
    ]
    if missing_datasets:
        print("\nmissing_formula_datasets")
        for dataset_name in missing_datasets:
            print(dataset_name)


if __name__ == "__main__":
    main()
