import argparse
import csv
import math
import os
import re


GROUP_FIELDS = (
    "noise_strength",
    "group",
    "r2_mean",
    "r2_var",
    "r2_valid_count",
    "total_count",
    "recovery_rate",
    "complexity_mean",
    "complexity_var",
    "complexity_count",
    "seconds_mean",
    "seconds_var",
    "seconds_count",
)

GROUP_ORDER = ("Feynman", "Strogatz", "Black-box")
DEFAULT_INPUT_CSVS = (
    "experiments/pmlb/results/pmlb_batch_inference_noise_0.1.csv",
    "experiments/pmlb/results/pmlb_batch_inference_noise_0.01.csv",
    "experiments/pmlb/results/pmlb_batch_inference_noise_0.001.csv",
    "experiments/pmlb/results/pmlb_results.csv",
)
NOISE_PATTERN = re.compile(r"noise_(.+?)(?:_simplified)?\.csv$")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csvs",
        nargs="+",
        default=DEFAULT_INPUT_CSVS,
    )
    parser.add_argument("--output_csv", default="experiments/pmlb/results/pmlb_results_summary.csv")
    return parser


def dataset_group(dataset_name):
    if dataset_name.startswith("feynman_"):
        return "Feynman"
    if dataset_name.startswith("strogatz_"):
        return "Strogatz"
    return "Black-box"


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_nonnegative_finite(value):
    number = parse_float(value)
    if number is None or not math.isfinite(number) or number < 0:
        return None
    return number


def mean(values):
    if not values:
        return ""
    return sum(values) / len(values)


def variance(values):
    if not values:
        return ""
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def summarize_group(rows):
    r2_values = []
    r2_valid_count = 0
    complexity_values = []
    seconds_values = []

    for row in rows:
        r2 = parse_float(row.get("r2"))
        if r2 is not None and math.isfinite(r2) and r2 >= 0:
            r2_valid_count += 1
            r2_values.append(r2)
        else:
            r2_values.append(0.0)

        if row.get("status") == "ok":
            complexity = parse_nonnegative_finite(row.get("complexity"))
            if complexity is not None:
                complexity_values.append(complexity)
            seconds = parse_nonnegative_finite(row.get("seconds"))
            if seconds is not None:
                seconds_values.append(seconds)

    total_count = len(rows)
    recovery_count = sum(value > 0.9 for value in r2_values)

    return {
        "r2_mean": mean(r2_values),
        "r2_var": variance(r2_values),
        "r2_valid_count": r2_valid_count,
        "total_count": total_count,
        "recovery_rate": recovery_count / total_count if total_count else "",
        "complexity_mean": mean(complexity_values),
        "complexity_var": variance(complexity_values),
        "complexity_count": len(complexity_values),
        "seconds_mean": mean(seconds_values),
        "seconds_var": variance(seconds_values),
        "seconds_count": len(seconds_values),
    }


def normalize_noise_strength(noise_strength):
    number = parse_float(noise_strength)
    if number is None or not math.isfinite(number) or number < 0:
        raise ValueError(f"invalid noise strength: {noise_strength}")
    return f"{number:g}"


def infer_noise_strength(input_csv):
    basename = os.path.basename(input_csv)
    match = NOISE_PATTERN.search(basename)
    if match is not None:
        return normalize_noise_strength(match.group(1))
    if basename in ("pmlb_results.csv", "pmlb_results_simplified.csv"):
        return "0"
    raise ValueError(f"cannot infer noise strength from input file: {input_csv}")


def load_rows(input_csv):
    grouped_rows = {group: [] for group in GROUP_ORDER}
    with open(input_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped_rows[dataset_group(row.get("dataset", ""))].append(row)
    return grouped_rows


def write_summary(output_csv, summaries_by_noise):
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_FIELDS)
        writer.writeheader()
        for noise_strength, summaries in summaries_by_noise:
            for group in GROUP_ORDER:
                writer.writerow(
                    {"noise_strength": noise_strength, "group": group, **summaries[group]}
                )


def format_number(value):
    if value == "":
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def print_summary_table(summaries_by_noise):
    header = (
        "noise_strength",
        "group",
        "r2_mean",
        "r2_var",
        "r2_valid/total",
        "recovery_rate",
        "complexity_mean",
        "complexity_var",
        "seconds_mean",
        "seconds_var",
    )
    print("\t".join(header))
    for noise_strength, summaries in summaries_by_noise:
        for group in GROUP_ORDER:
            summary = summaries[group]
            row = (
                noise_strength,
                group,
                format_number(summary["r2_mean"]),
                format_number(summary["r2_var"]),
                f"{summary['r2_valid_count']}/{summary['total_count']}",
                format_number(summary["recovery_rate"]),
                format_number(summary["complexity_mean"]),
                format_number(summary["complexity_var"]),
                format_number(summary["seconds_mean"]),
                format_number(summary["seconds_var"]),
            )
            print("\t".join(row))


def main():
    args = build_parser().parse_args()
    summaries_by_noise = []
    for input_csv in args.input_csvs:
        noise_strength = infer_noise_strength(input_csv)
        grouped_rows = load_rows(input_csv)
        summaries = {
            group: summarize_group(grouped_rows[group])
            for group in GROUP_ORDER
        }
        summaries_by_noise.append((noise_strength, summaries))
    summaries_by_noise.sort(key=lambda item: parse_float(item[0]))
    write_summary(args.output_csv, summaries_by_noise)
    print_summary_table(summaries_by_noise)
    print(f"saved summary to {args.output_csv}")


if __name__ == "__main__":
    main()
