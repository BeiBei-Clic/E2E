import argparse
import csv
import math
import statistics


SUMMARY_FIELDS = (
    "noise_strength",
    "group",
    "r2_mean",
    "r2_std",
    "r2_valid_count",
    "total_count",
    "recovery_rate",
    "complexity_mean",
    "complexity_std",
    "complexity_count",
    "seconds_mean",
    "seconds_std",
    "seconds_count",
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", nargs="+", required=True)
    parser.add_argument(
        "--output_csv",
        default="experiments/pmlb/results/pmlb_results_summary.csv",
    )
    return parser


def group_from_dataset_name(dataset_name):
    if dataset_name.startswith("feynman_"):
        return "Feynman"
    if dataset_name.startswith("strogatz_"):
        return "Strogatz"
    return "Black-box"


GROUP_ORDER = {"Feynman": 0, "Strogatz": 1, "Black-box": 2}


def parse_float(value):
    if value in (None, ""):
        return None
    return float(value)


def clamped_r2(raw_r2):
    if raw_r2 is None or not math.isfinite(raw_r2):
        return 0.0
    return max(0.0, raw_r2)


def finite_values(rows, field):
    values = []
    for row in rows:
        if row["status"] != "ok":
            continue
        value = parse_float(row.get(field))
        if value is None or not math.isfinite(value):
            raise ValueError(
                f"non-finite {field}={value!r} for dataset={row['dataset']}"
            )
        values.append(value)
    return values


def mean_std(values):
    if not values:
        return "", ""
    return statistics.mean(values), statistics.pstdev(values)


def summarize_rows(rows):
    raw_r2 = [parse_float(row.get("r2")) for row in rows]
    clamped = [clamped_r2(value) for value in raw_r2]
    complexity = finite_values(rows, "complexity")
    seconds = finite_values(rows, "seconds")
    complexity_mean, complexity_std = mean_std(complexity)
    seconds_mean, seconds_std = mean_std(seconds)
    return {
        "r2_mean": statistics.mean(clamped),
        "r2_std": statistics.pstdev(clamped),
        "r2_valid_count": sum(
            1 for value in raw_r2 if value is not None and math.isfinite(value) and value >= 0
        ),
        "total_count": len(rows),
        "recovery_rate": sum(1 for value in raw_r2 if value is not None and value > 0.9)
        / len(rows),
        "complexity_mean": complexity_mean,
        "complexity_std": complexity_std,
        "complexity_count": len(complexity),
        "seconds_mean": seconds_mean,
        "seconds_std": seconds_std,
        "seconds_count": len(seconds),
    }


def main():
    args = build_parser().parse_args()
    grouped = {}
    for input_path in args.input_csv:
        with open(input_path, "r", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["status"] == "skip":
                    continue
                noise_strength = parse_float(row.get("noise_strength")) or 0.0
                key = (noise_strength, group_from_dataset_name(row["dataset"]))
                grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (noise_strength, group) in sorted(
        grouped, key=lambda key: (key[0], GROUP_ORDER[key[1]])
    ):
        summary_rows.append(
            {
                "noise_strength": noise_strength,
                "group": group,
                **summarize_rows(grouped[(noise_strength, group)]),
            }
        )

    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"{'noise':<8}{'group':<10}{'r2_mean':<9}{'r2_std':<9}"
        f"{'valid':<7}{'total':<7}{'recovery':<10}{'cx_mean':<9}{'sec_mean':<9}"
    )
    for row in summary_rows:
        print(
            f"{row['noise_strength']:<8g}"
            f"{row['group']:<10}"
            f"{row['r2_mean']:<9.4f}"
            f"{row['r2_std']:<9.4f}"
            f"{row['r2_valid_count']:<7}"
            f"{row['total_count']:<7}"
            f"{row['recovery_rate']:<10.4f}"
            f"{row['complexity_mean']:<9.2f}"
            f"{row['seconds_mean']:<9.2f}"
        )


if __name__ == "__main__":
    main()
