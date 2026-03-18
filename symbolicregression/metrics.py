# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict
import numpy as np
import scipy


def compute_metrics(infos, metrics="r2"):
    results = defaultdict(list)
    if metrics == "":
        return {}

    if "true" in infos:
        true, predicted = infos["true"], infos["predicted"]
        assert len(true) == len(predicted), "issue with len, true: {}, predicted: {}".format(len(true), len(predicted))
        for i in range(len(true)):
            if predicted[i] is None: continue
            if len(true[i].shape)==2:
                true[i]=true[i][:,0]
            if len(predicted[i].shape)==2:
                predicted[i]=predicted[i][:,0]
            assert true[i].shape == predicted[i].shape, "Problem with shapes: {}, {}".format(true[i].shape, predicted[i].shape)

    def append_prediction_metric(metric_name, fn):
        true, predicted = infos["true"], infos["predicted"]
        for i in range(len(true)):
            if predicted[i] is None or np.isnan(predicted[i]).any():
                results[metric_name].append(np.nan)
                continue
            try:
                results[metric_name].append(fn(true[i], predicted[i]))
            except Exception:
                results[metric_name].append(np.nan)

    for metric in metrics.split(","):
        if metric == "r2":
            append_prediction_metric(metric, lambda truth, prediction: r2_score(truth, prediction))
        elif metric == "r2_zero":
            append_prediction_metric(metric, lambda truth, prediction: max(0, r2_score(truth, prediction)))

        elif metric.startswith("accuracy_l1"):
            if metric == "accuracy_l1":
                atol, rtol = 0.0, 0.1
                tolerance_point = 0.95
            elif metric == "accuracy_l1_biggio":
                ## default is biggio et al.
                atol, rtol = 1e-3, 0.05
                tolerance_point = 0.95
            else:
                atol = 0 #float(metric.split("_")[-3])
                rtol = float(metric.split("_")[-1])
                tolerance_point = 0.95 #float(metric.split("_")[-1])

            append_prediction_metric(
                metric,
                lambda truth, prediction: float(
                    np.isclose(prediction, truth, atol=atol, rtol=rtol).mean() >= tolerance_point
                ),
            )

        elif metric == "_mse":
            append_prediction_metric(metric, lambda truth, prediction: mean_squared_error(truth, prediction))
        elif metric == "_nmse":
            append_prediction_metric(
                metric,
                lambda truth, prediction: np.mean(np.square(truth - prediction)) / np.mean(truth),
            )
        elif metric == "_rmse":
            append_prediction_metric(
                metric,
                lambda truth, prediction: np.sqrt(mean_squared_error(truth, prediction)),
            )
        elif metric == "_complexity":
            if "predicted_tree" not in infos: 
                results[metric].extend([np.nan for _ in range(len(infos["true"]))])
                continue
            predicted_tree = infos["predicted_tree"]
            for i in range(len(predicted_tree)):
                if predicted_tree[i] is None:
                    results[metric].append(np.nan)
                else:
                    results[metric].append(len(predicted_tree[i].prefix().split(",")))
                    
        elif metric == "_relative_complexity":
            if "tree" not in infos or "predicted_tree" not in infos: 
                results[metric].extend([np.nan for _ in range(len(infos["true"]))])
                continue
            tree = infos["tree"]
            predicted_tree = infos["predicted_tree"]
            for i in range(len(predicted_tree)):
                if predicted_tree[i] is None:
                    results[metric].append(np.nan)
                else:
                    results[metric].append(len(predicted_tree[i].prefix().split(",")) - len(tree[i].prefix().split(",")))

        elif metric == "is_symbolic_solution":
            def symbolic_solution_score(truth, prediction):
                diff = truth - prediction
                div = truth / (prediction + 1e-100)
                std_diff = scipy.linalg.norm(np.abs(diff - diff.mean(0)))
                std_div = scipy.linalg.norm(np.abs(div - div.mean(0)))
                return 1.0 if std_diff < 1e-10 and std_div < 1e-10 else 0.0

            append_prediction_metric(metric, symbolic_solution_score)

        elif metric == "_l1_error":
            def l1_error(truth, prediction):
                error = np.mean(np.abs(truth - prediction))
                return np.inf if np.isnan(error) else error

            append_prediction_metric(metric, l1_error)
    return results
