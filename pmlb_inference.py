import time

import pandas as pd
import torch

import symbolicregression.model
from symbolicregression.metrics import compute_metrics


dataset_name = "feynman_III_10_19"
dataset_path = f"pmlb/datasets/{dataset_name}/{dataset_name}.tsv.gz"

df = pd.read_csv(dataset_path, sep="\t").iloc[:200].copy()
X = df.drop(columns=["target"]).to_numpy(dtype=float)
y = df["target"].to_numpy(dtype=float)

start = time.time()
model = torch.load("model.pt", map_location=torch.device("cpu"), weights_only=False)
est = symbolicregression.model.SymbolicTransformerRegressor(
    model=model,
    max_input_points=200,
    n_trees_to_refine=100,
    rescale=True,
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
    metrics="r2,_rmse",
)
elapsed = time.time() - start

print(f"dataset={dataset_name}")
print(f"rows={len(df)}")
print(f"refinement_type={tree_info['refinement_type']}")
print(f"expr={tree_info['relabed_predicted_tree'].infix()}")
print(f"r2={metrics['r2'][0]}")
print(f"rmse={metrics['_rmse'][0]}")
print(f"seconds={elapsed}")
