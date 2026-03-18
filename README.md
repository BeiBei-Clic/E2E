# Deep Symbolic Regression

This repository contains code for the paper [End-to-end symbolic regression with transformers](https://arxiv.org/abs/2204.10532).
An interactive demonstration of the paper may be found [here](https://symbolicregression.metademolab.com/).

The code is based on the repository [Deep Learning for Symbolic Mathematics](https://github.com/facebookresearch/SymbolicMathematics).
Most of the code specific to recurrent sequences lies in the folder ```src/envs```.

## Install dependencies 
Using conda and the environment.yml file:

```conda env create --name symbolic regression --file=environment.yml```

Also manually install a fork of sympytorch:

```pip install git+https://github.com/pakamienny/sympytorch```


## Run the model

To launch a model training use with additional arguments (arg1,val1), (arg2,val2):

```python train.py --arg1 val1 --arg2 --val2```

All hyper-parameters related to training are specified in parsers.py, and environment HPs are in envs/environment.py

To launch evaluation, please use the flag ```reload_checkpoint``` to specify in which folder the saved model is located.
```python evaluate.py --reload_checkpoint XXX```

## Try out a pre-trained model

We include a small notebook that loads a pre-trained model you can play with in ```Example.ipynb```

## 用 uv 跑通预训练示例

推理接口在 `symbolicregression.model.SymbolicTransformerRegressor`。

```bash
uv venv -p /usr/bin/python3.10 .venv
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python numpy scipy sympy requests scikit-learn pandas numexpr matplotlib seaborn tqdm ipython jupyter sympytorch torch torchvision torchaudio
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy curl -L https://dl.fbaipublicfiles.com/symbolicregression/model1.pt -o model.pt
PYTHONPATH=. .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace Example.ipynb
```

用本地 `pmlb/datasets` 中的一个数据集抽样 200 行做推理测试：

```bash
PYTHONPATH=. .venv/bin/python pmlb_inference.py
```

先用更小参数快速跑一遍全量数据集列表，检查 CSV 保存链路：

```bash
PYTHONPATH=. .venv/bin/python pmlb_batch_inference.py --output_csv pmlb_results_smoke.csv --sample_rows 20 --n_trees_to_refine 1
```

再用默认推荐配置跑完整个数据集列表：

```bash
PYTHONPATH=. .venv/bin/python pmlb_batch_inference.py --device cuda:1
```

默认会遍历全部数据集，并且每个数据集读取前 200 个样本；测试命令只是把参数调小来快速验证。

按 Feynman、Strogatz 和其余黑盒数据集汇总 `pmlb_results.csv` 的统计结果：

```bash
PYTHONPATH=. .venv/bin/python pmlb_results_summary.py --input_csv pmlb_results.csv --output_csv pmlb_results_summary.csv
```

You can also check the demo website where you can play with the model without a single line of code [here](https://symbolicregression.metademolab.com/).

## Multinode training

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit) with grid-search:
```
pip install submitit
```

To launch a run on 2 nodes with 8 GPU each, use the ```submit.py``` script.

## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [SymPy](https://www.sympy.org/)
- [PyTorch](http://pytorch.org/) (tested on version 1.3)

## Citation

If you want to reuse this material, please considering citing the following:
```
@article{kamienny2022end,
  title={End-to-end symbolic regression with transformers},
  author={Kamienny, Pierre-Alexandre and d'Ascoli, St{\'e}phane and Lample, Guillaume and Charton, Fran{\c{c}}ois},
  journal={arXiv preprint arXiv:2204.10532},
  year={2022}
}
```

## License

The majority of this repository is released under the Apache 2.0 license as found in the LICENSE file.
