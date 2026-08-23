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

推理接口在 `symbolicregression.model.SymbolicTransformerRegressor`。系统缺 python3.10-dev 头文件，新版 torch 的 triton JIT 会编译失败，安装时把 torch 钉在 2.10.0。

```bash
uv venv -p /usr/bin/python3.10 .venv
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python numpy scipy sympy requests scikit-learn pandas numexpr matplotlib seaborn tqdm ipython jupyter sympytorch torch==2.10.0 torchvision torchaudio
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy curl -L https://dl.fbaipublicfiles.com/symbolicregression/model1.pt -o model.pt
PYTHONPATH=. .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace Example.ipynb
```

PMLB 相关脚本和结果统一放在 `experiments/pmlb/` 下管理。

单数据集推理：

```bash
PYTHONPATH=. .venv/bin/python experiments/pmlb/pmlb_inference.py --device cuda:0
```

批量推理先用小参数验证链路，特征数超过 10 的数据集会写 `status=skip` 行且不参与汇总：

```bash
PYTHONPATH=. .venv/bin/python experiments/pmlb/pmlb_batch_inference.py --device cpu --output_csv /tmp/pmlb_smoke.csv --sample_rows 20 --dataset_limit 6 --n_trees_to_refine 1
```

正式全量跑完整个数据集列表，需要噪声时在命令末尾追加 `--noise_strength` 和 `--noise_seed`，已跑过的数据集自动跳过：

```bash
PYTHONPATH=. .venv/bin/python experiments/pmlb/pmlb_batch_inference.py --device cuda:2
```

鲁棒性实验共四档噪声强度 0、0.001、0.01、0.1，0 档即上面的默认命令，0.1 档已跑完，还需补跑两档：

```bash
PYTHONPATH=. .venv/bin/python experiments/pmlb/pmlb_batch_inference.py --device cuda:4 --noise_strength 0.001
PYTHONPATH=. .venv/bin/python experiments/pmlb/pmlb_batch_inference.py --device cuda:5 --noise_strength 0.01
```

按 Feynman、Strogatz、Black-box 三组汇总 r2 均值/标准差、复原率（r2>0.9 占比）、复杂度和时间：

```bash
PYTHONPATH=. .venv/bin/python experiments/pmlb/pmlb_results_summary.py --input_csv experiments/pmlb/results/pmlb_batch_inference_noise_0.csv experiments/pmlb/results/pmlb_batch_inference_noise_0.001.csv experiments/pmlb/results/pmlb_batch_inference_noise_0.01.csv experiments/pmlb/results/pmlb_batch_inference_noise_0.1.csv --output_csv experiments/pmlb/results/pmlb_results_summary.csv
```

统计真值公式的复杂度，口径与推理结果一致可直接对比：

```bash
PYTHONPATH=. .venv/bin/python experiments/pmlb/pmlb_true_expr_complexity.py
```

复杂度统一按标准化输入空间的模型树统计前缀节点数，逆标准化引入的缩放常数不计入；`expr` 列保存的是映射回原始尺度的表达式，因此会包含缩放常数。

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
