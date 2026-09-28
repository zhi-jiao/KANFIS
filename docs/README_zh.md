# KANFIS 中文使用说明

本项目提供两种模型：Type-1 KANFIS（`variant="it1"`）和区间 Type-2
KANFIS（`variant="it2"`）。两者均支持单输出回归与二分类/多分类。
本实现中的 IT2 使用区间高斯前件，并对上下隶属度取中点近似；它不把区间端点传播到
输出层，也没有实现 Karnik-Mendel 类型约简。论文中应按这一准确范围描述模型。

## 论文摘要与架构

论文英文摘要见主文档的 [Abstract](../README.md#abstract) 章节。

![KANFIS 架构：单层与多层结构、T1/IT2 高斯隶属度函数及规则推理流程。](assets/kanfis-architecture.png)

*图 1. KANFIS 概念架构，包括单层与多层网络、T1/IT2 高斯隶属度函数，以及带正则化和加权聚合的规则推理流程。*

图中的语言标签用于概念示意；当前 release 导出数值规则参数，具体实现范围见本文开头的说明与规则输出接口。

## 安装

固定环境支持 Python 3.10、3.11 或 3.12。

```bash
python -m venv .venv
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install -e .
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

开发环境与测试命令：

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## 回归 `fit` 示例

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from kanfis import KANFISRegressor

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = KANFISRegressor(
    variant="it1",
    n_rules=8,
    n_memberships=3,
    max_epochs=100,
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## 分类 `fit` 示例

```python
from sklearn.datasets import load_iris
from kanfis import KANFISClassifier

data = load_iris()
model = KANFISClassifier(variant="it2", n_rules=8, random_state=42)
model.fit(data.data, data.target_names[data.target])
labels = model.predict(data.data)
probabilities = model.predict_proba(data.data)
```

## 超参数设置接口

所有超参数均可通过构造函数或 sklearn 风格接口设置：

```python
model.set_params(
    variant="it2",
    n_rules=12,
    n_memberships=4,
    learning_rate=5e-4,
    lambda_entropy=0.01,
    lambda_distinct=0.01,
)
parameters = model.get_params()
```

主要参数包括 `variant`、`n_rules`、`n_memberships`、`n_layers`、
`membership`、`learning_rate`、`batch_size`、`max_epochs`、`patience`、
`validation_fraction`、`min_delta`、`lambda_entropy`、`lambda_distinct`、
`weight_decay`、`random_state`、`device` 和 `verbose`。

## 底层 `train` 接口

底层训练接口面向已有 PyTorch DataLoader 的实验代码。输入张量需要由调用者预先完成
缩放。

```python
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from kanfis import IT1KANFISNetwork, KANFISTrainer

data = load_diabetes()
X_train, X_val, y_train, y_val = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

def make_loader(X, y, *, shuffle):
    return DataLoader(
        TensorDataset(
            torch.as_tensor(x_scaler.transform(X), dtype=torch.float32),
            torch.as_tensor(
                y_scaler.transform(y.reshape(-1, 1)), dtype=torch.float32
            ),
        ),
        batch_size=32,
        shuffle=shuffle,
    )

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val, shuffle=False)
network = IT1KANFISNetwork(
    in_features=X_train.shape[1],
    n_rules=8,
    out_features=1,
    n_memberships=3,
)
trainer = KANFISTrainer(network, task="regression", device="cpu")
history = trainer.train(train_loader, val_loader, max_epochs=100, patience=20)
```

训练循环放在独立的 `KANFISTrainer` 中，因此不会覆盖 PyTorch 原生的
`network.train(mode)`。

## 规则输出接口

```python
rules = model.get_rules(feature_names=data.feature_names, threshold=0.05)
model.export_rules(
    "outputs/rules.json",
    feature_names=data.feature_names,
    threshold=0.05,
)
model.export_rules(
    "outputs/rules.txt",
    feature_names=data.feature_names,
    threshold=0.05,
)
```

规则包含规则编号、特征名称、掩码值、隶属度函数中心与宽度以及后件权重。
IT2 规则分别给出下界宽度和上界宽度；分类规则使用原始类别标签作为后件名称。
高层估计器会反变换内部标准化，因此特征参数和回归后件使用原始数据量纲；
直接对底层网络调用 `extract_rules` 时则使用网络输入量纲。
为保证后件权重确实直接对应第一层规则，规则导出要求 `n_layers=1`；深层模型仍可用于训练和预测。

## 论文实验复现

```bash
python scripts/run_experiment.py --config configs/diabetes_it1.yaml
python scripts/run_experiment.py --config configs/diabetes_it2.yaml
python scripts/run_experiment.py --config configs/iris_it1.yaml
python scripts/run_experiment.py --config configs/iris_it2.yaml
python scripts/run_all.py
```

每个实验会在 `outputs/<dataset>/<variant>/seed_<seed>/` 下生成完整配置、
指标、运行环境版本、训练历史、JSON/文本规则以及模型检查点。批量入口额外生成
`outputs/summary.csv`。

正式报告实验结果前，应在目标环境执行测试和复现命令。

## 引用

如果在研究中使用 KANFIS，请引用：

```bibtex
@inproceedings{yong2026kanfis,
  title     = {{KANFIS}: A Neuro-Symbolic Framework for Interpretable and Uncertainty-Aware Learning},
  author    = {Yong, Binbin and Pei, Haoran and Shen, Jun and Li, Haoran and Zhou, Qingguo and Su, Zhao},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
