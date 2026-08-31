# 红外反射膜结构优化程序

根据文本膜系文件（固定 n,k）与波段反射率目标，用 **TMM（传输矩阵法）** 计算光谱，以 **Adam / LM / DE / 模拟退火** 优化各镀膜层厚度（入射介质与基底厚度固定）。

## 快速开始

```bash
cd /path/to/IR-reflective-film
python3 -m venv sim/.venv
sim/.venv/bin/pip install -r sim/requirements.txt
sim/.venv/bin/python sim/optimize_film.py \
  sim/examples/example_stack.txt \
  sim/examples/example_optimize_film.json
```

结果默认写到 `sim/out/optimize_example/`：

| 文件 | 内容 |
|------|------|
| `rt_before_after.png` | 优化前后反射率/透射率对比 |
| `rt_best.png` / `rt_final.png` | 最优与最终迭代光谱 |
| `spectrum_before_after.csv` | 同上光谱数值 |
| `stack_best.txt` / `stack_final.txt` | 优化后膜系（文本格式，与 `plot_rt_txt.py` 兼容） |
| `loss_history.csv` | 迭代/epoch 损失曲线 |

不传参数时，默认使用 `example_stack.txt` 与 `example_optimize_film.json`。

## 功能概览

1. **文本膜系输入**：`index material thickness_nm n k` 格式（见 `plot_rt_txt.py`）
2. **多波段 R 目标**：每段设 `objective: maximize|minimize`（对应 R→1 / R→0）
3. **归一化损失**：波段内均值误差，与采样点数和绝对权重无关
4. **可选正则**：`smooth_weight`（平滑）、`ripple_weight`（抑制纹波）、`error_power`（>2 时加重离群点）
5. **优化方法**：`adam`（默认）、`lm`、`de`、`dual_annealing`；Adam 支持波长 mini-batch
6. **绘图**：matplotlib 输出优化前后 R/T 与波段着色

## 目录结构

```
sim/
  optimize_film.py      # 入口：文本膜系 + JSON 配置、优化、出图
  plot_rt_txt.py        # 文本膜系 R/T 计算与绘图
  plot_rt.py            # JSON 膜系 R/T 计算与绘图（含共享绘图工具）
  rt_calculator.py      # TMM / 外部 R/T 接口
  lm_optimizer.py       # LM / Adam / DE 厚度优化核心
  needle.py             # 逐层增加法（独立模块，供高级设计使用）
  tmm.py                # 传输矩阵核心
  dispersion.py         # 材料色散（plot_rt.py JSON 工作流使用）
  design.py             # 既有 OMO/OMOMO 评估脚本（无优化）
  requirements.txt
  examples/
    example_stack.txt              # 文本膜系示例
    example_optimize_film.json     # 优化配置示例
    example_plot_rt.json           # JSON 膜系绘图示例
    external_rt_stub.py            # 外部引擎桩
```

## 输入文件格式

### 膜系文本（`example_stack.txt`）

```
# index  material  thickness_nm  n  k
0  air      0    1.0  0
1  tio2    55    2.4  0
2  sio2    85    1.46 0
...
```

首行入射介质、末行基底厚度不参与优化。

### 优化配置（JSON）

```json
{
  "n_bands": 3,
  "method": "adam",
  "bands": [
    {"wavelength_nm": [420, 700], "objective": "minimize", "weight": 2.0},
    {"wavelength_nm": [800, 1200], "objective": "maximize", "weight": 1.5},
    {"wavelength_nm": [1400, 1800], "objective": "maximize", "weight": 1.0}
  ],
  "wavelength_step_nm": 15,
  "max_iter": 40,
  "error_power": 4.0,
  "smooth_weight": 0.5,
  "ripple_weight": 0.2,
  "adam_lr_nm": 2.0,
  "incident_angle_deg": 0,
  "polarization": "unpolarized",
  "plot_wavelength_nm": [400, 1800],
  "plot_step_nm": 5,
  "output_dir": "../out/optimize_example",
  "mini_batch": false
}
```

要点：

- **`method`**：`adam` / `lm` / `de` / `dual_annealing`（后两者需 scipy）
- **`mini_batch`**：`true` 或嵌套对象 `{"batch_size", "n_batches", "n_epochs", "shuffle_seed"}`，仅 `method=adam` 时生效
- **`checkpoint_on_best`**（默认 `true`）：运行中 best 变好时更新 `stack_best.txt`，并追加 `best_updates.csv`
- **`use_cuda`**：`true` 时走 CuPy 批量 TMM（仅 NVIDIA CUDA；macOS 不可用）

## 相关工具

```bash
# 文本膜系 R/T 光谱
sim/.venv/bin/python sim/plot_rt_txt.py sim/examples/example_stack.txt 400 1800

# JSON 膜系（材料库色散）R/T 光谱
sim/.venv/bin/python sim/plot_rt.py sim/examples/example_plot_rt.json
```

## 依赖

- Python 3.10+（推荐）
- `matplotlib`（见 `requirements.txt`）
- `scipy`（可选；`method=de` / `dual_annealing` 需要）
- `cupy`（可选；`use_cuda: true`，仅 NVIDIA CUDA）

既有基线评估仍可无额外依赖：

```bash
python3 sim/design.py
```

## 与文档的关系

项目级说明见仓库根目录 [README.md](../README.md) 与 `docs/`。算法设计详见 [光学薄膜层数与厚度联合优化算法设计](../docs/film-optimization-algorithm.md)（含合成/精修分类、算法选型、可选路径 A–I 与配置示例）。本程序侧重**可脚本化的膜系厚度/层数自动设计**；OghmaNano 等 GUI 仿真实操见 `docs/oghmanano-ir-film-simulation.md`。
