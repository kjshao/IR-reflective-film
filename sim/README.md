# 红外反射膜结构优化程序

根据文本膜系文件与波段反射率目标，用 **TMM（传输矩阵法）** 计算光谱。默认采用**多起点 TRF（有界信赖域最小二乘）+ 必要时 DE 全局回退**；也支持 LM、Adam、CG、L-BFGS-B、DE、双退火及灵敏度 Needle 层数合成。

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
| `rt_before_after.png` | 优化前后反射率/透射率对比（含实际使用的 n,k） |
| `nk_used.png` / `nk_used.csv` | 膜系实际使用的材料 n(λ),k(λ) |
| `rt_best.png` / `rt_final.png` | 最优与最终迭代光谱 |
| `spectrum_before_after.csv` | 同上光谱数值 |
| `stack_best.txt` / `stack_final.txt` | 优化后膜系（文本格式，与 `plot_rt_txt.py` 兼容） |
| `loss_history.csv` | 迭代/epoch 损失曲线 |

不传参数时，默认使用 `example_stack.txt` 与 `example_optimize_film.json`。

## 功能概览

1. **文本膜系输入**：`index material thickness_nm n k` 格式（见 `plot_rt_txt.py`）
2. **多波段 R 目标**：每段设 `R_target`（0–1），或 `objective: maximize|minimize`（默认 1 / 0）
3. **统一损失**：波段归一化 MSE（相对各段 `R_target`）+ 可选 `thickness_weight`；所有算法优化并报告同一数值
4. **优化方法**：`auto`（默认）、`multistart`、`trf`、`lm`、`adam`、`cg`、`lbfgs`、`de`、`dual_annealing`
5. **层数合成**：`use_needle: true` 时在各界面探测插层灵敏度，对 top-k 候选局部精修，最后尝试剪除薄层
6. **绘图**：matplotlib 输出优化前后 R/T、波段着色，以及实际使用的 n,k

## 目录结构

```
sim/
  optimize_film.py      # 入口：文本膜系 + JSON 配置、优化、出图
  multistart_optimize.py # 按参数生成指定层数 H/L 膜系并执行多起点优化
  plot_rt_txt.py        # 文本膜系 R/T 计算与绘图
  plot_rt.py            # JSON 膜系 R/T 计算与绘图（含共享绘图工具）
  rt_calculator.py      # TMM / 外部 R/T 接口
  lm_optimizer.py       # TRF / LM / Adam / DE 等厚度优化核心
  needle.py             # 灵敏度插层、Deep Search 候选与剪枝
  tmm.py                # 传输矩阵核心
  dispersion.py         # 材料色散库（n+ik；见 materials/SOURCES.md）
  materials/            # PVD 优先的表列 n,k（SOURCES.md；NK_TABLES_300_1800.md）
  design.py             # 既有 OMO/OMOMO 评估脚本（无优化）
  requirements.txt
  examples/
    example_stack.txt              # 文本膜系示例
    example_optimize_film.json     # 优化配置示例
    example_multistart_optimize.json # 自动生成膜系的多起点配置
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
  "method": "auto",
  "multistart_method": "trf",
  "multistart_n": 6,
  "auto_de_fallback": true,
  "bands": [
    {"wavelength_nm": [420, 700], "objective": "minimize", "R_target": 0.05, "weight": 2.0},
    {"wavelength_nm": [800, 1200], "objective": "maximize", "R_target": 0.95, "weight": 1.5},
    {"wavelength_nm": [1400, 1800], "objective": "maximize", "R_target": 0.90, "weight": 1.0}
  ],
  "wavelength_step_nm": 15,
  "max_iter": 40,
  "thickness_weight": 0.0,
  "adam_lr_nm": 2.0,
  "min_thickness_nm": 8,
  "max_total_thickness_nm": 1800,
  "thickness_bounds_nm": {
    "tio2": [8, 500],
    "sio2": [8, 550]
  },
  "multistart_sampling_bounds_nm": {
    "tio2": [40, 180],
    "sio2": [60, 250]
  },
  "nk_source": "library",
  "incident_angle_deg": 0,
  "polarization": "unpolarized",
  "plot_wavelength_nm": [400, 1800],
  "plot_step_nm": 5,
  "output_dir": "../out/optimize_example",
  "mini_batch": false
}
```

要点：

- **`method`**：`auto` / `multistart` / `trf` / `lm` / `adam` / `cg` / `lbfgs` / `de` / `dual_annealing`
- **`auto`**：先运行多起点局部优化；若指标仍未满足或改善低于阈值，再运行 DE，并用局部方法 polish
- **`multistart_n`** / **`multistart_method`** / **`multistart_seed`**：多起点数量、局部方法（`trf` / `lbfgs` / `lm` / `cg` / `adam`，默认 TRF）与随机种子
- **`multistart_sampler`**：`lhs`（默认）/ `sobol` / `extra_trees` / `optical_qw` / `optical_extra_trees`；混合采样器使用 optical-QW + Sobol 生成真实训练集和代理候选池，再由 Extra Trees 按 LCB 与多样性选点
- **`multistart_candidate_n`** / **`surrogate_pool_n`**：Extra Trees 的真实 loss 预筛选数量（默认 256）和代理候选池数量（默认 10000）
- **`surrogate_trees`** / **`surrogate_exploration_beta`** / **`surrogate_diversity_weight`**：树数量、LCB 探索强度和起点距离多样性权重
- **`surrogate_selection_gpu_min_work`**：Selection 的 `候选数 × 起点数 × 自由层数` 达到该阈值且 `use_cuda=true` 时使用 CuPy float64；默认 `10000000`
- **`multistart_final_polish_method`**：对多起点最优结果进行最终精修；推荐 `trf`
- **`gpu_ids`**：例如 `[0, 1, 2, 3]`；指定 CUDA 设备，要求 `use_cuda: true`；省略时自动使用所有可见 GPU（旧键 `multistart_gpu_ids` 仍兼容）
- **`multistart_progress_interval_s`**：Surrogate 各阶段及多 GPU 任务的进度心跳间隔，默认 `10` 秒
- **`auto_de_fallback`** / **`auto_min_relative_improvement`**：控制自动全局回退
- **`use_needle`**：允许改变层数；相关参数为 `max_layers`、`needle_candidate_mode`、`deep_search_candidates`、`needle_probe_nm`、`prune_threshold_nm`
- **`cg_initial_step_nm`** / **`cg_max_step_nm`** / **`cg_restart`**：CG 专用（默认分别跟 `adam_lr_nm`、`adam_max_step_nm`、自由层数）
- **`lbfgs_m`** / **`lbfgs_maxls`**：L-BFGS-B 历史向量数（默认 10）与线搜索最大步数（默认 20）
- **`R_target`**：写在每个 `bands[]` 上，反射率目标 ∈ [0, 1]；省略时由 `objective` 得到 1（maximize）或 0（minimize）
- **`nk_source`**：`library`（默认，按材料名从色散库读 n(λ),k(λ)）或 `fixed`（用文本膜系中的常数 n,k）；也可用 `use_fixed_nk: true`
- **`min_thickness_nm`**：单层最小厚度（nm，默认 `8`）；优化时抬高各材料厚度下界
- **`max_total_thickness_nm`**：所有膜层总厚度的硬上限（nm）；多起点会直接在总厚度可行域内采样，局部优化产生的超限候选则投影回可行域
- **`thickness_bounds_nm`**：按材料设置优化全过程的硬边界 `[下限, 上限]`（nm）；未列出的材料沿用内置范围
- **`multistart_sampling_bounds_nm`**：设置所有多起点候选的初值采样范围；必须位于对应的 `thickness_bounds_nm` 之内，未列出的材料沿用厚度硬边界
- **`mini_batch`**：`true` 或嵌套对象 `{"batch_size", "n_batches", "n_epochs", "shuffle_seed"}`；在 `method=adam` 或 `multistart_method=adam` 时生效
- **`checkpoint_on_best`**（默认 `true`）：运行中 best 变好时更新 `stack_best.txt`，并追加 `best_updates.csv`
- **`use_cuda`**：`true` 时走 CuPy 批量 TMM，并在多 GPU 上并行 Surrogate 真实 loss 预筛选和局部优化有限差分；Multistart 起点仍按每卡一个进程并行（仅 NVIDIA CUDA；macOS 不可用）

多 GPU 并行 Multistart：

```json
{
  "method": "multistart",
  "multistart_n": 8,
  "multistart_method": "trf",
  "use_cuda": true,
  "gpu_ids": [0, 1, 2, 3],
  "multistart_progress_interval_s": 10
}
```

上述配置启动 4 个进程，每块 GPU 依次处理约 2 个起点；最终结果仍由
主进程按统一 loss 选择。

Multistart + mini-batch Adam：

```json
{
  "method": "multistart",
  "multistart_n": 8,
  "multistart_method": "adam",
  "multistart_sampler": "extra_trees",
  "multistart_candidate_n": 256,
  "surrogate_trees": 200,
  "surrogate_exploration_beta": 1.0,
  "surrogate_pool_n": 10000,
  "surrogate_diversity_weight": 0.1,
  "surrogate_selection_gpu_min_work": 10000000,
  "multistart_final_polish_method": "trf",
  "adam_lr_nm": 2.0,
  "mini_batch": {
    "batch_size": 8,
    "n_batches": 10,
    "n_epochs": 40,
    "shuffle_seed": 0
  }
}
```

光学厚度 + Extra Trees 混合采样：

`optical_extra_trees` 使用 `d = q·λ/(4·n(λ)·cosθ)` 生成物理候选，
并按 `optical_sampler_fraction` 与 Sobol 混合。真实 TMM 预筛选集和
代理候选池均采用该混合比例，最后由 Extra Trees 选择局部优化起点。

```json
{
  "method": "multistart",
  "multistart_sampler": "optical_extra_trees",
  "multistart_n": 16,
  "optical_q_range": [0.7, 1.3],
  "optical_wavelength_nm": [800, 1800],
  "optical_pair_shared_wavelength": true,
  "optical_chirp": true,
  "optical_sampler_fraction": 0.7,
  "multistart_candidate_n": 512,
  "surrogate_trees": 300,
  "surrogate_pool_n": 20000,
  "surrogate_exploration_beta": 1.0,
  "surrogate_diversity_weight": 0.15,
  "max_total_thickness_nm": 1800,
  "multistart_final_polish_method": "trf"
}
```

## 指定层数并自动生成膜系

`multistart_optimize.py` 根据层数生成交替 H/L 四分之一波长初始膜系，
然后使用配置中的 `method=multistart|auto` 优化厚度：

```bash
sim/.venv/bin/python sim/multistart_optimize.py \
  --layers 8 \
  --config sim/examples/example_multistart_optimize.json
```

可用 `--mode hlh|lhl` 覆盖首层材料顺序，或用 `--output-dir` 覆盖输出目录。
`--method auto` 可覆盖配置并启用“多起点局部优化 → 必要时 DE 回退”：

```bash
sim/.venv/bin/python sim/multistart_optimize.py \
  --layers 8 \
  --config sim/examples/example_multistart_optimize.json \
  --method auto
```

配置中的 `stack_init.design_wavelength_nm` 决定初始四分之一波长厚度；
也可直接设置 `high_thickness_nm`、`low_thickness_nm`。生成的 stack 和最终生效配置保存在输出目录的 `_inputs/` 中。

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
- `scipy`（默认 TRF、L-BFGS-B、DE 和双退火需要）
- `cupy`（可选；`use_cuda: true`，仅 NVIDIA CUDA）

既有基线评估仍可无额外依赖：

```bash
python3 sim/design.py
```

## 与文档的关系

项目级说明见仓库根目录 [README.md](../README.md) 与 `docs/`。算法设计详见 [光学薄膜层数与厚度联合优化算法设计](../docs/film-optimization-algorithm.md)（含合成/精修分类、算法选型、可选路径 A–I 与配置示例）。本程序侧重**可脚本化的膜系厚度/层数自动设计**；OghmaNano 等 GUI 仿真实操见 `docs/oghmanano-ir-film-simulation.md`。
