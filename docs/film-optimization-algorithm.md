# 光学薄膜层数与厚度联合优化算法设计

本文档描述如何同时优化**膜层数量**与**各层厚度**，使反射率 \(R(\lambda)\) 与透射率 \(T(\lambda)\) 满足多波段目标。算法面向本仓库 `sim/` 下的 TMM 仿真与优化模块，并与现有代码对应。

**相关代码**

| 模块 | 职责 |
|------|------|
| `sim/tmm.py` | 传输矩阵法（TMM）核心 |
| `sim/rt_calculator.py` | TMM / 外部引擎统一接口 |
| `sim/lm_optimizer.py` | 厚度优化（LM / Adam / DE / 模拟退火） |
| `sim/needle.py` | 逐层增加法（层数可变） |
| `sim/optimize_film.py` | 文本膜系入口（当前：固定层数厚度优化） |
| `sim/plot_rt.py` / `sim/plot_rt_txt.py` | 光谱计算与绘图 |

---

## 1. 问题定义

### 1.1 设计变量

给定入射角 \(\theta\)、偏振态、基底与可选材料库 \(\mathcal{M}\)（如 TiO₂ / SiO₂），设计变量为：

\[
\text{设计} = \bigl( N,\; (m_1,d_1),\ldots,(m_N,d_N) \bigr)
\]

- \(N\)：镀膜层数（**离散**）
- \(m_i \in \mathcal{M}\)：第 \(i\) 层材料（**离散**）
- \(d_i \in [d_{\min}(m_i),\, d_{\max}(m_i)]\)：物理厚度（**连续**，单位 m）

入射介质与基底厚度通常固定，不参与优化。

### 1.2 前向模型

对每个波长 \(\lambda\)，用 TMM 计算：

\[
R(\lambda),\; T(\lambda) = \mathrm{TMM}\bigl(\theta,\, \text{stack},\, \lambda\bigr)
\]

材料折射率 \(N(\lambda)=n+ik\) 可来自色散库（`dispersion.py`）或文本栈中的常数 \(n,k\)。

### 1.3 波段目标

每个波段 \(b\) 在 \([\lambda_{b,\mathrm{lo}}, \lambda_{b,\mathrm{hi}}]\) 上定义：

| 类型 | 含义 | 示例 |
|------|------|------|
| 不等式 | \(R \ge R_{\min}\)、\(R \le R_{\max}\)、\(T \ge T_{\min}\)、\(T \le T_{\max}\) | 可见高透、红外高反 |
| 点目标 | \(R \to R^\*\)、\(T \to T^\*\) | 连续拟合 |
| 极值 | 最大化 / 最小化 \(R\) | `optimize_film.py` 的 `objective: maximize\|minimize` |

每波段可有权重 \(w_b > 0\)。

### 1.4 目标函数

综合损失（越小越好）：

\[
L = L_{\mathrm{spec}} + \lambda_d L_{\mathrm{thick}} + \lambda_s L_{\mathrm{smooth}} + \lambda_r L_{\mathrm{ripple}}
\]

**光谱项**（推荐波段归一化，与采样点数无关）：

\[
L_{\mathrm{spec}} = \frac{\sum_b w_b \cdot \mathrm{mean}_{\lambda \in b}\, \phi_b(R,T)}{\sum_b w_b}
\]

**厚度正则**（压薄）：

\[
L_{\mathrm{thick}} = \left(\frac{\sum_i d_i}{d_{\mathrm{ref}}}\right)^2
\]

**可选纹波抑制**（见 `optimize_film.py`）：

- \(L_{\mathrm{smooth}}\)：波段内相邻波长 \(\Delta R\) 的均方
- \(L_{\mathrm{ripple}}\)：波段内 \((R_{\max}-R_{\min})^2\)

### 1.5 约束

- 最大层数 \(N \le N_{\max}\)
- 最大总厚度 \(\sum_i d_i \le D_{\max}\)
- 可选：相邻层材料不得相同；仅允许 H/L 交替
- 工艺厚度上下界（`lm_optimizer.DEFAULT_BOUNDS`）

### 1.6 问题性质

这是**混合整数非线性规划（MINLP）**：

- 层数、材料序列 → 离散，决策空间不连续
- 厚度 → 连续，但随拓扑跳变
- \(L\) 多局部极小，无全局最优保证

因此采用**分层 / 分阶段**启发式，而非单一黑盒求解器。

---

## 2. 总体架构

```
输入（波段目标、材料库、初始种子）
        │
        ▼
┌───────────────────┐
│ 外层：拓扑搜索     │  层数 N、材料序列、增删层
│ (Needle / 束搜索)  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 中层：厚度优化     │  粗搜索 → LM / Adam → DE polish
│ (lm_optimizer)    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 内层：光谱前向     │  TMM 计算 R(λ), T(λ)
│ (tmm / rt_calc)   │
└─────────┬─────────┘
          │
          ▼
    达标？ ──否──► 回到外层
          │
         是
          ▼
┌───────────────────┐
│ 后处理：剪枝压薄   │
└─────────┬─────────┘
          ▼
    输出 stack + 光谱报告
```

**原则**：外层改结构，中层改厚度，内层算光谱。

---

## 3. 残差构造（光谱目标 → 可优化量）

`lm_optimizer.build_residuals` 将光谱目标转为残差向量 \(\mathbf{r}\)，最小化 \(\frac{1}{2}\|\mathbf{r}\|^2\)。

对每个波段 \(b\)、每个采样波长 \(\lambda\)：

```
若 R < R_min  →  r += √w · (R_min - R)
若 R > R_max  →  r += √w · (R - R_max)
若 T < T_min  →  r += √w · (T_min - T)
若 T > T_max  →  r += √w · (T - T_max)
若已达标且有 R_target →  r += √w' · (R - R_target)
若已达标且有 T_target →  r += √w' · (T - T_target)
```

**设计要点**

1. **可行性优先**：未达标时主要惩罚违反量；达标后再加强连续目标，避免已满足波段被压垮。
2. **波段归一化**：每波段先取均值再加权（`optimize_film.reflectance_mse`），使 \(w_b\) 与波段内采样点数无关。
3. **厚度项**：\(\sqrt{2\lambda_d}\cdot\sum_i d_i/d_{\mathrm{ref}}\) 作为额外残差分量。

**极值目标映射**（文本栈优化器）：

| `objective` | 目标 \(t_b\) | 等价 BandSpec |
|-------------|-------------|---------------|
| `maximize` | \(R \to 1\) | `R_min=0.9, R_target=1` |
| `minimize` | \(R \to 0\) | `R_max=0.1, R_target=0` |

---

## 4. 中层：厚度优化

固定拓扑 \((m_1,\ldots,m_N)\) 后，仅优化厚度向量 \(\mathbf{d}\)。

### 4.1 粗坐标下降（coarse descent）

- 对自由层厚度做随机 / 坐标扰动（典型 ±5–20 nm）
- 仅接受 cost 下降，或满足 stop-band 约束的候选（`accept_fn`）
- 作用：跳出 LM 局部极小，为精修提供更好初值

### 4.2 Levenberg–Marquardt（默认局部法）

求解阻尼正规方程：

\[
(J^\top J + \lambda_{\mathrm{LM}} I)\,\Delta\mathbf{d} = -J^\top \mathbf{r}
\]

- \(J\)：残差对厚度的雅可比（前向有限差分，`fd_step_nm` 控制步长）
- 厚度投影到 \([d_{\min}, d_{\max}]\)
- 适合不等式 + 连续目标混合、残差维数适中的问题

### 4.3 Adam + 波长 mini-batch

- 层数多、波长点多时，每步随机抽取波长子集
- 每 epoch 记录**全网格** cost，防止过拟合子集
- 配置见 `optimize_film.py` 的 `mini_batch` 对象

### 4.4 全局厚度搜索（DE / 模拟退火）

- 多峰或初值差时，先用差分进化（`method=de`）或双退火（`dual_annealing`）在厚度空间全局搜索
- 再用 `global_polish_method`（`lm` / `adam`）局部精修
- 输出 `stack_global.*`（全局阶段）与 `stack_polished.*`（精修后）

### 4.5 最优 checkpoint 选择

不仅比较 cost，还比较相对初始厚度的 RMS 变化 \(\Delta\)（`checkpoint_score` / `is_better_checkpoint`）：

\[
\mathrm{score} = \mathrm{cost} + w_\Delta \cdot (\mathrm{rms\_nm} / 100)
\]

近并列时优先选厚度变化更小者，避免无意义的剧烈抖动。

---

## 5. 外层：层数与拓扑（Needle 逐层合成）

实现见 `sim/needle.py` 的 `NeedleSynthesizer`。适用于**可见高透 + 红外高反**等分阶段目标。

### 5.1 波段拆分

```
all_bands
    ├── stop_bands   （含 R_min，如红外反射带）
    └── pass_bands   （含 R_max / T_min，如可见透过带）
```

### 5.2 Phase 1：停带合成（层数增长）

**目标**：先满足 `stop_bands`（硬约束波段）。

```
layers ← 初始种子（啁啾 1/4 波长堆 或用户 layers）
optimizer.bands ← stop_bands

repeat round = 1 .. max_add_rounds:
    layers ← LM_optimize(layers)
    if stop_ok(layers):
        break
    λ0 ← design_wavelengths[round % n_centres]   // 对数间隔设计波长
    layers += QW_pair(H, L) at λ0                // 追加高/低折射率 1/4 波长对
    if len(layers) >= max_layers:
        break

layers ← best_stop_feasible(layers)
```

**设计波长** \(\lambda_0\)：在波段并集上取对数间隔中心（`n_design_centres` 个），使增层覆盖短波到长波。

### 5.3 Phase 2a：可见 AR 匹配

**目标**：在不破坏 stop 的前提下改善 pass 波段。

1. **全局厚度缩放**（0.85–1.15）：将 Bragg 谐波移出可见区
2. **前端 AR 层插入**：在膜系入射侧加 2–`ar_layers` 层低折射率优先的匹配层
3. **前端受限优化**（`_refine_front`）：只优化前 `n_front` 层，IR 核心受 `_stop_ok` 约束
4. **迭代插入**：若仍未达标，继续在前端插入薄层并精修

### 5.4 Phase 2b：全波段联合精修

```
optimizer.bands ← all_bands
layers ← coarse_descent(layers, accept_fn=stop_ok)
layers ← LM_optimize(layers)
```

达标后逐步增大 `thickness_weight`，在保持指标前提下压薄。

### 5.5 经典 Needle 法（扩展方向）

在相邻层界面 \(i\) 处试探厚度 \(\delta \to 0^+\) 的“针状”薄层：

1. 计算 \(\partial L / \partial \delta\)（伴随法或有限差分）
2. 若灵敏度为负且超阈值，在该位置插入新材料
3. 插入后调用中层厚度优化
4. **反向剪枝**：若某层 \(d_i < d_{\mathrm{prune}}\) 或灵敏度 \(\approx 0\)，删除该层并重优化

比盲目追加 H/L 对更精细，可作为 Phase 1 的升级路径。

### 5.6 全局拓扑搜索（无好初值时）

| 方法 | 编码 | 说明 |
|------|------|------|
| 遗传算法 / DE | \((N, m_{1..N}, d_{1..N})\) 可变长 | 每代评估需调用中层优化 |
| 模拟退火 | 随机增删层 + 扰动厚度 | 适合层数 8–20 |
| 束搜索 | 逐层扩展，保留 top-K | 计算量可控 |

---

## 6. 完整默认管线（伪代码）

```text
算法 FilmSynthesize(target_bands, materials, config):

  // ── 0. 初始化 ──
  layers ← resolve_initial_layers(config)     // layers 或 chirped seed
  (stop_bands, pass_bands) ← split_bands(target_bands)
  calc ← make_calculator(rt_engine, use_cuda)
  optimizer ← LMThicknessOptimizer(calc, bands, **opt_kwargs)

  // ── 1. 外层 Phase 1：停带合成 ──
  optimizer.bands ← stop_bands
  if not stop_ok(layers):
      repeat until stop_ok or max_add_rounds:
          layers ← optimizer.optimize(layers)
          if not stop_ok:
              layers ← append_qw_pair(layers, λ_design[i])
  layers ← best_stop_stack(layers)

  // ── 2. 外层 Phase 2a：可见 AR（若有 pass_bands）──
  optimizer.bands ← target_bands
  layers ← global_scale_search(layers, keep_stop=True)
  if pass_bands:
      for n_ar in 2 .. ar_layers:
          cand ← ar_front(n_ar) + layers
          cand ← refine_front(cand, freeze_core=True)
          keep best cand under stop constraint

  // ── 3. 中层：全波段联合优化 ──
  layers ← coarse_descent(layers)
  result ← optimizer.optimize(layers)       // LM
  if multimodal or poor convergence:
      result ← DE_optimize(layers)
      result ← LM_polish(result.layers)

  // ── 4. 后处理：剪枝 + 压薄 ──
  for each layer i (thin or low sensitivity):
      trial ← delete_layer(layers, i)
      trial ← optimizer.optimize(trial)
      if cost(trial) ≤ cost(layers) and specs_ok(trial):
          layers ← trial
  gradually increase thickness_weight
  layers ← optimizer.optimize(layers)

  // ── 5. 输出 ──
  write stack_best, stack_final, rt_*.png, band_stats_*.csv
  return layers, spectrum, specs_report
```

---

## 7. 与现有代码的对应关系

| 算法步骤 | 实现位置 | 当前状态 |
|----------|----------|----------|
| TMM 前向 | `tmm.py`, `rt_calculator.py` | ✅ 已实现 |
| 残差 / cost | `lm_optimizer.build_residuals` | ✅ 已实现 |
| LM / Adam / DE | `lm_optimizer.LMThicknessOptimizer` | ✅ 已实现 |
| 粗搜索 | `coarse_descent` | ✅ 已实现 |
| checkpoint 选择 | `checkpoint_score`, `is_better_checkpoint` | ✅ 已实现 |
| Needle 外层 | `needle.NeedleSynthesizer` | ✅ 已实现 |
| 固定层数 R 极值优化 | `optimize_film.py` | ✅ 当前入口 |
| Needle + 统一入口 | `optimize_film.py` + `needle.py` | ⬜ 待整合 |
| 剪枝 / 灵敏度 Needle | — | ⬜ 待实现 |
| 文本栈层数可变 | `optimize_film.py` | ⬜ 待扩展 |

---

## 8. 推荐整合路线

### 阶段 A：统一目标语义

- 将 `optimize_film.py` 的 `RObjectiveBand`（maximize/minimize）与 `lm_optimizer.BandSpec`（R_min/R_max/T_min）双向映射
- 文本栈与 JSON 膜系共用同一套 `bands` 配置格式

### 阶段 B：接入 Needle 外层

- `optimize_film.py` 增加 `use_needle: true` 开关
- 层数由 `NeedleSynthesizer` 决定，厚度由 `LMThicknessOptimizer` 决定
- 输出 `stack_best.txt` 在增层时同步更新层数

### 阶段 C：剪枝与灵敏度 Needle

- Phase 2 结束后删除 \(d < d_{\mathrm{prune}}\) 的层并重优化
- 可选：界面 Needle 灵敏度指导增层位置

### 阶段 D：性能

- `use_cuda: true` 启用 CuPy 波长批量 TMM
- Adam mini-batch 用于层数 > 12 或波长点 > 200 的情形

---

## 9. 复杂度与调参建议

### 9.1 计算复杂度（量级）

| 操作 | 复杂度 | 备注 |
|------|--------|------|
| 单次 TMM 光谱 | \(O(N_{\mathrm{layer}} \cdot N_\lambda)\) | GPU 批量可加速 |
| LM 一步（有限差分） | \(O(N_\lambda \cdot N_{\mathrm{free}}^2)\) | \(N_{\mathrm{free}}\) = 自由层数 |
| Needle 外层 | \(O(K \cdot T_{\mathrm{LM}})\) | \(K\) = 增层轮数，通常 < 20 |

### 9.2 关键参数

| 参数 | 典型值 | 作用 |
|------|--------|------|
| `wavelength_step_nm` | 10–20 | 优化采样密度；绘图用更密 `plot_step_nm` |
| `max_layers` | 20–40 | 层数上限 |
| `max_add_rounds` | 8–20 | Phase 1 增层轮数 |
| `thickness_weight` | 0 → 0.01–0.05 | 达标后逐步增大以压薄 |
| `fd_step_nm` | 0.5–2 | LM 有限差分步长 |
| `error_power` | 2 或 4 | >2 时加重离群点（纹波） |
| `checkpoint_delta_weight` | 0 | >0 时偏好厚度变化小的解 |

### 9.3 失败时的排查顺序

1. 检查目标是否物理可达（全介质堆难以同时满足极严格可见 AR + 宽红外高反）
2. 换种子（不同 `centres_nm` 或手动 `layers`）
3. 增大 `max_layers` / `max_add_rounds`
4. 对固定层数尝试 `method=de` 全局搜索
5. 分阶段达标：先只优化 stop_bands，再放开 pass_bands

---

## 10. 输入输出约定

### 10.1 输入

**文本膜系**（`plot_rt_txt` 格式）+ **JSON 配置**：

```json
{
  "bands": [
    {"wavelength_nm": [420, 700], "objective": "minimize", "weight": 2.0},
    {"wavelength_nm": [780, 1800], "objective": "maximize", "weight": 1.0}
  ],
  "method": "adam",
  "max_iter": 40,
  "use_needle": false,
  "wavelength_step_nm": 15,
  "plot_wavelength_nm": [400, 1800],
  "output_dir": "../out/optimize_example"
}
```

**JSON 膜系**（色散库，Needle 工作流）：

```json
{
  "seed": {"centres_nm": [900, 1200, 1500], "periods_per_centre": 3},
  "bands": [
    {"wavelength_nm": [420, 700], "T_min": 0.94, "R_max": 0.02, "weight": 2.0},
    {"wavelength_nm": [780, 1800], "R_min": 0.50, "weight": 1.0}
  ],
  "optimizer": {
    "method": "lm",
    "use_needle": true,
    "max_layers": 32,
    "max_add_rounds": 12
  }
}
```

### 10.2 输出

| 文件 | 含义 |
|------|------|
| `stack_best.txt` / `stack_best.json` | cost + Δ 意义下的最优膜系 |
| `stack_final.*` | 最后一次迭代膜系 |
| `rt_best.png` / `rt_final.png` | 最优 / 最终光谱图 |
| `band_stats_*.csv` | 各波段 R/T 统计 |
| `loss_history.csv` | 迭代 / epoch 损失曲线 |
| `best_updates.csv` | 运行中 best 更新日志 |

---

## 11. 小结

光学薄膜的 R/T 目标优化是**离散（层数、材料）+ 连续（厚度）**的混合问题。推荐策略：

1. **外层 Needle**：决定有多少层、材料序列、增删位置
2. **中层 LM / Adam / DE**：决定每层厚度
3. **内层 TMM**：计算光谱是否达标
4. **后处理剪枝压薄**：在达标前提下减小总厚度

本仓库已具备中层与外层核心实现；下一步是将 `needle.py` 重新接入 `optimize_film.py` 统一入口，并补充剪枝与灵敏度 Needle 扩展。

---

## 参考文献与延伸阅读

- TFCalc / Essential Macleod：工业薄膜设计软件，基于 TMM + Needle 优化
- W. H. Southwell, *Coating design using very thin high- and low-index layers*, Appl. Opt. (1983) — Needle 法经典文献
- 本仓库 `docs/IR-reflective-film-overview.md` — 制备方案与仿真方法选型
- 本仓库 `sim/README.md` — 脚本用法与配置说明
