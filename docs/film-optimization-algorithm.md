# 光学薄膜层数与厚度联合优化算法设计

本文档描述如何同时优化**膜层数量**与**各层厚度**，使反射率 $R(\lambda)$ 与透射率 $T(\lambda)$ 满足多波段目标。算法面向本仓库 `sim/` 下的 TMM 仿真与优化模块，并与现有代码对应。

内容涵盖：问题定义、合成/精修双轴分类、Needle 与全局元启发式算法的关系、文献依据下的选型、多条可选优化路径，以及实现映射与调参建议。

**相关代码**

| 模块 | 职责 |
|------|------|
| `sim/tmm.py` | 传输矩阵法（TMM）核心 |
| `sim/rt_calculator.py` | TMM / 外部引擎统一接口 |
| `sim/lm_optimizer.py` | 厚度精修（TRF / LM / Adam / L-BFGS-B / DE / 双退火） |
| `sim/needle.py` | 灵敏度 Needle、Deep Search 候选与剪枝 |
| `sim/optimize_film.py` | 文本膜系统一入口（固定层数或 `use_needle` 合成） |
| `sim/plot_rt.py` / `sim/plot_rt_txt.py` | 光谱计算与绘图 |

**配置示例**

| 场景 | 示例文件 |
|------|----------|
| 固定层数 LM/Adam 精修 | `sim/examples/example_optimize_film.json` |
| 固定层数 DE 全局精修 | `sim/examples/example_optimize_de.json` |
| Needle 分阶段合成 | `sim/examples/example_vis_pass_ir_reflect.json` |
| Adam mini-batch | `sim/examples/example_optimize_film_minibatch.json` |

---

## 1. 问题定义

### 1.1 设计变量

给定入射角 $\theta$、偏振态、基底与可选材料库 $\mathcal{M}$（如 TiO$_2$ / SiO$_2$），设计变量为：

$$
\text{设计} = \bigl( N,\; (m_1,d_1),\ldots,(m_N,d_N) \bigr)
$$

- $N$：镀膜层数（**离散**，可变）
- $m_i \in \mathcal{M}$：第 $i$ 层材料（**离散**）
- $d_i \in [d_{\min}(m_i),\, d_{\max}(m_i)]$：物理厚度（**连续**，单位 m）

入射介质与基底厚度通常固定，不参与优化。

### 1.2 前向模型

对每个波长 $\lambda$，用 TMM 计算：

$$
R(\lambda),\; T(\lambda) = \mathrm{TMM}\bigl(\theta,\, \text{stack},\, \lambda\bigr)
$$

材料折射率 $N(\lambda)=n+\mathrm{i}k$ 可来自色散库（`dispersion.py`）或文本栈中的常数 $n,k$。

### 1.3 波段目标

每个波段 $b$ 在 $[\lambda_{b,\mathrm{lo}},\, \lambda_{b,\mathrm{hi}}]$ 上定义：

| 类型 | 含义 | 示例 |
|------|------|------|
| 不等式 | $R \ge R_{\min}$、$R \le R_{\max}$、$T \ge T_{\min}$、$T \le T_{\max}$ | 可见高透、红外高反 |
| 点目标 | $R \to R^{*}$、$T \to T^{*}$ | 连续拟合 |
| 极值 | 最大化 / 最小化 $R$ | `optimize_film.py` 的 `objective: maximize\|minimize` |

每波段可有权重 $w_b > 0$。

### 1.4 目标函数

综合损失（越小越好）：

$$
L = L_{\mathrm{spec}} + \lambda_d L_{\mathrm{thick}} + \lambda_s L_{\mathrm{smooth}} + \lambda_r L_{\mathrm{ripple}}
$$

**光谱项**（推荐波段归一化，与采样点数无关）：

$$
L_{\mathrm{spec}} = \frac{\sum_b w_b \cdot \mathrm{mean}_{\lambda \in b}\, \phi_b(R,T)}{\sum_b w_b}
$$

**厚度正则**（压薄）：

$$
L_{\mathrm{thick}} = \left(\frac{\sum_i d_i}{d_{\mathrm{ref}}}\right)^2
$$

**可选纹波抑制**（见 `optimize_film.py`）：

- $L_{\mathrm{smooth}}$：波段内相邻波长 $\Delta R$ 的均方
- $L_{\mathrm{ripple}}$：波段内 $(R_{\max}-R_{\min})^2$

### 1.5 约束

- 最大层数 $N \le N_{\max}$
- 最大总厚度 $\sum_i d_i \le D_{\max}$
- 可选：相邻层材料不得相同；仅允许 H/L 交替
- 工艺厚度上下界（`lm_optimizer.DEFAULT_BOUNDS`）

### 1.6 问题性质

这是**混合整数非线性规划（MINLP）**：

- 层数、材料序列 $\Rightarrow$ 离散，决策空间不连续
- 厚度 $\Rightarrow$ 连续，但随拓扑跳变
- $L$ 多局部极小，无全局最优保证

因此采用**合成 + 精修交织**的分层启发式，而非单一黑盒全局求解器。

---

## 2. 算法分类与总体架构

### 2.1 文献分类：合成与精修

光学镀膜数值方法按 Dobrowolski 传统分为两类（Tikhonravov 等，Appl. Opt. **51**, 7319, 2012 再次强调）：

| 类别 | 英文 | 职责 | 典型方法 |
|------|------|------|----------|
| **合成** | Synthesis | 不需好初值，构建层数与结构 | Needle、Gradual Evolution、Flip-flop、GA、SA |
| **精修** | Refinement | 固定拓扑，优化厚度 | LM、阻尼最小二乘、Adam；固定 $N$ 时的 DE / PSO / SA |

> **现代合成方法均将局部精修作为其组成部分（integral parts）**——每次结构变更后必须对厚度做 Refinement。

### 2.2 双轴定位模型

除「合成 $\leftrightarrow$ 精修」主层级外，按实现方式可再分一轴：

```text
轴 1（主层级）：合成 Synthesis  ←→  精修 Refinement
轴 2（实现方式）：领域专用（Needle / GE）  ←→  黑盒元启发式（GA / DE / PSO / SA）
```

**Needle 与 DE/GA/PSO 并非同一类算法**：

| 维度 | Needle | DE / GA / PSO / SA |
|------|--------|---------------------|
| 类型 | 领域专用**合成**算法 | 通用**元启发式** |
| 决策 | 插层位置、材料、层数 | 可编码 $(N,m_i,d_i)$ 或仅 $\mathbf{d}$ |
| 物理先验 | 扰动函数 $P(n,z)$ 指导增层 | 无，纯 merit 驱动 |
| 每步结构变更后 | **内嵌** Refinement | 精修层可内嵌，也可独立使用 |
| 文献定位 | 非局部合成（nonlocal synthesis） | 黑盒全局/局部搜索 |

Ma 等（arXiv:2409.17199, 2024）对多层薄膜逆设计的归纳：

| 方法 | 全局设计（层数+材料） | 典型表示 |
|------|:---:|:---:|
| Needle | ✓ | 组合式（combined） |
| GA | ✓ | 组合式 |
| PSO | ✗ | 向量化（仅厚度，层数固定） |

### 2.3 三层计算流水线

```text
输入（波段目标、材料库、初始种子）
        │
        ▼
┌───────────────────────────────────────────┐
│ 合成层：拓扑搜索                            │
│ Needle / Gradual Evolution / GA·SA / 束搜索 │
│ 决定 N、材料序列、增删位置                   │
└─────────────────┬─────────────────────────┘
                  │ 每次结构变更后
                  ▼
┌───────────────────────────────────────────┐
│ 精修层：厚度优化                            │
│ 多起点 TRF → DE / 双退火 → TRF polish       │
│ 决定 d_1,…,d_N（固定拓扑）                  │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 前向层：TMM 计算 R(λ), T(λ)                 │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
            达标？ ──否──► 回到合成层
                  │
                 是
                  ▼
┌───────────────────────────────────────────┐
│ 后处理：剪枝压薄（Design Cleaner）           │
└─────────────────┬─────────────────────────┘
                  ▼
            输出 stack + 光谱报告
```

**原则**：合成改结构，精修改厚度，TMM 算光谱；合成与精修**交织**而非二选一。

### 2.4 Needle 标准循环（文献与工业实践）

Tikhonravov 等（Appl. Opt. **35**, 5493, 1996）定义的标准 Needle 合成循环：

$$
\delta \mathrm{MF} = P(n_k, z)\,\delta z + o(\delta z)
$$

其中 $P(n,z)$ 为扰动函数，$z$ 为膜系截面位置。流程：

```text
repeat:
    计算 P(n, z)，在 P 最负处插入薄层（选材料 n_k）
    Refinement：对全部层厚度做局部优化（LM / 阻尼最小二乘）
until P 处处非负 或 新层厚度 < d_prune
```

OptiLayer、FSRStools 等均采用「插入 $\to$ Refinement」两步循环。Tikhonravov 1996 称其为**非局部合成技术**，并指出**无严格全局收敛证明**，但经验上通常可达接近全局最小的解。

Trubetskov（Appl. Opt. **59**, A75, 2020）进一步指出：**标准 Needle 每步为贪心策略**（只选当前 $P$ 最负的一处插入）。为增强全局性，同一团队提出 **Deep Search Needle**：对所有 $P$ 局部极小位置逐一「插入 + Refinement」，再选 MF 下降最大的一步——在 Needle 合成层内部叠加分支枚举，而非用 GA/DE 替代 Needle。

---

## 3. 残差构造（光谱目标 → 可优化量）

`lm_optimizer.build_residuals` 将光谱目标转为**固定长度**残差向量 $\mathbf{r}$，所有算法统一最小化

$$
\frac{1}{2}\,\lVert \mathbf{r} \rVert^2
$$

对每个波段 $b$、每个采样波长 $\lambda$，同时支持四个方向：

$$
\begin{aligned}
R < R_{\min} &\Rightarrow r \mathrel{+}= \sqrt{w}\,(R_{\min} - R) \\
R > R_{\max} &\Rightarrow r \mathrel{+}= \sqrt{w}\,(R - R_{\max}) \\
T < T_{\min} &\Rightarrow r \mathrel{+}= \sqrt{w}\,(T_{\min} - T) \\
T > T_{\max} &\Rightarrow r \mathrel{+}= \sqrt{w}\,(T - T_{\max})
\end{aligned}
$$

存在连续目标时始终加入：

$$
\begin{aligned}
R_{\mathrm{target}} \text{ 已设} &\Rightarrow r \mathrel{+}= \sqrt{w'}\,(R - R_{\mathrm{target}}) \\
T_{\mathrm{target}} \text{ 已设} &\Rightarrow r \mathrel{+}= \sqrt{w'}\,(T - T_{\mathrm{target}})
\end{aligned}
$$

每项实际乘以 $\sqrt{2w_b/(W n_b)}$，其中 $W=\sum_b w_b$、$n_b$ 为波段采样数，使 $L$ 等于波段归一化均方误差。厚度残差始终存在而非仅在光谱可行时出现，因此可行边界两侧的残差维度不变。

**设计要点**

1. **语义完整**：`R_min/R_max/T_min/T_max` 均进入优化，而不只是进入最终检查。
2. **波段归一化**：每波段先取均值再加权（`optimize_film.reflectance_mse`），使 $w_b$ 与波段内采样点数无关。
3. **同一目标**：TRF、LM、Adam、CG、L-BFGS-B、DE 与报告值均使用 $L=\frac12\|\mathbf r\|^2$。
4. **厚度项**：$\sqrt{2\lambda_d}\cdot\sum_i d_i/d_{\mathrm{ref}}$ 作为额外残差分量。

**极值目标映射**（文本栈优化器）：

| `objective` | 目标 $t_b$ | 等价 BandSpec |
|-------------|-------------|---------------|
| `maximize` | $R \to 1$ | `R_min=0.9, R_target=1` |
| `minimize` | $R \to 0$ | `R_max=0.1, R_target=0` |

---

## 4. 精修层：厚度优化

固定拓扑 $(m_1,\ldots,m_N)$ 后，仅优化厚度向量 $\mathbf{d}=(d_1,\ldots,d_N)^{\mathsf T}$。精修层方法也常被 Needle 合成层**内嵌调用**。

### 4.1 粗坐标下降（coarse descent）

- 对自由层厚度做随机 / 坐标扰动（典型 $\pm 5$–$20\,\mathrm{nm}$）
- 仅接受 cost 下降，或满足 stop-band 约束的候选（`accept_fn`）
- 作用：跳出 LM 局部极小，为精修提供更好初值

### 4.2 TRF（默认局部精修）

默认采用 SciPy `least_squares(method="trf")`：

- 原生处理各材料膜厚上下界；
- 以 nm 作为数值变量，避免以 m 优化导致尺度病态；
- 利用残差雅可比和信赖域控制步长；
- 适合作为多起点局部优化及 DE 后 polish。

### 4.3 Levenberg–Marquardt

求解阻尼正规方程：

$$
(J^{\mathsf T} J + \lambda_{\mathrm{LM}}\, I)\,\Delta\mathbf{d} = -J^{\mathsf T} \mathbf{r}
$$

- $J_{ij} = \partial r_i / \partial d_j$：残差对厚度的雅可比（前向有限差分，`fd_step_nm` 控制步长）
- 厚度投影到 $[d_{\min},\, d_{\max}]$
- 适合不等式 + 连续目标混合、残差维数适中的问题；Needle 每步插入后的 Refinement 首选

### 4.4 Adam + 波长 mini-batch

- 层数多、波长点多时，每步随机抽取波长子集
- 每 epoch 记录**全网格** cost，防止过拟合子集
- 配置见 `optimize_film.py` 的 `mini_batch` 对象；仅 `method=adam` 时生效

### 4.5 多起点与自动回退

- `method=multistart`：Latin-hypercube 生成多个厚度初值，每个初值运行 TRF（也可选 L-BFGS-B/LM/CG/Adam）；选择 Adam 时可启用波长 mini-batch。
- `method=auto`：先多起点局部优化；若指标仍未满足或相对改善不足，自动执行 DE，再用局部方法 polish。
- TMM 计算较便宜时，多起点通常比直接全空间运行大种群更节省预算。

### 4.6 全局厚度精修（DE / 双退火）

固定拓扑、多峰或初值差时：

- `method=de`：差分进化，在厚度盒约束 $[d_{\min}, d_{\max}]^N$ 内搜索
- `method=dual_annealing`：双退火，更强随机跳跃
- 再用 `global_polish_method`（推荐 `trf`，也可用 `lm` / `adam` / `lbfgs`）局部精修
- 输出 `stack_global.*`（全局阶段）与 `stack_polished.*`（精修后）

文献（IEEE IMOC 2009；IET Optoelectronics 2025）表明：在固定拓扑的连续厚度精修中，**DE 通常优于 GA**，且优于 PSO 处理有界多峰问题。

### 4.7 最优 checkpoint 选择

checkpoint 严格按统一 loss 选择，不加入相对初始膜厚或日志 $\Delta$，
也不使用膜厚变化打破平局。日志仍输出：

- `Δ`：当前 loss 减去本轮起始 loss；
- `thickness_delta_nm`：相对本轮起点的 RMS 膜厚变化。

两者只用于观察优化过程，不参与 loss 计算、多起点选择、全局/局部结果比较或 best checkpoint 更新。

---

## 5. 合成层：层数与拓扑

实现见 `sim/needle.py` 的 `NeedleSynthesizer`。适用于**可见高透 + 红外高反**等分阶段目标。与 GA/SA 在合成层**同级**。默认在所有界面分别插入 H/L 探针层，按有限差分 merit 灵敏度排序，对 top-k 候选执行完整局部精修；`candidate_mode=append` 保留旧的末端追加模式。

### 5.1 波段拆分

```text
all_bands
    ├── stop_bands   （含 R_min，如红外反射带）
    └── pass_bands   （含 R_max / T_min，如可见透过带）
```

### 5.2 Phase 1：停带合成（层数增长）

**目标**：先满足 `stop_bands`（硬约束波段）。

```text
layers ← 初始种子（啁啾 1/4 波长堆 或用户 layers）
optimizer.bands ← stop_bands

repeat round = 1 .. max_add_rounds:
    layers ← TRF_optimize(layers)         // 精修层
    if stop_ok(layers):
        break
    probes ← all (interface, H/L) needle variations
    layers ← best(top-k probes after TRF refinement)
    if len(layers) >= max_layers:
        break

layers ← best_stop_feasible(layers)
```

**设计波长** $\lambda_0$：在波段并集上取对数间隔中心（`n_design_centres` 个），使增层覆盖短波到长波。

### 5.3 Phase 2a：可见 AR 匹配

**目标**：在不破坏 stop 的前提下改善 pass 波段。

1. **全局厚度缩放**（$s \in [0.85,\,1.15]$）：将 Bragg 谐波移出可见区
2. **前端 AR 层插入**：在膜系入射侧加 $2$–`ar_layers` 层低折射率优先的匹配层
3. **前端受限优化**（`_refine_front`）：只优化前 `n_front` 层，IR 核心受 `_stop_ok` 约束
4. **迭代插入**：若仍未达标，继续在前端插入薄层并精修

### 5.4 Phase 2b：全波段联合精修

```text
optimizer.bands ← all_bands
layers ← coarse_descent(layers, accept_fn=stop_ok)
layers ← TRF_optimize(layers)
```

达标后逐步增大 $\lambda_d$（`thickness_weight`），在保持指标前提下压薄。

### 5.5 灵敏度 Needle 与剪枝

在相邻层界面 $i$ 处试探厚度 $\delta \to 0^{+}$ 的针状薄层：

1. 以 `needle_probe_nm` 在每个界面试探 H/L，有限差分计算 $\partial L / \partial \delta$
2. 若 $\partial L/\partial \delta < 0$ 且超阈值，在该位置插入新材料
3. 插入后调用精修层（§4）
4. **反向剪枝**：若 $d_i <$ `prune_threshold_nm`，删除并重优化；仅在指标仍满足且 merit 不恶化时接受

`deep_search_candidates` 控制进行完整精修的候选数量。当前实现为有限差分 Needle；伴随灵敏度和低灵敏度厚层剪枝仍是后续性能升级方向。

### 5.6 黑盒合成备选（无好初值时）

与 Needle **同级**的合成层替代方案（本仓库 ⬜ 未实现外层驱动，算法设计已预留）：

| 方法 | 编码 | 说明 | 文献 |
|------|------|------|------|
| 遗传算法 GA | $(N,\, m_{1..N},\, d_{1..N})$ 可变长 | 每代个体需内嵌精修；层数可固定不增 | Martin et al., Appl. Opt. 2008 |
| 模拟退火 SA | 随机增删层 + 扰动厚度 | 不需初值；层数 8–20 常见 | Boudet & Chaton, SPIE 1996 |
| 差分进化 DE | 同上（可变长） | 连续厚度子问题优于 GA | CPB 27, 106802, 2018 |
| 束搜索 | 逐层扩展，保留 top-$K$ | 计算量可控 | — |

**GA 相对 Needle 的独有场景**（Girshova et al., Computer Optics, 2022）：当工艺要求**不得增加界面数**时，GA 在固定层数下搜索厚度组合，而 Needle 会自动增层。

---

## 6. 算法选型与文献依据

### 6.1 按问题维度的适用性

| 优化子问题 | 推荐策略 | 本仓库配置 |
|-----------|---------------------|-----------------|
| 层数 + 材料 + 厚度（完整 MINLP） | Needle/GE $\gg$ SA $\gtrsim$ GA $\gtrsim$ DE（可变长）$\gg$ PSO | `use_needle: true` |
| 固定拓扑，厚度多峰 | 多起点局部搜索；失败后 DE + 局部 polish | `method=auto` |
| 固定拓扑，光滑 merit | 有界 TRF / L-BFGS-B | `trf` / `lbfgs` |
| 层数 $>12$ 或 $\lambda$ 点 $>200$ | Adam mini-batch +（可选）CUDA | `adam` + `mini_batch` |

### 6.2 膜层 + 厚度联合优化总排序

综合 Tikhonravov、Trubetskov、Ma et al. 及工业软件实践：

$$
\text{推荐度：}\;
\underbrace{\text{Needle}+\text{Refinement}}_{\text{合成+精修}}
\;>\;
\underbrace{\text{DE}+\text{TRF polish}}_{\text{精修层全局}}
\;>\;
\text{SA/DA}
\;>\;
\text{GA}
\;>\;
\text{PSO}
$$

说明：

1. **Needle + Refinement** 不是单一黑盒算法，而是文献与工业（OptiLayer、TFCalc）公认的联合优化首选。
2. **DE** 适合固定拓扑厚度精修及 Needle 内局部搜索失败时的救火，但不存在对所有目标和计算预算都成立的算法总排序。
3. **PSO** 在薄膜领域多为向量化厚度优化（Ma 2024：非全局设计），不宜作为完整 MINLP 的首选。

### 6.3 协同关系小结

| 关系 | 含义 | 示例 |
|------|------|------|
| **合成 $\to$ 精修** | 主层级；每次改结构后必精修厚度 | Needle 插入 $\to$ TRF |
| **合成 $\parallel$ 合成** | 同级替代 | Needle $\leftrightarrow$ GA/SA |
| **精修 $\subset$ 合成** | 元启发式作为 Needle 内嵌引擎 | Needle 步后 `method=de` polish |
| **精修 $\to$ 精修** | 全局精修后局部 polish | DE $\to$ `global_polish_method=trf` |
| **合成增强** | 同族扩展，非黑盒替代 | Deep Search Needle |

---

## 7. 可选设计与优化路径

以下路径按起点条件与失败恢复组织；配置示例见 §10 与 `sim/examples/`。

### 7.1 路径选择总览

```text
                        起点条件
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   有合理种子            层数/材料已定         完全无初值
   （啁啾/手动栈）       只调厚度              不知层数
        │                   │                   │
        ▼                   ▼                   ▼
   路径 A 工业标准      路径 B 快速精修      路径 E 盲搜合成
   路径 C 分阶段合成    路径 D 多峰救火       （规划）
        │                   │
        └─────────┬─────────┘
                  ▼
           LM 不收敛 / 多局部极小？
                  │
         是 → 路径 D 或 路径 F（DE 增强）
         否 → 路径 G（剪枝压薄，待实现）
```

### 7.2 路径 A：Needle 工业标准（默认首选）

| 项 | 内容 |
|----|------|
| **适用** | 多波段 R/T 目标；有啁啾种子；可见+红外联合设计 |
| **合成** | Needle Phase 1/2（`NeedleSynthesizer`） |
| **精修** | `method=trf` |
| **示例** | `sim/examples/example_vis_pass_ir_reflect.json` |
| **文献** | Tikhonravov et al., Appl. Opt. 35, 5493 (1996) |

### 7.3 路径 B：固定拓扑快速精修

| 项 | 内容 |
|----|------|
| **适用** | 实验栈、文献复现；层数与材料已定 |
| **合成** | — |
| **精修** | `method=trf` 或 `lbfgs` |
| **入口** | `optimize_film.py` + 文本 `stack.txt` |
| **示例** | `sim/examples/example_optimize_film.json` |

```bash
python3 sim/optimize_film.py \
  sim/examples/example_stack.txt \
  sim/examples/example_optimize_film.json
```

### 7.4 路径 C：分阶段波段合成（可见 AR + 红外高反）

| 项 | 内容 |
|----|------|
| **适用** | pass/stop 波段冲突；IR 反射膜核心场景 |
| **流程** | stop_bands 合成 $\to$ 全局缩放 $\to$ 前端 AR $\to$ 全波段精修 |
| **关键参数** | `ar_layers`、`front_free_extra`、`max_add_rounds` |
| **实现** | `needle.py` 内自动拆分 `stop_bands` / `pass_bands` |

### 7.5 路径 D：固定拓扑多峰救火（DE $\to$ LM）

| 项 | 内容 |
|----|------|
| **适用** | 拓扑固定；LM/Adam cost 平台化 |
| **精修** | `method=de`，`global_polish_method=trf` |
| **示例** | `sim/examples/example_optimize_de.json` |

```json
{
  "method": "de",
  "max_iter": 40,
  "de_popsize": 15,
  "global_polish": true,
  "global_polish_method": "trf"
}
```

### 7.6 路径 E：黑盒盲搜合成（规划）

| 项 | 内容 |
|----|------|
| **适用** | 无初值；Needle 不收敛 |
| **合成** | GA / SA 可变长编码（⬜ 待实现 `SynthesisDriver`） |
| **精修** | 每个体/每步内嵌 LM |
| **文献** | Boudet & Chaton, SPIE 1996；Martin et al., Appl. Opt. 2008 |

### 7.7 路径 F：Needle + DE 增强精修（混合）

| 项 | 内容 |
|----|------|
| **适用** | Needle 增层后 LM 反复陷入局部极 |
| **流程** | Needle 合成 $\to$ 若 cost 无改善：对当前拓扑 `method=de` $\to$ TRF polish $\to$ 继续 Needle |
| **现状** | `method=auto` 已实现多起点局部优化与 DE 自动 fallback |

### 7.8 路径 G：大规模性能（Adam mini-batch）

| 项 | 内容 |
|----|------|
| **适用** | $N_{\mathrm{layer}}>12$ 或 $N_{\lambda}>200$ |
| **精修** | `method=adam` + `mini_batch`；可选 `use_cuda` |
| **示例** | `sim/examples/example_optimize_film_minibatch.json` |
| **注意** | Needle 外层建议 `method=trf`；mini-batch 仅用于大规模精修阶段 |

### 7.9 路径 H：双退火精修

| 项 | 内容 |
|----|------|
| **适用** | 固定拓扑；DE 效果一般；merit 地形更崎岖 |
| **精修** | `method=dual_annealing`，`global_polish_method=trf` |
| **文献** | Chang et al., Opt. Lett. 15, 595 (1990) |

### 7.10 路径 I：后处理剪枝压薄（规划）

| 项 | 内容 |
|----|------|
| **适用** | 光谱达标但总厚度偏大 |
| **流程** | 删 $d_i<d_{\mathrm{prune}}$ 或低灵敏度层 $\to$ LM $\to$ 增大 $\lambda_d$ |
| **文献** | OptiLayer Design Cleaner；Tikhonravov 2007 |
| **现状** | 已实现薄层阈值剪枝；低灵敏度厚层剪枝仍待实现 |

### 7.11 失败恢复矩阵

| 症状 | 优先 | 次选 | 最后手段 |
|------|------|------|---------|
| 不知层数、有波段目标 | 路径 A / C | 路径 F | 路径 E |
| 层数已定、快速调厚度 | 路径 B | — | 路径 D |
| LM cost 平台化 | 路径 D | 路径 H | 路径 F |
| 层数多、波长点多 | 路径 G | 路径 B + CUDA | — |
| 可见达标、红外不达标 | 路径 C Phase 1 加长 | 换 `centres_nm` | 增大 `max_layers` |
| 红外达标、可见破坏 | 路径 C Phase 2a | `global_scale_search` | 增大 `ar_layers` |
| 达标但太厚 | 路径 I | 增大 `thickness_weight` | 手动删层 + 路径 B |
| 目标物理不可达 | 放宽 $R_{\min}$/$T_{\min}$ | 仅优化 stop_bands | 换材料库 |

### 7.12 本仓库 IR 反射膜推荐默认

```text
默认：路径 C（分阶段 Needle），method=trf，种子 chirped_qw
失败：method=auto（多起点 TRF → DE + TRF polish）
固定实验栈：路径 B（optimize_film.py，method=auto）
固定栈局部搜索卡住：路径 D（de + global_polish_method=trf）
```

---

## 8. 完整默认管线（伪代码）

```text
算法 FilmSynthesize(target_bands, materials, config):

  // ── 0. 初始化 ──
  layers ← resolve_initial_layers(config)     // layers 或 chirped seed
  (stop_bands, pass_bands) ← split_bands(target_bands)
  calc ← make_calculator(rt_engine, use_cuda)
  optimizer ← LMThicknessOptimizer(calc, bands, **opt_kwargs)

  // ── 1. 合成层 Phase 1：停带合成 ──
  optimizer.bands ← stop_bands
  if not stop_ok(layers):
      repeat until stop_ok or max_add_rounds:
          layers ← optimizer.optimize(layers)       // 精修
          if not stop_ok:
              layers ← append_qw_pair(layers, λ_design[i])  // 合成
  layers ← best_stop_stack(layers)

  // ── 2. 合成层 Phase 2a：可见 AR（若有 pass_bands）──
  optimizer.bands ← target_bands
  layers ← global_scale_search(layers, keep_stop=True)
  if pass_bands:
      for n_ar in 2 .. ar_layers:
          cand ← ar_front(n_ar) + layers
          cand ← refine_front(cand, freeze_core=True)
          keep best cand under stop constraint

  // ── 3. 精修层：全波段联合优化 ──
  layers ← coarse_descent(layers)
  result ← optimizer.optimize(layers)       // LM
  if multimodal or poor convergence:
      result ← DE_optimize(layers)          // 路径 D / F
      result ← LM_polish(result.layers)

  // ── 4. 后处理：剪枝 + 压薄（路径 I）──
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

## 9. 与现有代码的对应关系

| 算法步骤 | 实现位置 | 当前状态 |
|----------|----------|----------|
| TMM 前向 | `tmm.py`, `rt_calculator.py` | ✅ 已实现 |
| 残差 / cost | `lm_optimizer.build_residuals` | ✅ 已实现 |
| TRF / LM / Adam / CG / L-BFGS-B / DE / 双退火 | `lm_optimizer.LMThicknessOptimizer` | ✅ 已实现 |
| 粗搜索 | `coarse_descent` | ✅ 已实现 |
| 仅按 loss 的 checkpoint 选择 | `is_better_checkpoint` | ✅ 已实现 |
| Needle 合成层 | `needle.NeedleSynthesizer` | ✅ 已实现 |
| 固定层数 R 极值优化 | `optimize_film.py` | ✅ 当前入口 |
| Needle + 统一入口 | `optimize_film.py` + `needle.py` | ✅ `use_needle: true` |
| DE fallback 自动化 | `method=auto` | ✅ 已实现 |
| 薄层剪枝 / 有限差分灵敏度 Needle | `needle.py` | ✅ 已实现 |
| 外层 GA/SA 合成驱动 | — | ⬜ 待实现（路径 E） |
| 文本栈层数可变 | `optimize_film.py` | ✅ 已实现 |

---

## 10. 整合状态与后续路线

### 阶段 A：统一目标语义（已完成）

- `build_residuals` 完整支持 `R_min/R_max/T_min/T_max`，且可行边界两侧残差长度不变
- 所有优化器与报告统一使用 $\frac12\|\mathbf r\|^2$
- 文本入口的 `RObjectiveBand` 已映射为 `BandSpec`；统一 R/T 输入格式仍可继续扩展

### 阶段 B：接入 Needle 合成层（已完成）

- `optimize_film.py` 增加 `use_needle: true` 开关
- 层数由 `NeedleSynthesizer` 决定，厚度由 TRF 等局部方法决定
- 输出 `stack_best.txt` 在增层时同步更新层数

### 阶段 C：剪枝与灵敏度 Needle（基础版本已完成）

- Phase 2 结束后尝试删除 $d <$ `prune_threshold_nm` 的层并重优化
- 默认枚举界面与 H/L 材料探针，并精修 top-k 候选（§5.5）
- 后续可增加伴随梯度和低灵敏度厚层剪枝

### 阶段 D：性能与自动 fallback（已完成）

- `use_cuda: true` 启用 CuPy 波长批量 TMM
- Adam mini-batch 用于 $N_{\mathrm{layer}}>12$ 或 $N_{\lambda}>200$（路径 G）
- `method=auto` 在多起点 TRF 未达标或改善不足时自动切换 DE（路径 F）

---

## 11. 复杂度与调参建议

### 11.1 计算复杂度（量级）

| 操作 | 复杂度 | 备注 |
|------|--------|------|
| 单次 TMM 光谱 | $O(N_{\mathrm{layer}} \cdot N_{\lambda})$ | GPU 批量可加速 |
| LM 一步（有限差分） | $O(N_{\lambda} \cdot N_{\mathrm{free}}^{2})$ | $N_{\mathrm{free}}$ = 自由层数 |
| Needle 外层 | $O(K \cdot T_{\mathrm{LM}})$ | $K$ = 增层轮数，通常 $<20$ |
| DE 一代 | $O(\mathrm{popsize} \cdot N_{\lambda} \cdot N_{\mathrm{layer}})$ | 精修层；需 scipy |

### 11.2 关键参数

| 参数 | 典型值 | 作用 |
|------|--------|------|
| `wavelength_step_nm` | 10–20 | 优化采样密度；绘图用更密 `plot_step_nm` |
| `max_layers` | 20–40 | 层数上限 |
| `max_add_rounds` | 8–20 | Phase 1 增层轮数 |
| `thickness_weight` | $0 \to 0.01$–$0.05$ | 达标后逐步增大以压薄 |
| `fd_step_nm` | 0.5–2 | LM 有限差分步长 |
| `method` | `auto` | 多起点 TRF，未达标或改善不足时 DE 回退 |
| `multistart_n` | 4–12 | Latin-hypercube 初值数量 |
| `multistart_method` | `trf` | 每个初值采用的有界局部方法 |
| `multistart_sampling_bounds_nm` | 按设计设定 | 仅限制 Latin-hypercube 初值采样；局部优化仍使用厚度硬边界 |
| `auto_min_relative_improvement` | 0.01 | 触发 DE 回退的最小相对改善 |
| `min_thickness_nm` | 8（默认） | 单层最小厚度（nm）；抬高优化厚度下界 |
| `max_total_thickness_nm` | 按工艺设定 | 所有膜层总厚度硬上限；候选点超限时投影回可行域 |
| `nk_source` | `library`（默认） / `fixed` | `library`：按材料名从色散库读 n(λ),k(λ)；`fixed`：用文本膜系常数 n,k |
| `de_popsize` | 12–20 | DE 种群规模 |
| `global_polish_method` | `trf` | DE/退火后局部精修 |
| `deep_search_candidates` | 2–5 | Needle 每轮进行完整精修的候选数量 |
| `prune_threshold_nm` | 8–15 | 达标后尝试删除的薄层阈值 |

### 11.3 失败时的排查顺序

对应 §7.11 失败恢复矩阵：

1. 检查目标是否物理可达（全介质堆难以同时满足极严格可见 AR + 宽红外高反）
2. 换种子（不同 `centres_nm` 或手动 `layers`）
3. 增大 `max_layers` / `max_add_rounds`
4. 对固定层数尝试 `method=de`（路径 D）
5. 分阶段达标：先只优化 stop_bands，再放开 pass_bands（路径 C）
6. Needle 合成后 LM 平台化：DE polish（路径 F）

---

## 12. 输入输出约定

### 12.1 输入

**文本膜系**（`plot_rt_txt` 格式）+ **JSON 配置**（路径 B/D/H）：

```json
{
  "bands": [
    {"wavelength_nm": [420, 700], "objective": "minimize", "weight": 2.0},
    {"wavelength_nm": [780, 1800], "objective": "maximize", "weight": 1.0}
  ],
  "method": "auto",
  "multistart_method": "trf",
  "multistart_n": 8,
  "auto_de_fallback": true,
  "max_iter": 40,
  "use_needle": false,
  "wavelength_step_nm": 15,
  "plot_wavelength_nm": [400, 1800],
  "output_dir": "../out/optimize_example"
}
```

**JSON 膜系**（色散库，Needle 工作流，路径 A/C）：

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

**DE 全局精修**（路径 D，固定层数）：

```json
{
  "method": "de",
  "use_needle": false,
  "de_popsize": 15,
  "global_polish": true,
  "global_polish_method": "trf",
  "global_seed": 42
}
```

### 12.2 输出

| 文件 | 含义 |
|------|------|
| `stack_best.txt` / `stack_best.json` | cost + $\Delta$ 意义下的最优膜系 |
| `stack_final.*` | 最后一次迭代膜系 |
| `stack_global.*` / `stack_polished.*` | DE/退火全局阶段与精修后（路径 D） |
| `rt_best.png` / `rt_final.png` | 最优 / 最终光谱图 |
| `band_stats_*.csv` | 各波段 R/T 统计 |
| `loss_history.csv` | 迭代 / epoch 损失曲线 |
| `best_updates.csv` | 运行中 best 更新日志 |

---

## 13. 小结

光学薄膜 R/T 优化是**离散（层数、材料）+ 连续（厚度）**的 MINLP。文献与工业实践支持以下策略：

1. **主层级为合成 $\leftrightarrow$ 精修**，而非在 Needle、DE、GA、PSO 中做简单横向排序。
2. **Needle + Refinement** 是联合优化层数与厚度的首选；Needle 每步内嵌精修，与 DE/GA 在合成层同级、在精修层嵌套。
3. **精修层**：有界 TRF 为默认；多峰时用多起点 TRF，必要时 DE $\to$ TRF polish；大规模可用 Adam mini-batch。
4. **多条可选路径**（§7）按起点条件与失败症状选择；IR 反射膜默认走路径 C，固定栈走路径 B/D。
5. **后处理剪枝压薄**在达标后减小总厚度；Deep Search Needle 用于 $>50$ 层或相位/GDD 等复杂目标。

本仓库已具备合成层（Needle）与精修层（LM/Adam/DE）核心实现；下一步是统一入口（`use_needle`）、自动 DE fallback 与剪枝扩展。

---

## 参考文献与延伸阅读

### 合成与精修分类

- J. A. Dobrowolski, *Refinement of optical multilayer systems with different optimization procedures*, Appl. Opt. **29**, 2876 (1990).
- A. V. Tikhonravov, M. K. Trubetskov, *Modern design tools and a new paradigm in optical coating design*, Appl. Opt. **51**, 7319 (2012).

### Needle 与 Deep Search

- A. V. Tikhonravov, M. K. Trubetskov, G. W. DeBell, *Application of the needle optimization technique to the design of optical coatings*, Appl. Opt. **35**, 5493 (1996).
- A. V. Tikhonravov, M. K. Trubetskov, G. W. DeBell, *Optical coating design approaches based on the needle optimization technique*, Appl. Opt. **46**, 704 (2007).
- M. K. Trubetskov, *Deep search methods for multilayer coating design*, Appl. Opt. **59**, A75 (2020).
- [OptiLayer Needle Optimization](https://optilayer.com/automatic-design-options/needle-optimization/) — 工业软件文档：插入 $\to$ Refinement 循环。

### 全局元启发式与薄膜应用

- T. Ma, M. Ma, L. J. Guo, *Optical Multilayer Thin Film Structure Inverse Design: From Optimization to Deep Learning*, arXiv:2409.17199 (2024) — 合成/精修与 PSO/GA/Needle 对比表。
- Design of thin film filters using differential evolution, IEEE IMOC (2009) — DE vs PSO。
- E. I. Girshova et al., *Genetic algorithm for optimizing Bragg and hybrid metal-dielectric reflectors*, Computer Optics (2022) — GA 与 Needle 同级比较。
- T. Boudet, P. Chaton, *Thin film design using simulated annealing*, SPIE (1996) — 无初值合成。
- Asadollahzadeh et al., *Metaheuristic optimisation methods for GMR filters*, IET Optoelectronics (2025) — DE vs GA vs PSO 系统比较。

### 工业软件与本仓库

- TFCalc / Essential Macleod：TMM + Needle + 局部优化。
- W. H. Southwell, *Coating design using very thin high- and low-index layers*, Appl. Opt. **24**, 457 (1985).
- 本仓库 `docs/IR-reflective-film-overview.md` — 制备方案与仿真方法选型。
- 本仓库 `sim/README.md` — 脚本用法与配置说明。
