# 手机屏幕红外反射膜：制备方案、结构与仿真指南

> 文中缩略词在**首次出现**时给出中英文全称；完整分类对照见 [第 8 节「缩略词与术语说明」](#8-缩略词与术语说明)。

## 1. 背景与应用

手机屏幕上的红外（IR）反射膜，核心目标是**在可见光波段保持高透过率**，同时在**近红外（NIR，约 780–1400 nm）乃至更宽 IR 波段实现高反射或低透过**，以满足以下典型需求：

| 应用场景 | 功能需求 |
|---------|---------|
| 屏下指纹 / 3D 面容识别（NITS，Near-Infrared Transmission System，近红外透过系统） | 可见光高透，NIR 定向反射/透射，避免干扰前置摄像头 |
| 盖板玻璃 IR-cut（Infrared-cut，红外截止） | 抑制环境 IR 进入相机模组，减少鬼影与眩光 |
| 显示热管理（Solar loading） | 反射太阳 NIR，降低 OLED（Organic Light-Emitting Diode，有机发光二极管）/ LCD（Liquid Crystal Display，液晶显示器）温升 |
| 隐私 / 节能窗膜（柔性贴附） | 透可见、反 IR，可贴附曲面屏或背光模组 |

下文按**制备路线**分类，分别说明层结构、材料体系与工艺特点，并给出对应的**光学仿真方法与流程**。

---

## 2. 常见制备方案与层结构

### 2.1 电介质/金属/电介质（DMD，Dielectric/Metal/Dielectric）结构

**原理**：极薄金属层（Ag、Al、Au 等）本身在 IR 段有高反射，但在可见光区吸收/反射过强；上下各叠加一层透明导电氧化物（TCO，Transparent Conductive Oxide）或宽禁带半导体，通过**法布里–珀罗干涉 + 导纳匹配**实现可见光增透、IR 高反射。

**典型结构**（自下而上）：

```
[ 基板: 玻璃 / PET（Polyethylene Terephthalate，聚对苯二甲酸乙二醇酯） / 盖板 ]
    └── ITO（Indium Tin Oxide，氧化铟锡） (30–70 nm)
    └── Ag   (10–20 nm)      ← 核心 IR 反射层
    └── ITO (30–70 nm)
    └── [ 可选: 硬膜 / 抗指纹涂层 ]
```

**常见变体**：

| 结构 | 材料 | 特点 |
|------|------|------|
| OMO（Oxide/Metal/Oxide，氧化物/金属/氧化物） | ITO/Ag/ITO | 工业最成熟；550 nm 透过率可达 ~70–90% |
| OMO（FTO 系） | FTO（Fluorine-doped Tin Oxide，氟掺杂氧化锡）/Ag/FTO | FTO 成本低、热稳定性好 |
| OMO（Al 系） | ITO/Al/ITO | Al 成本低；可见/ IR 兼容性能均可 >80% |
| OMOMO（Oxide/Metal/Oxide/Metal/Oxide） | ITO/Ag/ITO/Ag/ITO | 多金属层叠加，IR 反射带宽更宽 |
| 种子层增强 | Ti/Ag/Ti 或 Cr 插层 | 改善 Ag 连续性与附着力 |

**制备工艺**：物理气相沉积（PVD，Physical Vapor Deposition）磁控溅射为主。

- ITO 层：RF（Radio Frequency，射频）溅射，室温 ~120 ℃，Ar 气氛，典型功率 50 W
- Ag 层：DC（Direct Current，直流）溅射，室温，厚度严格控制在 10–20 nm（<10 nm 不连续，>20 nm 可见光吸收增大）
- 亦可采用卷对卷（R2R，Roll-to-Roll）溅射实现大规模生产

**代表性能指标**（文献/专利典型值）：

- 可见光（550 nm）透过率：70–90%
- NIR（780–1400 nm）反射率：75–90%+
- 面电阻：3–10 Ω/□（方块电阻，若同时作导电层）

---

### 2.2 全介质干涉膜堆（Dielectric Stack）

**原理**：交替沉积高/低折射率无机介质薄膜，利用**薄膜干涉**在 IR 区形成高反射带（或截止带），在可见区保持高透。无金属层，**色偏可控、无电磁屏蔽**，但层数较多（通常 20–60 层）。

**典型结构**（以盖板玻璃 IR-cut / 低反射 + IR 阻隔为例）：

```
[ 玻璃 / 蓝宝石盖板 ]
    └── [H] Nb₂O₅ / TiO₂ / Ta₂O₅ / ZrO₂  (n ≈ 2.0–2.4, 5–200 nm)
    └── [L] SiO₂ / MgF₂ / Al₂O₃           (n ≈ 1.46–1.65, 75–220 nm)
    └── … 重复 N 周期 …
    └── [ 可选: 外层高硬度 DLC（Diamond-Like Carbon，类金刚石碳） / SiO₂ 防护 ]
```

**常见设计**：

- **IR-cut 滤波器**：TiO₂(n=2.30)/SiO₂(n=1.46) 交替，可见透过 >90%，IR OD（Optical Density，光密度，OD = −log₁₀(T)） 3–7
- **低反射 + IR 阻隔**：6 层及以上定制厚度比，可见反射 <6%，IR 阻隔 >50%
- **双面镀膜**：上表面阻 900–1100 nm IR，下表面阻 700–900 nm IR，拓宽截止带宽

**制备工艺**：

- 电子束蒸发（EBE，Electron Beam Evaporation）或磁控溅射
- 离子辅助沉积（IAD，Ion-Assisted Deposition）改善致密性
- 可在盖板玻璃单面或双面镀膜（1-side / 2-side coating）

**优势**：硬度高、耐刮擦、适合直接做盖板外层；可与 AR（Anti-Reflection，抗反射）膜联合设计。

**劣势**：制备周期长、大角度色偏需额外优化；对厚度控制精度要求高（±1–2 nm）。

---

### 2.3 聚合物多层光学膜（MOF，Multilayer Optical Film）

**原理**：通过**多层共挤（coextrusion）** 制备数百层交替的高低折射率聚合物薄膜，利用层间干涉实现光谱选择性反射。全聚合物、**质轻、柔性、无金属**，适合贴附在显示模组或偏光片之间。

**典型结构**：

```
[ PET 基膜 / 偏光片 / OCA（Optically Clear Adhesive，光学透明胶）胶层 ]
    └── Skin 层 (PET 保护, ~μm 级)
    └── 光学包层: 300–650+ 层交替
    │       ├── 高折射率层 A: PET / PMMA（Polymethyl Methacrylate，聚甲基丙烯酸甲酯） / co-PMMA  (n ≈ 1.6–1.7)
    │       └── 低折射率层 B: 含氟聚合物 / PETG（Polyethylene Terephthalate Glycol-modified，乙二醇改性 PET）      (n ≈ 1.3–1.5)
    └── Skin 层
    └── [ OCA 贴合至 LCD/OLED 模组 ]
```

**特殊单元胞设计**（抑制可见区高阶反射）：

| 单元胞类型 | 层序（厚度比） | 功能 |
|-----------|---------------|------|
| AB 双层层胞 | A–B | 基础 Bragg 反射 |
| 2A1B2C1B 四层胞 | 2:1:2:1 (A:B:C:B) | 三材料，nA>nB>nC，nB=√(nA·nC) |
| **711 六层胞** | 7A1B1A7B1A1B | 抑制可见区 2–4 阶反射，IR 850–1850 nm 反射 82–100%，可见透过 70–90% |

**制备工艺**：

1. 多流道共挤 feedblock 组装层流
2. 层倍增器（layer multiplier）将层数倍增至数百层
3. 流延铸片（cast）→ 骤冷
4. 线性拉延（draw ratio ~7:1）+ 热定型（heat set）
5. 可选：热成型为菲涅尔/锯齿结构，实现角度选择性 IR 反射

**代表产品**：3M UCSF（Ultra Clear Solar Film，超透明太阳能膜）、IDTMF（Industrial Display Thermal Management Film，工业显示热管理膜）、NITS 光学膜组等。

**优势**：大规模量产、柔性、可集成偏振控制；对 5G（第五代移动通信）/无线充电无屏蔽。

**劣势**：耐温/耐刮擦不如无机镀膜；需与 OCA 贴合，工艺窗口受模组制程约束。

---

### 2.4 吸收型 + 反射型复合 IR 滤波

**原理**：在玻璃或吸收型 IR 玻璃（如含特定离子的滤光片）两侧，分别镀反射型介质膜堆，**吸收 + 反射协同**，拓宽 IR 截止带宽。

**典型结构**：

```
[ 吸收型 IR 玻璃基板 ]
    ├── 第一侧: 反射膜堆 (透可见, 反射部分 IR)
    └── 第二侧: 反射膜堆 (反射剩余 IR)
```

常用于**相机模组 IR-cut 滤光片**，而非主显示屏，但在手机光学系统中与屏幕 IR 管理密切相关。

---

### 2.5 金属网栅 / 纳米线嵌入结构（新兴方案）

**原理**：在透明基底中嵌入 Ag 纳米线（AgNWs，Silver Nanowires）或金属网栅，兼顾导电与 IR 反射/吸收。

**典型结构**：

```
[ 基板 ]
    └── ITO
    └── Ag 纳米线网络层
    └── ITO
```

**制备**：溶液法涂布 AgNWs + 溅射 ITO 封装。适合柔性、低成本场景，但均匀性与可靠性仍在优化中。

---

## 3. 方案对比总览

| 方案 | 主要材料 | 典型层数 | 工艺 | 可见透过 | NIR 反射 | 柔性 | 量产成熟度 |
|------|---------|---------|------|---------|---------|------|-----------|
| DMD (OMO) | ITO/Ag/ITO | 3–5 | 磁控溅射 | 70–90% | 75–90% | ✓ (PET) | ★★★★★ |
| 全介质堆 | TiO₂/SiO₂ 等 | 20–60 | 溅射/蒸发 | 85–95% | 可调 | ✗ | ★★★★☆ |
| 聚合物 MOF | PET/含氟聚合物 | 300–650+ | 共挤拉延 | 70–90% | 82–100% | ✓ | ★★★★☆ |
| 吸收+反射复合 | IR 玻璃 + 介质膜 | 10–40 | 溅射 | 90%+ | 高 OD | ✗ | ★★★★☆ |
| AgNW 嵌入 | ITO/AgNW/ITO | 3+ | 涂布+溅射 | 70–85% | 中等 | ✓ | ★★☆☆☆ |

---

## 4. 光学仿真方法

### 4.1 方法选型

```
                    ┌─────────────────────────────────────┐
                    │         结构类型判定                  │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
     平面均匀多层膜            周期微结构               复杂 3D / 宽带
   (DMD / 全介质 / MOF)     (光栅 / 菲涅尔 MOF)        (非周期 / 散射)
              │                       │                       │
              ▼                       ▼                       ▼
        TMM / STACK              RCWA                   FDTD
     (首选, 最快)            (周期结构, 较快)          (通用, 最慢)
```

| 方法 | 全称 | 适用结构 | 速度 | 精度 | 典型工具 |
|------|------|---------|------|------|---------|
| **TMM** | 传输矩阵法 | 1D 平面多层，法向/斜入射 | 极快 | 高（解析） | TFCalc, Essential Macleod, OpenFilters, tmm, PyTMM |
| **STACK** | 堆栈解析求解 | 均匀平面多层 | 极快 | 高 | Lumerical STACK |
| **RCWA** | 严格耦合波分析 | 周期结构（光栅、光子晶体） | 快 | 高 | Lumerical RCWA, grcwa (Python) |
| **FDTD** | 时域有限差分 | 任意几何、宽带、非均匀 | 慢 | 高（数值） | Lumerical FDTD, Tidy3D, MEEP |
| **FEM** | 有限元法 | 任意 2D/3D 几何（频域） | 慢 | 高（数值） | COMSOL, JCMsuite |

**手机屏幕 IR 反射膜的主流仿真路径**：

- **DMD / 全介质镀膜** → **TMM**（工业标准，TFCalc、Essential Macleod 等）
- **聚合物 MOF（平面）** → **TMM**（各层视为均匀介质，n 取有效折射率）
- **结构化 MOF / 微棱镜** → **RCWA** 或 **FDTD**
- **斜入射、大角度响应** → TMM 斜入射模式 或 RCWA

使用 OghmaNano 建立 400–700 nm 透射、700–1300 nm 反射的实际工程步骤、OMOMO 起始厚度及可复现计算结果，见 [OghmaNano 红外反射膜仿真实操](oghmanano-ir-film-simulation.md)。

TMM 与 STACK 同属平面多层的解析/半解析求解；FDTD 与 FEM（Finite Element Method，有限元法）同属全波数值求解。RCWA 在周期假设下严格求解麦克斯韦方程，介于二者之间。

### 4.1.1 TFCalc 中的仿真方法

**TFCalc 不做全波求解。** 它是光学镀膜设计软件，核心是平面多层膜上的特征矩阵 / 传输矩阵（TMM），在「横向无限大、每层均匀、平面波入射」假设下解析计算膜系响应。

TFCalc 可计算与优化的量包括：反射率、透过率、吸收、光密度、损耗、相位、ψ、电场强度（EFI，Electric Field Intensity）、导纳、色度，以及超快光学量 GD（Group Delay，群延迟）、GDD（Group Delay Dispersion，群延迟色散）。薄膜层通常按相干叠加；基板背面反射可按非相干处理。入射可为单角度、锥角或用户定义角分布。渐变折射率 / rugate 膜用多层变折射率近似。设计侧提供局部优化（Variable Metric、Gradient、Simplex）、全局搜索与 Needle 优化。

这些能力都属于 TMM 镀膜设计，**不能**替代 FDTD / FEM 对横向不均匀、衍射或三维散射的建模。

### 4.1.2 TMM 与全波求解的区别、准确性与是否需要全波

**TMM**（含 TFCalc、Essential Macleod、STACK）把麦克斯韦方程在沿膜厚方向分层、横向均匀的结构上化为 2×2 传输矩阵（或特征矩阵）连乘，得到 R(λ)、T(λ)。斜入射时对 s/p 偏振分别计算，仍是一维问题。

**全波求解**（FDTD、频域 FEM）在 2D/3D 网格上直接离散麦克斯韦方程，可处理横向不均匀、衍射、散射与有限尺寸。RCWA 则对周期结构做傅里叶展开后严格求解。

| | TMM / TFCalc | 全波（FDTD / FEM） | RCWA |
|--|--|--|--|
| 几何 | 平面均匀多层 | 任意 2D/3D | 周期微结构 |
| 方程 | 1D 分层的解析/半解析解 | 离散麦克斯韦 | 周期麦克斯韦 + 傅里叶 |
| 速度 | 极快 | 慢 | 中等 |
| 侧向散射/衍射 | 不建模 | 能建模 | 能建模（周期） |
| 粗糙、岛状金属、纳米线 | 只能用等效 n、k | 可直接建模 | 视是否周期而定 |

对**理想平面多层膜**，TMM 就是该几何下麦克斯韦问题的精确解，不是工程简化。全波在网格足够密、色散模型一致时，应与 TMM **重合**。平面多层上偏差 **>1%** 时，优先检查材料 n(λ)、k(λ) 或网格精度，而不是先怀疑 TMM。

TMM 与实测的偏差通常来自**模型假设**（层厚、色散数据、界面互扩散、粗糙度、金属是否连续成膜、基板非相干处理），而不是矩阵算法本身。全波若未把真实三维形貌与材料数据建入，同样消除不了这些误差。

**多数 IR 反射膜设计不需要全波。** DMD、全介质堆、平面聚合物 MOF 用 TFCalc 等 TMM 工具即可，也是工业首选。斜入射、大角度同样用 TMM 的斜入射模式。需要全波或 RCWA 的情形包括：光栅、微棱镜、菲涅尔型结构化 MOF；Ag 纳米线、岛状/不连续金属、明显粗糙或散射；有限孔径、边缘、像素级 3D 结构；需要近场热点或非平面等离激元。对平面多层，全波主要用于与 TMM 交叉验证 R/T，不是设计主路径；厚度优化、公差与制程仍应在 TMM 工具中完成。

---

### 4.2 传输矩阵法（TMM）仿真流程

TMM 是设计 DMD 和介质膜堆的**首选方法**。其核心是对每一层建立 2×2 传输矩阵，连乘得到整个膜系的反射率 R(λ) 和透过率 T(λ)。

#### 4.2.1 仿真流程（以 TFCalc / Essential Macleod 为例）

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. 定义目标   │ →  │ 2. 建立膜系   │ →  │ 3. 材料库    │
│   光谱指标    │    │   初始结构    │    │   n(λ), k(λ) │
└──────────────┘    └──────────────┘    └──────────────┘
        │                                        │
        ▼                                        ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 6. 验证测试   │ ←  │ 5. 导出制程   │ ←  │ 4. 优化      │
│   vs 实测    │    │   参数        │    │   Refinement │
└──────────────┘    └──────────────┘    └──────────────┘
```

**Step 1 — 定义光学目标**

示例（OMOMO 结构）：

| 波段 | 波长范围 | 目标 |
|------|---------|------|
| UV（Ultraviolet，紫外） | 100–300 nm | T < 1% |
| 可见 | 550 nm | T ≥ 70% |
| NIR | 780–1400 nm | R ≥ 75% |

**Step 2 — 建立初始膜系**

- 设置基板（如 PET 125 μm 或 Corning 玻璃 0.7 mm）
- 添加初始层结构，如 ITO(30)/Ag(10)/ITO(70)/Ag(10)/ITO(30) nm
- 选择入射介质（空气 n=1）和出射半空间

**Step 3 — 配置材料色散**

- 从材料库导入 ITO、Ag、SiO₂、TiO₂ 等的 **n(λ) + k(λ)**（折射率实部与消光系数）复折射率
- 常用数据源：RefractiveIndex.info、厂商实测数据、Drude-Lorentz 模型拟合

**Step 4 — 运行优化（Refinement）**

- TFCalc / Macleod：设定各波段目标与权重后做局部或 Needle 优化（Macleod 中为 `Refinement → Target → Generate Target`）
- 优化变量：各层厚度（材料固定）或同时优化材料组合
- 约束：Ag 厚度 8–20 nm，ITO 单层 20–100 nm
- 输出：最优厚度组合及预测 R(λ)、T(λ) 曲线

**Step 5 — 导出制程参数**

- 输出各层目标厚度（nm 精度）
- 敏感性分析：厚度偏差 ±2 nm 对性能的影响
- 角度响应：0°–60° 斜入射扫描

**Step 6 — 实验验证闭环**

- 按优化参数溅射制样
- UV-Vis-NIR（紫外–可见–近红外）分光光度计测量 T(λ)、R(λ)
- SEM（Scanning Electron Microscope，扫描电子显微镜）/ TEM（Transmission Electron Microscope，透射电子显微镜）测量实际厚度，反馈修正仿真模型

#### 4.2.2 开源 TMM 代码示例（Python）

```python
import numpy as np
from tmm import coh_tmm  # pip install tmm

# 波长扫描 (nm)
wavelengths = np.linspace(300, 2000, 500)

# 材料复折射率 @ 550 nm 示例（实际应使用色散模型）
n_air = 1.0
n_ito = 1.9 + 0.01j
n_ag  = 0.055 + 3.5j
n_pet = 1.65

# 膜系: Air | ITO(40nm) | Ag(12nm) | ITO(40nm) | PET
d_ito, d_ag = 40e-9, 12e-9  # 厚度 (m)

T_list, R_list = [], []
for wl in wavelengths * 1e-9:
    # 简化: 此处应替换为 n(λ) 插值
    n_list = [1+0j, n_ito, n_ag, n_ito, n_pet]
    d_list = [np.inf, d_ito, d_ag, d_ito, np.inf]
    T_list.append(coh_tmm('s', n_list, d_list, 0, wl)['T'])
    R_list.append(coh_tmm('s', n_list, d_list, 0, wl)['R'])

# 绘制 T(λ), R(λ) 曲线进行设计评估
```

**常用开源/商业工具**：

| 工具 | 类型 | 说明 |
|------|------|------|
| TFCalc | 商业 | 平面多层 TMM 设计与优化（R/T/A、相位、EFI、Needle 等），不做全波 |
| Essential Macleod | 商业 | 工业薄膜设计标准，DMD/MOF 优化 |
| OpenFilters | 开源 | Python GUI（Graphical User Interface，图形用户界面），支持 TMM + 优化 |
| tmm / PyTMM | 开源库 | Python/MATLAB TMM 实现 |
| RefractiveIndex.info | 数据库 | 材料 n,k 色散数据 |

---

### 4.3 RCWA 仿真流程（周期/结构化 MOF）

适用于菲涅尔结构化 IR 反射膜、光栅耦合等。

**流程**：

1. **几何建模**：定义周期 Λ、层厚度、二维介电常数分布 ε(x,y,z)
2. **设置源**：平面波，TE/TM（Transverse Electric / Transverse Magnetic，横电 / 横磁）偏振，入射角 θ
3. **谐波截断**：设置 Fourier 阶数 N（通常 15–30）
4. **频率/波长扫描**：计算各阶衍射效率、总 R/T
5. **优化**（可选）：使用 grcwa + autograd 进行逆设计

**工具**：Lumerical RCWA、grcwa (Python)、pMaxwell-RCWA

---

### 4.4 FDTD 仿真流程（复杂结构验证）

当结构非均匀、非周期，或需要宽带脉冲响应时使用。

**流程**：

1. **3D 几何建模**：Multilayer boxes 或自定义结构
2. **材料定义**：频散模型（Drude, Lorentz, Sellmeier）
3. **源与边界**：平面波源 + PML（Perfectly Matched Layer，完美匹配层）吸收边界
4. **网格收敛测试**：逐步加密至 R/T 收敛
5. **监视器**：频域功率监视器提取 R(λ)、T(λ)
6. **后处理**：与 TMM 结果交叉验证

**工具**：Lumerical FDTD、Tidy3D（云加速）、MEEP（开源）

> **选型建议**：对平面多层，TMM（含 TFCalc）与 FDTD 结果应一致；若偏差 >1%，检查材料色散数据或网格精度。全波用于复杂三维验证，不替代 TMM 设计主路径。见 §4.1.2。

---

## 5. 完整设计–制备–验证工作流

```
  ┌─────────────────────────────────────────────────────────┐
  │                    需求分析                              │
  │  目标波段 / 截止深度 / 可见透过 / 角度范围 / 成本 / 柔性  │
  └────────────────────────┬────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │              方案选型 (DMD / 介质 / MOF)                  │
  └────────────────────────┬────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │         光学仿真 (TMM 为主)                               │
  │  · 材料库建立 → 初始结构 → 目标优化 → 厚度灵敏度分析       │
  └────────────────────────┬────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │              工艺可行性评估                               │
  │  · 溅射/共挤窗口 · 层间附着 · 热预算 · 量产一致性         │
  └────────────────────────┬────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │              样品制备                                     │
  │  · 小试片 → 光谱测试 (UV-Vis-NIR) → SEM 厚度确认         │
  └────────────────────────┬────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │              仿真–实测对比 & 迭代                         │
  │  · 修正 n(λ),k(λ) · 重新优化 · 可靠性/环境测试            │
  └────────────────────────┬────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │              模组集成 & 量产                              │
  │  · 贴合验证 · 显示画质评估 · 可靠性 (高温高湿/跌落)        │
  └─────────────────────────────────────────────────────────┘
```

---

## 6. 设计要点与常见问题

### 6.1 DMD 结构

- **Ag 厚度窗口极窄**（10–20 nm）：需精确膜厚监控（石英晶振 / 光学监控）
- **种子层**（Ti、Cr）改善 Ag 成核，但需控制吸收
- **氧化防护**：顶层 ITO 或 SiO₂ 钝化，防止 Ag 迁移与黄变

### 6.2 全介质膜堆

- **厚度监控精度** ±1 nm 级，否则可见区波纹（ripple）明显
- **大角度色偏**：需优化色散匹配或使用非四分之一波设计
- **硬膜外层**：SiO₂ 或 DLC 提升耐刮擦

### 6.3 聚合物 MOF

- **711 单元胞** 等高级设计可抑制可见区寄生反射
- **拉延比** 影响层厚与折射率，仿真需使用拉延后有效 n
- **贴合**：OCA 胶层折射率匹配（~1.47）需纳入仿真

### 6.4 仿真常见误差来源

| 误差来源 | 影响 | 对策 |
|---------|------|------|
| 材料 n(λ) 不准确 | 峰值偏移 10–50 nm | 实测椭偏仪标定 |
| 界面粗糙度 | 散射损失 | 引入有效介质层或经验损耗 |
| 厚度偏差 | 可见/ IR 性能急剧变化 | 灵敏度分析 + 制程窗口 |
| 忽略基板吸收 | IR 段 T 偏低 | 使用复折射率含 k 值 |

---

## 7. 参考文献与延伸阅读

1. ITO/Ag/ITO 多层结构设计与仿真 — *Journal of the Korean Solar Energy Society*, 2023
2. ITO/Al/ITO 红外-可见光兼容薄膜 — *人工晶体学报*, 2025
3. FTO/Ag/FTO 高透明红外隐身薄膜 — *物理学报*, 2023
4. 3M Multilayer Optical Film (MOF) 技术 — 3M Optical Solutions
5. 711 层单元胞 IR 反射聚合物膜 — US Patent 11,298,918
6. 盖板 IR-cut 抗反射镀膜 — US Patent 10,830,930
7. TMM 教程 — [OghmaNano Optical Filter Tutorial](https://www.oghma-nano.com/manual/tutorial-optical-filter.html)
8. 材料折射率数据库 — [RefractiveIndex.info](https://refractiveindex.info)
9. TFCalc 功能说明 — [Software Spectra / Hulinks TFCalc](https://www.sspectra.com/summary.html)

---

## 8. 缩略词与术语说明

下文按类别汇总本文出现的缩略词。化学元素符号（Ag、Al、Au、Ti、Cr 等）为周期表惯用写法，不单独展开。已在正文首次出现处给出全称的条目，此处再作集中对照，便于查阅。

### 8.1 波段与光学量

| 缩略词 | 英文全称 | 中文说明 |
|--------|---------|---------|
| IR | Infrared | 红外辐射，波长通常大于约 780 nm |
| NIR | Near Infrared | 近红外，本文约 780–1400 nm |
| UV | Ultraviolet | 紫外，波长通常小于约 400 nm |
| UV-Vis-NIR | Ultraviolet–Visible–Near Infrared | 紫外–可见–近红外光谱测量范围 |
| IR-cut | Infrared-cut | 红外截止滤波，抑制环境 IR 进入相机等模组 |
| OD | Optical Density | 光密度，描述截止深度；OD = −log₁₀(T)，数值越大阻隔越强 |
| n(λ), k(λ) | refractive index, extinction coefficient | 复折射率的实部（折射率）与虚部（消光系数）随波长的色散 |
| Ω/□ | ohms per square | 方块电阻（面电阻）单位 |
| 5G | 5th Generation | 第五代移动通信 |

### 8.2 结构与膜系

| 缩略词 | 英文全称 | 中文说明 |
|--------|---------|---------|
| DMD | Dielectric / Metal / Dielectric | 电介质/金属/电介质三明治结构，可见高透、IR 高反 |
| TCO | Transparent Conductive Oxide | 透明导电氧化物，常用作 DMD 的介质层 |
| OMO | Oxide / Metal / Oxide | 氧化物/金属/氧化物，DMD 的常见实现，如 ITO/Ag/ITO |
| OMOMO | Oxide / Metal / Oxide / Metal / Oxide | 双金属层变体，IR 反射带宽更宽 |
| MOF | Multilayer Optical Film | 聚合物多层光学膜，由数百层高低折射率聚合物交替构成 |
| AR | Anti-Reflection | 抗反射（减反射）镀膜 |

### 8.3 材料与器件

| 缩略词 | 英文全称 | 中文说明 |
|--------|---------|---------|
| ITO | Indium Tin Oxide | 氧化铟锡，最常用的透明导电氧化物 |
| FTO | Fluorine-doped Tin Oxide | 氟掺杂氧化锡，成本较低、热稳定性较好 |
| PET | Polyethylene Terephthalate | 聚对苯二甲酸乙二醇酯，常用柔性基板/基膜 |
| PETG | Polyethylene Terephthalate Glycol-modified | 乙二醇改性 PET，常用作低折射率聚合物层 |
| PMMA | Polymethyl Methacrylate | 聚甲基丙烯酸甲酯（有机玻璃） |
| co-PMMA | copolymer PMMA | 共聚改性 PMMA |
| OCA | Optically Clear Adhesive | 光学透明胶，用于把光学膜贴合到显示模组 |
| DLC | Diamond-Like Carbon | 类金刚石碳，高硬度防护层 |
| AgNW / AgNWs | Silver Nanowire(s) | 银纳米线 |
| OLED | Organic Light-Emitting Diode | 有机发光二极管显示 |
| LCD | Liquid Crystal Display | 液晶显示 |

### 8.4 工艺与表征

| 缩略词 | 英文全称 | 中文说明 |
|--------|---------|---------|
| PVD | Physical Vapor Deposition | 物理气相沉积 |
| RF | Radio Frequency | 射频；文中指磁控溅射电源类型 |
| DC | Direct Current | 直流；文中指磁控溅射电源类型 |
| R2R | Roll-to-Roll | 卷对卷连续镀膜或涂布 |
| EBE | Electron Beam Evaporation | 电子束蒸发 |
| IAD | Ion-Assisted Deposition | 离子辅助沉积，改善薄膜致密性 |
| SEM | Scanning Electron Microscope | 扫描电子显微镜，用于形貌与厚度测量 |
| TEM | Transmission Electron Microscope | 透射电子显微镜，用于纳米级厚度与界面观察 |

### 8.5 仿真方法

| 缩略词 | 英文全称 | 中文说明 |
|--------|---------|---------|
| TMM | Transfer Matrix Method | 传输矩阵法，平面多层膜光学计算的首选方法；TFCalc 等镀膜软件的核心算法 |
| STACK | — | Lumerical 等工具中的堆栈解析求解器，适用于均匀平面多层 |
| RCWA | Rigorous Coupled-Wave Analysis | 严格耦合波分析，适用于光栅、光子晶体等周期微结构 |
| FDTD | Finite-Difference Time-Domain | 时域有限差分，全波数值方法，适用于任意三维几何与宽带响应 |
| FEM | Finite Element Method | 有限元法，此处指频域全波求解 |
| EFI | Electric Field Intensity | 膜系内电场强度，TFCalc 等可计算 |
| GD | Group Delay | 群延迟 |
| GDD | Group Delay Dispersion | 群延迟色散 |
| TE / TM | Transverse Electric / Transverse Magnetic | 横电 / 横磁偏振 |
| PML | Perfectly Matched Layer | 完美匹配层，FDTD 中的吸收边界条件 |
| GUI | Graphical User Interface | 图形用户界面 |

### 8.6 产品与系统名称

| 缩略词 | 英文全称 | 中文说明 |
|--------|---------|---------|
| NITS | Near-Infrared Transmission System | 3M 近红外透过系统：一组光学膜，使 NIR 相机可透过 LCD 成像，用于屏下指纹或面容识别 |
| UCSF | Ultra Clear Solar Film | 3M 超透明太阳能膜：可见光高透、近红外高反的聚合物多层膜 |
| IDTMF | Industrial Display Thermal Management Film | 3M 工业显示热管理膜：非含金属多层膜，用于降低太阳热负载 |
| TFCalc | — | Software Spectra 的多层光学薄膜设计软件，基于 TMM，不做全波求解 |

---

*文档版本：v1.2 | 更新日期：2026-08-19*
