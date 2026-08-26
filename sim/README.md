# 红外反射膜结构优化程序

根据入射角、初始膜系、波段反射/透射指标，用 **TMM（传输矩阵法）** 计算光谱，以 **阻尼最小二乘（Levenberg–Marquardt）** 优化各层厚度，并用 **逐层增加法** 自动增减膜层；在满足指标的前提下尽量减小总厚度。

## 快速开始

```bash
cd /path/to/IR-reflective-film
python3 -m venv sim/.venv
sim/.venv/bin/pip install -r sim/requirements.txt
sim/.venv/bin/python sim/optimize_film.py sim/examples/example_vis_pass_ir_reflect.json
```

结果默认写到 `sim/out/optimize_example/`：

| 文件 | 内容 |
|------|------|
| `rt_and_n_before_after.png` | 优化前后反射率/透射率，以及材料折射率 |
| `spectrum_before_after.csv` | 同上光谱数值 |
| `stack_optimised.json` | 优化后各层材料与厚度 |

不传参数时，默认使用上述示例输入文件。

## 功能概览

1. **输入驱动**：入射角、初始结构（或啁啾种子）、波段 R/T 上下限、优化选项  
2. **材料库**：从色散模型取复折射率 \(n+ik\)（吸收系数 \(\alpha=4\pi k/\lambda\)）  
3. **目标**：各波段反射率/透射率约束 + 总厚度最小化  
4. **光学引擎**：内置 TMM；可切换外部程序（`rt_engine=external`）  
5. **优化**：阻尼最小二乘；前端粗坐标下降；上层逐层增加（先满足长波反射带，再全指标精修）  
6. **默认材料**：air、SiO2、TiO2、glass（另有 Ag、ITO、PET）  
7. **绘图**：matplotlib 输出优化前后 R/T 与折射率曲线  

## 目录结构

```
sim/
  optimize_film.py      # 入口：读 JSON、优化、出图
  rt_calculator.py      # TMM / 外部 R/T 接口
  lm_optimizer.py       # 阻尼最小二乘 + 粗坐标下降
  needle.py             # 逐层增加法
  tmm.py                # 传输矩阵核心
  dispersion.py         # 材料色散
  design.py             # 既有 OMO/OMOMO 评估脚本（无优化）
  requirements.txt
  examples/
    example_vis_pass_ir_reflect.json   # 示例输入
    external_rt_stub.py                # 外部引擎桩
  out/optimize_example/                # 示例输出（运行后生成）
```

## 输入文件格式（JSON）

```json
{
  "incident_angle_deg": 0,
  "polarization": "unpolarized",
  "incident_medium": "air",
  "substrate": "glass",
  "substrate_thickness_m": 0.0007,
  "exit_medium": "air",
  "substrate_model": "semi_infinite",
  "rt_engine": "tmm",
  "seed": {
    "type": "chirped_qw",
    "centres_nm": [850, 1100, 1400, 1650],
    "periods_per_centre": 3,
    "cell": ["tio2", "sio2"]
  },
  "layers": [
    {"material": "tio2", "thickness_nm": 100},
    {"material": "sio2", "thickness_nm": 150}
  ],
  "bands": [
    {
      "wavelength_nm": [420, 700],
      "T_min": 0.94,
      "R_max": 0.02,
      "R_target": 0.0,
      "T_target": 0.97,
      "weight": 2.0
    },
    {
      "wavelength_nm": [780, 1800],
      "R_min": 0.50,
      "R_target": 0.70,
      "weight": 1.0
    }
  ],
  "optimizer": {
    "method": "lm",
    "use_needle": true,
    "thickness_weight": 0.01,
    "wavelength_step_nm": 20,
    "max_layers": 40,
    "max_add_rounds": 8,
    "max_iter": 40,
    "high_index": "tio2",
    "low_index": "sio2"
  },
  "plot_wavelength_nm": [400, 1800],
  "plot_step_nm": 5,
  "output_dir": "../out/optimize_example"
}
```

要点：

- **`optimizer.method`**：局部 `lm` / `adam`；全局（需 scipy）`de`（differential evolution）或 `dual_annealing`。全局可调 `de_popsize`、`global_seed`、`global_polish` 等。  
- **`global_polish_method`**：全局搜索后再局部精修，`none` / `lm` / `adam`（兼容旧字段 `global_polish_lm: true` → `lm`）。开启后会同时写出 `stack_global.json`（DE/DA）与 `stack_polished.json`（精修后；`stack_optimised.json` 同精修结果）。  
- **`checkpoint_on_best`**（默认 `true`）：运行中 best 变好时更新 `stack_best.json`，并追加 `best_updates.csv`。全局 DE/DA 仍按每次 NEW best 写入；局部 LM/Adam 按 `checkpoint_local_every` 节流（默认全网格每 5 iter、mini-batch 每 1 epoch），结束时若有未写出的 best 会补写。  
- **`layers` 与 `seed`**：有 `layers` 则用给定膜系；否则可用 `seed` 生成啁啾 1/4 波长堆；都没有则用简单单周期种子。  
- **`bands`**：每段可设 `R_min` / `R_max` / `T_min` / `T_max`，以及可选的连续目标 `R_target` / `T_target`。  
- **`rt_engine`**：`tmm`（默认 CPU）、`tmm_cuda`（CuPy 波长批量 TMM，需 NVIDIA GPU）、或 `external`。也可用顶层 `"use_cuda": true`。  
- **`use_cuda`**：`true` 时走 GPU 批量相干 TMM（仅 `semi_infinite`；`incoherent_slab` 回退 CPU）。macOS 无 CUDA，请保持 `false`。  
- **`substrate_model`**：`semi_infinite`（镀膜设计默认，忽略厚基底背面）或 `incoherent_slab`（含非相干基底双面）。含背面时，可见光 R≤2% 往往物理上很难达到。  

## 示例指标

`example_vis_pass_ir_reflect.json`：

| 波段 | 指标 |
|------|------|
| 420–700 nm | 透射率 ≥ 94%，反射率 ≤ 2% |
| 780–1800 nm | 反射率 ≥ 50% |

全介质啁啾堆在可见区常有高阶反射带，严格同时满足上述不等式较难。程序会在保持红外反射的前提下尽量改善可见光，并在终端与 `stack_optimised.json` 中标明是否达标。

## 外部 R/T 接口

输入 JSON 中：

```json
"rt_engine": "external",
"external_command": "sim/.venv/bin/python sim/examples/external_rt_stub.py --input {input} --output {output}"
```

外部程序应：

1. 读取 `{input}`（JSON：波长、角度、膜层等）  
2. 写出 `{output}` CSV：`wavelength_m,R,T`（可与输入波长一一对应）  

仓库内 `external_rt_stub.py` 用内置 TMM 演示该协议，可换成自有求解器。

## 优化流程简述

1. **相位 1（逐层增加）**：仅针对带 `R_min` 的波段（通常为红外反射带）；若不达标则交替追加高/低折射率 1/4 波长对并 LM 精修。  
2. **相位 2**：恢复全部波段约束；先做粗坐标下降，再 LM；全部达标后加大厚度权重以压薄。  

残差策略：不等式未满足时只推可行性；满足后加入连续目标与厚度项，且不等式项始终保留以免压垮已达标波段。

## 依赖

- Python 3.10+（推荐）  
- `matplotlib`（见 `requirements.txt`）  
- `scipy`（可选；`method=de` / `dual_annealing` 全局厚度搜索需要）  
- `cupy`（可选；`use_cuda` / `rt_engine=tmm_cuda`，仅 NVIDIA CUDA；macOS 不可用）  
- 标准库即可跑 TMM；优化入口需 matplotlib 出图  

既有基线评估仍可无额外依赖：

```bash
python3 sim/design.py
```

## 与文档的关系

项目级说明见仓库根目录 [README.md](../README.md) 与 `docs/`。本程序侧重**可脚本化的膜系厚度/层数自动设计**；OghmaNano 等 GUI 仿真实操见 `docs/oghmanano-ir-film-simulation.md`。
