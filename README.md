# IR-reflective-film

手机屏幕红外（IR, Infrared）反射膜的技术调研与仿真指南。

## 文档

- [IR 反射膜制备方案、结构与仿真指南](docs/IR-reflective-film-overview.md) — 涵盖 DMD、全介质膜堆、聚合物 MOF 等常见制备路线，以及 TMM / RCWA / FDTD 仿真流程；说明 TFCalc 基于 TMM、与全波求解的区别及是否需要全波。文中缩略词均给出中英文全称，并附分类对照表。
- [OghmaNano 红外反射膜仿真实操](docs/oghmanano-ir-film-simulation.md) — 以 400–700 nm 透射、700–1300 nm 反射为目标，给出 OMOMO 起始膜系、软件设置、材料导入、参数扫描和可复现的角度谱结果。

## 可复现仿真

```bash
python3 sim/design.py
```

脚本仅使用 Python 标准库，输出 OMO / OMOMO / 全介质基线的 R/T 光谱及 OghmaNano 示例材料文件。

## 薄膜结构自动优化

详见 **[sim/README.md](sim/README.md)**（输入格式、模块说明、外部 R/T 接口与示例）。

```bash
python3 -m venv sim/.venv
sim/.venv/bin/pip install -r sim/requirements.txt
sim/.venv/bin/python sim/optimize_film.py sim/examples/example_vis_pass_ir_reflect.json
```

示例目标：420–700 nm 透射率 ≥94%、反射率 ≤2%；780–1800 nm 反射率 ≥50%。结果写入 `sim/out/optimize_example/`。

## 主要内容

| 主题 | 说明 |
|------|------|
| 制备方案 | DMD（Dielectric/Metal/Dielectric，电介质/金属/电介质，如 ITO/Ag/ITO）、全介质干涉堆、聚合物 MOF（Multilayer Optical Film，多层光学膜）、吸收+反射复合 |
| 层结构 | 各方案的典型材料、厚度与工艺流程 |
| 光学仿真 | TMM（Transfer Matrix Method，传输矩阵法，首选；TFCalc / Essential Macleod 等）、RCWA（Rigorous Coupled-Wave Analysis，严格耦合波分析）、FDTD（Finite-Difference Time-Domain，时域有限差分）的方法选型、TMM 与全波对比，以及完整工作流 |
| 缩略词 | 波段、结构、材料、工艺、仿真与产品名称的中英文对照，见文档第 8 节 |
