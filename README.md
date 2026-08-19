# IR-reflective-film

手机屏幕红外（IR, Infrared）反射膜的技术调研与仿真指南。

## 文档

- [IR 反射膜制备方案、结构与仿真指南](docs/IR-reflective-film-overview.md) — 涵盖 DMD、全介质膜堆、聚合物 MOF 等常见制备路线，以及 TMM / RCWA / FDTD 仿真流程；说明 TFCalc 基于 TMM、与全波求解的区别及是否需要全波。文中缩略词均给出中英文全称，并附分类对照表。

## 主要内容

| 主题 | 说明 |
|------|------|
| 制备方案 | DMD（Dielectric/Metal/Dielectric，电介质/金属/电介质，如 ITO/Ag/ITO）、全介质干涉堆、聚合物 MOF（Multilayer Optical Film，多层光学膜）、吸收+反射复合 |
| 层结构 | 各方案的典型材料、厚度与工艺流程 |
| 光学仿真 | TMM（Transfer Matrix Method，传输矩阵法，首选；TFCalc / Essential Macleod 等）、RCWA（Rigorous Coupled-Wave Analysis，严格耦合波分析）、FDTD（Finite-Difference Time-Domain，时域有限差分）的方法选型、TMM 与全波对比，以及完整工作流 |
| 缩略词 | 波段、结构、材料、工艺、仿真与产品名称的中英文对照，见文档第 8 节 |
