# Material optical-constant sources (PVD-oriented)

Defaults target **physical vapour deposition** (magnetron sputter / e-beam /
thermal evaporation) thin films. Crystal or bulk references remain available
under explicit keys. Numeric tables are derived from the public-domain
[refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database)
(CC0) unless noted.

**Tabulated n,k (300–1800 nm):** see [`NK_TABLES_300_1800.md`](NK_TABLES_300_1800.md)
(defaults marked; n,k @ 550 nm highlighted per material).

## Default library keys

| Key | Material | Source | Notes / PVD relevance |
|-----|----------|--------|------------------------|
| `air` | Standard air | Ciddor 1996 (formula) | dry air 15 °C, 101.325 kPa, 450 ppm CO₂; k = 0 |
| `sio2` / `sio2_fused` | Fused silica (bulk) | Franta et al. 2016 | **default L**; n≈Malitson; n(550)≈1.460 |
| `sio2_pvd` | Sputtered SiO₂ | Lemarchand / Gao 2012–13 | magnetron film on BK7; n(550)≈1.475 |
| `tio2` / `tio2_pvd` / `tio2_sputter` | RF-sputtered TiO₂ | NIST Wang 2014 + Franta IR | **default H**; Denton Discovery 550; n(550)≈2.36; λ>1.35 µm scaled Franta |
| `tio2_eb` | E-beam TiO₂ | Franta et al. 2015 | evaporated amorphous/fine poly; n(550)≈2.35 |
| `tio2_amorphous` | Amorphous TiO₂ | Jolivet et al. 2023 @ 200 °C | ALD; NIR Sellmeier beyond ~0.82 µm |
| `tio2_a` / `tio2-a` | Anatase TiO₂ | Jolivet et al. 2023 @ 300 °C | annealed crystalline film; higher n than as-deposited PVD |
| `tio2_rutile` | Rutile (ordinary) | Jellison 2024 + Bond 1965 | **crystal upper bound**, not typical as-deposited PVD |
| `ag` | Evaporated Ag | McPeak et al. 2015 | template-stripped thermal Ag; ~0.30–1.70 µm |
| `ito` | Commercial ITO | König et al. 2014 + Drude NIR | 72 nm on BK7 to 1 µm; λ>1 µm → moderate-carrier Drude |
| `glass` | Cover glass | Schott AF32eco | alumino-borosilicate; n≈1.51 @ 590 nm |
| `pet` | PET film | weak Sellmeier fit | n(550)≈1.65 |

## Why PVD films differ from crystal

As-deposited sputtered / evaporated oxides are usually **less dense** than
single-crystal rutile, so **n is lower**. Annealing or ion-assist densification
raises n toward crystalline references (`tio2_a`, `tio2_rutile`). Sputtered
SiO₂ (`sio2_pvd`) is slightly higher-n than fused silica (`sio2`).

Typical VIS indices used here:

| Material | n @ 550 nm | Role |
|----------|------------|------|
| `sio2` / `sio2_fused` | ≈ 1.460 | **default** low-index |
| `sio2_pvd` (sputter) | ≈ 1.475 | PVD film alternative |
| `tio2` (sputter) | ≈ 2.36 | default high-index |
| `tio2_eb` | ≈ 2.35 | e-beam alternative |
| `tio2_a` (anatase) | ≈ 2.49 | annealed ALD |
| `tio2_rutile` (n_o) | ≈ 2.65 | crystal ceiling |

## File map

| CSV | Used by |
|-----|---------|
| `sio2_franta.csv` | `sio2` / `sio2_fused` |
| `sio2_pvd_lemarchand.csv` | `sio2_pvd` |
| `tio2_pvd_sputter.csv` | `tio2` / `tio2_pvd` |
| `tio2_pvd_franta.csv` | `tio2_eb` |
| `tio2_amorphous_jolivet.csv` | `tio2_amorphous` |
| `tio2_a_jolivet.csv` | `tio2_a` |
| `tio2_rutile_bond_o.csv` | `tio2_rutile` (NIR n) |
| `ag_mcpeak.csv` | `ag` |
| `ito_konig.csv` | `ito` (≤1 µm) |
| `glass_af32eco_k.csv` | `glass` (k only) |

## Rutile combination detail

- **n, VIS:** G. E. Jellison, W. F. Cureton, O. Arteaga, *Surf. Sci. Spectra* **31**, 024001 (2024), DOI [10.1116/6.0003719](https://doi.org/10.1116/6.0003719). Ordinary Sellmeier: \(n^2=1+A\lambda^2/(\lambda^2-\lambda_o^2)\) with \(A=4.8393\), \(\lambda_o=241.89\,\mathrm{nm}\) (fit 450–850 nm).
- **n, NIR:** W. L. Bond, *J. Appl. Phys.* **36**, 1674 (1965), ordinary ray tabulated 0.45–2.4 µm (`tio2_rutile_bond_o.csv`).
- **k:** same Jellison paper, polarized transmission Urbach tail for ordinary ray: \(\alpha=100\,\mathrm{cm^{-1}}\cdot\exp[(E-E_{go})/E_u]\) with \(E_{go}=3.023\,\mathrm{eV}\), \(E_u=0.0302\,\mathrm{eV}\).

## References

- P. E. Ciddor, *Appl. Opt.* **35**, 1566 (1996) — air
- L. Gao, F. Lemarchand, M. Lequime, *J. Eur. Opt. Soc. Rapid Publ.* **8**, 13010 (2013) — sputtered SiO₂
- X. Wang et al. / NIST (refractiveindex.info `main/TiO2/Wang.yml`) — RF-sputtered TiO₂
- D. Franta et al., *Appl. Surf. Sci.* **351**, 1056 (2015) — e-beam TiO₂; Proc. SPIE **9890**, 989014 (2016) — fused SiO₂
- A. Jolivet et al., *Appl. Surf. Sci.* **608**, 155214 (2023) — ALD TiO₂ (amorphous / anatase)
- G. E. Jellison et al., *Surf. Sci. Spectra* **31**, 024001 (2024) — rutile VIS
- W. L. Bond, *J. Appl. Phys.* **36**, 1674 (1965) — rutile NIR
- K. M. McPeak et al., *ACS Photonics* **2**, 326 (2015) — evaporated Ag
- T. A. F. König et al., *ACS Nano* **8**, 6182 (2014) — commercial ITO
- SCHOTT Zemax catalog 2017-01-20b — AF32eco
