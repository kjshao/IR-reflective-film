# Material optical-constant sources

Numeric tables are derived from the public-domain
[refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database) (CC0)
unless noted.

| Key | Material | Source | Notes |
|-----|----------|--------|-------|
| `air` | Standard air | Ciddor 1996 (formula) | dry air 15 °C, 101.325 kPa, 450 ppm CO₂; k = 0 |
| `sio2` | Fused silica | Franta et al. 2016 (tabulated n,k) | matches Malitson n to ~2e-4 in VIS |
| `tio2_a` / `tio2-a` | TiO₂ anatase | Jolivet et al. 2023 (tabulated n,k) | ALD film @ 300 °C; beyond table: Devore-style Sellmeier fit, k→0 |
| `tio2_rutile` / `tio2-rutile` | TiO₂ rutile (ordinary) | **Jellison 2024** (n Sellmeier + Urbach k) **+ Bond 1965** (NIR n) | blend 0.85–0.95 µm; polycrystalline ≈ n_o |
| `glass` | Phone/display cover glass | Schott AF32eco (Sellmeier + k) | alkali-free alumino-borosilicate; n≈1.51 @ 590 nm (Gorilla Glass core ≈1.50) |
| `tio2` | alias | → `tio2_a` | backward compatibility |

### Rutile combination detail

- **n, VIS:** G. E. Jellison, W. F. Cureton, O. Arteaga, *Surf. Sci. Spectra* **31**, 024001 (2024), DOI [10.1116/6.0003719](https://doi.org/10.1116/6.0003719). Ordinary Sellmeier: \(n^2=1+A\lambda^2/(\lambda^2-\lambda_o^2)\) with \(A=4.8393\), \(\lambda_o=241.89\,\mathrm{nm}\) (fit 450–850 nm).
- **n, NIR:** W. L. Bond, *J. Appl. Phys.* **36**, 1674 (1965), ordinary ray tabulated 0.45–2.4 µm (`tio2_rutile_bond_o.csv`).
- **k:** same Jellison paper, polarized transmission Urbach tail for ordinary ray: \(\alpha=100\,\mathrm{cm^{-1}}\cdot\exp[(E-E_{go})/E_u]\) with \(E_{go}=3.023\,\mathrm{eV}\), \(E_u=0.0302\,\mathrm{eV}\). VIS–NIR k is negligible; UV above the gap is a capped continuation, not full ellipsometric ε₂.

References (DOI / catalog):
- P. E. Ciddor, Appl. Opt. 35, 1566 (1996)
- D. Franta et al., Proc. SPIE 9890, 989014 (2016) — SiO₂
- A. Jolivet et al., Appl. Surf. Sci. 608, 155214 (2023) — anatase TiO₂
- G. E. Jellison et al., Surf. Sci. Spectra 31, 024001 (2024) — rutile VIS n,k
- W. L. Bond, J. Appl. Phys. 36, 1674 (1965) — rutile NIR n
- SCHOTT Zemax catalog 2017-01-20b — AF32eco
