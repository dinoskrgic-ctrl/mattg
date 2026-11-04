# Code for "Displacement field as a hybridization dial in magic-angle twisted trilayer graphene"

D. Škrgić and D. Stojkovic, University at Buffalo.

This folder reproduces every number, table and figure in the paper from the published data. Run the scripts from this folder:

```
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` runs the scripts in order. Each script's printed output goes to `outputs/<script>.txt` and every figure goes to `figures/`. `17_sharp_bound.py` takes about 10 minutes; each of the others takes under two minutes. A few scripts write `.npy` files that later scripts read, so keep the order (see "Dependencies" below).

## Data

`data/` holds the resistance curves of device D3 (twist angle 1.27°, filling ν = 2.1) at eight displacement fields D = 0.10–0.38 V/nm. They were digitized from Supplemental Fig. S15b of Zhang et al., Phys. Rev. Lett. **137**, 046301 (2026).

- `S15b_D3_nu2.1_digitized.csv`: all eight curves, colour-matched at 600 dpi. Columns are D (V/nm), T (K), R (kΩ). The D = 0.10, 0.14 and 0.38 V/nm traces are unreliable below about 12 K because the curves overlap there.
- `S15b_D0.10_redigitized.csv`: the D = 0.10 V/nm curve, colour-tracked a second time from the embedded raster. Columns are T (K), R (kΩ).

The analysis uses 12–78 K. `common.py` loads and cleans the curves and holds the exact Kubo model.

## Scripts and where they enter the paper

### Resistance data (Sec. IV)

| script | what it gives |
|---|---|
| `01_profile_shapes.py` | Table II (shared-profile fits for eight profile shapes), the common-offset test, per-curve offsets and amplitudes, SVD amplitudes |
| `02_synthetic_kubo.py` | Table I: the exact thermal average on synthetic curves (quartic share and position of the maximum) |
| `03_carrier_number_vs_rate.py` | how much of the field dependence a Matthiessen prefactor (a change of carrier number or mass) can carry |
| `04_subset_fits.py` | the four-parameter law on different subsets of curves, extrapolated to the lowest fields |
| `05_exact_kubo_fit.py` | the full thermal average fitted to the data: single level against symmetric ±Δ, κ and Δ |
| `06_four_parameter_law.py` | R_el, W₂, D₀, T* at D ≥ 0.20 V/nm; amplitude factors and elastic offsets at 0.14 and 0.10 V/nm |
| `07_background.py` | the field-independent background term b·T and its effect on D₀ and T* |
| `08_slope_errors.py` | the 12–32 K slopes and their errors (fit scatter combined with the spread over fit windows) |
| `09_slope_law.py` | a(1 + D²/D₀²), free exponent, quartic share, linear law, the D = 0.10 V/nm test, self-doping law, mirror offset |
| `10_rank_maximum_exponent.py` | rank-one test through the maximum, per-curve position of the maximum, width exponent n |
| `11_maximum_with_background.py` | how a field-independent background moves the apparent maximum at low field |
| `12_effective_medium.py` | smooth mirror-breaking disorder: 2D Bruggeman effective medium and its quartic share |

### Entropy and the bound (Sec. V, Appendices B and C)

| script | what it gives |
|---|---|
| `13_bdial_check.py` | numerical check of b_dial = 1 + 3I₀/π² from the O(V²) free energy |
| `14_relaxed_bound.py` | the relaxed bound Q_dial ≤ 3.23 at b_dial = 1.54 (keeping only φ ≥ \|x\|) |
| `15_excess_sign.py` | when the thermal excess φ − \|x\| is non-negative (monotonic against peaked baths) |
| `16_high_energy_floor.py` | a high-energy floor c′ adds no logarithm; the relaxed ceiling as a function of c′ |
| `17_sharp_bound.py` | the sharp bound over realizable flat bands (step mixtures, zero floor): Q_dial ≤ 2.73 |
| `18_golden_gap.py` | the maximizing single gap at 2 ln[(1+√5)/2] k_BT: C_b = 6.24, b_dial = 2.28; conformal 2.457 |
| `19_negative_floor.py` | the ceiling also holds for baths slower than conformal (c′ < 0) |

### Figures

| script | figure |
|---|---|
| `fig1_schematic.py` | Fig. 1, schematic with the rate-not-number test and the Q_dial gauge |
| `fig2_fig3_fig4.py` | Fig. 2 (the eight curves and their collapse), Fig. 3 (slopes; elastic against inelastic parts), Fig. 4 (map of Planckian baths) |
| `fig5_bound.py` | Fig. 5, Q_dial against b_dial for random realizable flat bands, with the 2.73 maximum and the relaxed envelope |

## Dependencies between scripts

- `06_four_parameter_law.py` writes `fit4.npy` and `offsets.npy`. These are read by `12_effective_medium.py` and `fig2_fig3_fig4.py`.
- `07_background.py` writes `fit_bg.npy` and `offsets_bg.npy`.
- `08_slope_errors.py` writes `slope_sigma.npy`. This is read by `09_slope_law.py` and `fig2_fig3_fig4.py`.
- `17_sharp_bound.py` writes `best_p_eq.npy`, the optimal step weights.

## Notation

κ is the Planckian coefficient of the flat band, Γ_f = κ k_B T. Δ is the flat-band level offset, and T* = Δ/κk_B in the zero-frequency profile. D₀ sets the field-independent mirror breaking, ⟨|V|²⟩ ∝ D² + D₀². φ(x) is the flat-band rate in units of k_B T as a function of x = ε/k_B T. b_dial and Q_dial = C_b/b_dial are defined in Sec. V. Resistances are in kΩ and slopes in Ω/K.
