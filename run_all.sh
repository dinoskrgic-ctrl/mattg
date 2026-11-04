#!/usr/bin/env bash
# Runs every script in order and writes its printed output to outputs/<script>.txt.
# 17_sharp_bound.py takes about 10 minutes; everything else takes under 2 minutes each.
set -e
cd "$(dirname "$0")"
mkdir -p outputs figures
for f in 01_profile_shapes 02_synthetic_kubo 03_carrier_number_vs_rate 04_subset_fits 05_exact_kubo_fit \
         06_four_parameter_law 07_background 08_slope_errors 09_slope_law 10_rank_maximum_exponent \
         11_maximum_with_background 12_effective_medium 13_bdial_check 14_relaxed_bound 15_excess_sign \
         16_high_energy_floor 17_sharp_bound 18_golden_gap 19_negative_floor \
         fig1_schematic fig2_fig3_fig4 fig5_bound; do
  echo "running $f"
  python3 "$f.py" > "outputs/$f.txt" 2>&1
done
echo "done: outputs/ and figures/"
