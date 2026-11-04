"""Figure 1: schematic and the two key results (one column).
Needs numpy and matplotlib (SciencePlots used if installed). Run from this folder: python3 fig1_schematic.py
(a) Alternating-twist stack with mirror M_z and a gate D; band picture with the Dirac cone d at E_F borrowing flat-band
    weight f at +-Delta (Planckian width) through V ~ D.
(b) Device D3, nu = 2.1 (Zhang et al., PRL 137, 046301 (2026), Suppl. Fig. S15b): inelastic 12-32 K slope and elastic
    offset R_el, both normalized; slope law a(1 + D^2/D0^2) with a = 3.41 Ohm/K, D0 = 0.129 V/nm (09_slope_law.py);
    elastic offsets from 06_four_parameter_law.py; open symbols (D = 0.10 V/nm) were not fitted.
(c) Gauge for Q_dial = C_b/b_dial (Sec. V). Needle: conformal flat band, 2 pi^3/21 zeta(3) = 2.457. Shaded: allowed for
    hybridization into a flat band with a monotonic bath and no positive high-energy floor, Q_dial <= 2.73
    (17_sharp_bound.py). Tick 4.91: the same conformal bath in its own channel."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Wedge, Circle, FancyBboxPatch

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    plt.rcParams.update({"font.family": "serif", "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True, "axes.linewidth": 0.6})
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6})

C_D, C_F, C_G, C_EL = "#1f5fa8", "#8c510a", "#b2182b", "#e07b00"

# ---------------- data ----------------
D        = np.array([0.10, 0.14, 0.20, 0.24, 0.27, 0.30, 0.34, 0.38])
slope    = np.array([5.48, 7.26, 12.07, 15.35, 18.16, 21.52, 27.53, 32.83])
slope_e  = np.array([0.60, 0.26, 0.33, 0.50, 0.56, 0.71, 0.91, 1.17])   # conservative errors (08_slope_errors.py)
R_el     = np.array([0.388, 0.469, 0.610, 0.641, 0.649, 0.641, 0.643, 0.613])
a, D0 = 3.41, 0.129
Q_CONF_DIAL = 2*np.pi**3/(21*1.2020569)      # 2.457
Q_CEIL      = 2.734
Q_OWN       = 2*np.pi**3/(7*1.2020569)/1.5   # 4.913

fig = plt.figure(figsize=(3.375, 3.1))
gs = fig.add_gridspec(2, 2, height_ratios=[0.66, 1.05], width_ratios=[1.0, 1.0], hspace=0.10, wspace=0.08,
                      left=0.02, right=0.98, top=0.985, bottom=0.10)

# ================= (a) schematic =================
s = fig.add_subplot(gs[0, :]); s.set_xlim(-1.3, 20); s.set_ylim(0.1, 6.0); s.axis("off"); s.set_aspect("equal", adjustable="datalim")
# stack: three sheets drawn as slanted parallelograms, middle one rotated
def sheet(y, sh, ang):
    P = np.array([[0.6+sh, y+0.5], [4.6+sh, y+0.5], [4.6-sh, y-0.5], [0.6-sh, y-0.5]])
    s.add_patch(Polygon(P, closed=True, fc="#e3e3e3", ec="0.25", lw=0.5, zorder=2))
    clip = Polygon(P, closed=True, transform=s.transData)
    for c in np.linspace(-3, 8, 26):            # lattice lines at the layer's twist angle
        xs = np.array([c-2, c+2]); ln, = s.plot(xs, y+np.tan(np.deg2rad(ang))*(xs-c), color="0.55", lw=0.3, zorder=3)
        ln.set_clip_path(clip)
for y, sh, an in ((5.0, 0.5, 62), (3.3, -0.5, 48), (1.6, 0.5, 62)):
    sheet(y, sh, an)
s.plot([-1.25, 5.4], [3.3, 3.3], ls=(0, (2, 1.5)), color="0.3", lw=0.7, zorder=4)
s.text(-1.25, 3.45, r"$M_z$", fontsize=6, color="0.3", ha="left", va="bottom")
s.text(2.65, 0.35, "twisted trilayer", fontsize=6, color="0.3", ha="center")
# gate knob
kx, ky = 6.9, 3.3
s.add_patch(Circle((kx, ky), 0.95, fc="white", ec=C_G, lw=1.0))
th = np.deg2rad(55); s.plot([kx, kx+0.8*np.cos(th)], [ky, ky+0.8*np.sin(th)], color=C_G, lw=1.4, solid_capstyle="round")
for t in np.deg2rad(np.linspace(200, -20, 7)):
    s.plot([kx+1.02*np.cos(t), kx+1.22*np.cos(t)], [ky+1.02*np.sin(t), ky+1.22*np.sin(t)], color=C_G, lw=0.5)
s.text(kx, ky-1.55, r"gate $D$", color=C_G, fontsize=6.5, ha="center", va="top")
s.add_patch(FancyArrowPatch((5.55, 1.2), (5.55, 5.4), arrowstyle="-|>", mutation_scale=6, color=C_G, lw=0.9))
# bands
x0, yF = 13.6, 3.3
s.plot([9.4, 19.6], [yF, yF], color="0.6", lw=0.5, ls="--")
s.text(19.6, yF+0.1, r"$E_F$", fontsize=5.8, color="0.45", ha="right", va="bottom")
for sg in (+1, -1):
    yy = yF + sg*2.0
    s.add_patch(Polygon([[9.6, yy-0.36], [19.4, yy-0.36], [19.4, yy+0.36], [9.6, yy+0.36]], closed=True,
                        fc=C_F, ec="none", alpha=0.22))
    s.plot([9.6, 19.4], [yy, yy], color=C_F, lw=1.2)
    s.text(19.4, yy+sg*0.45, r"$+\Delta$" if sg > 0 else r"$-\Delta$", fontsize=5.6, color=C_F, ha="right",
           va="bottom" if sg > 0 else "top")
s.text(9.45, yF+2.0, "flat band $f$", fontsize=6, color=C_F, ha="left", va="bottom")
k = np.linspace(-1.0, 1.0, 80); y0 = yF-0.95
s.plot(x0+k, y0+2.2*np.abs(k), color=C_D, lw=1.4)
s.text(x0-1.15, y0+0.35, "Dirac $d$", fontsize=6, color=C_D, ha="right", va="center")
kF = (yF-y0)/2.2
for xs in (x0+kF, x0-kF):
    s.plot(xs, yF, "o", ms=2.6, color=C_D, zorder=5)
    for sg in (+1, -1):
        s.add_patch(FancyArrowPatch((xs, yF+sg*0.12), (xs+0.35*np.sign(xs-x0), yF+sg*1.62), arrowstyle="<|-|>",
                                    mutation_scale=4.5, color=C_G, lw=0.6, connectionstyle="arc3,rad=-0.25"))
s.text(x0+kF+0.75, yF+0.85, r"$V\propto D$", color=C_G, fontsize=6.2, va="center")
s.text(-1.3, 6.0, "(a)", fontsize=7, va="top")

# ================= (b) rate, not number =================
ax = fig.add_subplot(gs[1, 0])
law = lambda d: a*(1+d**2/D0**2); n_in = law(0.30); n_el = R_el[D >= 0.20].mean(); fit = D >= 0.14
DD = np.linspace(0, 0.42, 200)
ax.plot(DD, law(DD)/n_in, "-", color=C_D, lw=0.9)
ax.errorbar(D[fit], slope[fit]/n_in, slope_e[fit]/n_in, fmt="o", ms=2.6, color=C_D, elinewidth=0.5, capsize=0)
ax.errorbar(D[~fit], slope[~fit]/n_in, slope_e[~fit]/n_in, fmt="o", ms=2.6, mfc="white", color=C_D, elinewidth=0.5, capsize=0)
ax.plot(D[fit], R_el[fit]/n_el, "s", ms=2.4, color=C_EL); ax.plot(D[~fit], R_el[~fit]/n_el, "s", ms=2.4, mfc="white", color=C_EL)
ax.axhline(1, color=C_EL, lw=0.5, ls=":")
ax.text(0.02, 1.82, "inelastic", color=C_D, fontsize=6, ha="left", va="center")
ax.text(0.02, 1.62, r"$\propto D^2+D_0^2$", color=C_D, fontsize=6, ha="left", va="center")
ax.text(0.41, 0.82, "elastic", color=C_EL, fontsize=6, ha="right", va="top")
ax.set_xlim(0, 0.42); ax.set_ylim(0, 2.05); ax.set_xticks([0, 0.2, 0.4]); ax.set_yticks([0, 1, 2])
ax.set_xlabel(r"$D$ (V/nm)", labelpad=1); ax.set_ylabel(r"part of $R_{xx}$ (norm.)", labelpad=1)
ax.text(-0.27, 1.04, "(b)", transform=ax.transAxes, fontsize=7, va="top")

# ================= (c) the dial gauge =================
g = fig.add_subplot(gs[1, 1]); g.set_xlim(-1.18, 1.18); g.set_ylim(-0.62, 1.22); g.set_aspect("equal"); g.axis("off")
QMAX = 5.0; ang = lambda q: 180.0*(1-q/QMAX)
R0, R1 = 0.68, 1.0
g.add_patch(Wedge((0, 0), R1, ang(Q_CEIL), 180, width=R1-R0, fc="#cfe3f3", ec="none"))
g.add_patch(Wedge((0, 0), R1, 0, ang(Q_CEIL), width=R1-R0, fc="#f6d2d2", ec="none"))
g.add_patch(Wedge((0, 0), R1, 0, 180, width=R1-R0, fc="none", ec="0.3", lw=0.6))
for q in range(6):
    t = np.deg2rad(ang(q)); g.plot([R0*np.cos(t), (R0+0.07)*np.cos(t)], [R0*np.sin(t), (R0+0.07)*np.sin(t)], color="0.3", lw=0.5)
    g.text(0.56*np.cos(t), 0.56*np.sin(t), str(q), fontsize=5.6, ha="center", va="center", color="0.3")
for q, lab, col in ((Q_CEIL, "2.73", "k"), (Q_OWN, "4.91", "0.35")):
    t = np.deg2rad(ang(q)); g.plot([R0*np.cos(t), (R1+0.05)*np.cos(t)], [R0*np.sin(t), (R1+0.05)*np.sin(t)], color=col, lw=0.9 if col == "k" else 0.6)
    if q < 4.5: g.text((R1+0.13)*np.cos(t), (R1+0.13)*np.sin(t), lab, fontsize=5.6, ha="center", va="center", color=col)
    else: g.text(R1+0.04, -0.03, lab, fontsize=5.6, ha="right", va="top", color=col)
t = np.deg2rad(ang(Q_CONF_DIAL))
g.add_patch(FancyArrowPatch((0, 0), (0.9*np.cos(t), 0.9*np.sin(t)), arrowstyle="-|>", mutation_scale=7, color=C_G, lw=1.4))
g.add_patch(Circle((0, 0), 0.06, fc=C_G, ec="none", zorder=5))
g.text(-0.5, 1.14, r"conformal 2.46", color=C_G, fontsize=6.2, ha="center", va="center")
t1 = np.deg2rad(150); g.text(0.84*np.cos(t1), 0.84*np.sin(t1), "allowed", fontsize=5.2, color="#2b6ca3", ha="center", va="center", rotation=60)
t2 = np.deg2rad(20); g.text(0.84*np.cos(t2), 0.84*np.sin(t2), "excluded", fontsize=5.2, color="#a33", ha="center", va="center", rotation=-70)
g.text(0, -0.2, r"$Q_{\rm dial}=\dfrac{\partial(\hbar/\tau)}{\partial(C/T\ {\rm log})}$", fontsize=6.2, ha="center", va="center")
g.text(0, -0.5, "coupling-free, per $D^2$", fontsize=5.4, ha="center", va="center", color="0.35")
g.text(-1.18, 1.22, "(c)", fontsize=7, va="top")

fig.savefig("figures/fig1_schematic.pdf")
fig.savefig("figures/fig1_schematic.png", dpi=300)
print("wrote figures/fig1_schematic.pdf and figures/fig1_schematic.png")
