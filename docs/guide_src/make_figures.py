"""Figures for the guide, made with dsdfit itself (run from docs/guide_src)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dsdfit import Conditions, welas_bins, model_fractions, simulate, fit, plot_fit
from dsdfit.plots import log_diameter_axis

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
cond, edges = Conditions(), welas_bins()
centers = np.sqrt(edges[:-1] * edges[1:])

# 1. what D_x and s_bar do
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
for D in [0.1, 0.3, 1.0, 3.0]:
    axes[0].plot(centers, model_fractions(D, 0.0009, cond, edges), label=rf"$D_x$ = {D:g}")
axes[0].set_title(r"changing $D_x$  ($\bar s$ = 0.0009)")
for s in [0.0006, 0.0009, 0.0012, 0.0015]:
    axes[1].plot(centers, model_fractions(0.3, s, cond, edges), label=rf"$\bar s$ = {s:g}")
axes[1].set_title(r"changing $\bar s$  ($D_x$ = 0.3 $\mu$m$^4$/s)")
for ax in axes:
    log_diameter_axis(ax, edges[0], edges[-1])
    ax.set_xlabel(r"droplet diameter ($\mu$m)")
    ax.legend(frameon=False, loc="upper left")
axes[0].set_ylabel("fraction of droplets in bin")
fig.savefig("figs/what_params_do.pdf")
plt.close(fig)

# 2. one histogram and its fit
hist = simulate(0.3, 0.0009, cond, edges, n=100_000, seed=1)
res = fit(hist, cond)
fig, ax = plt.subplots(figsize=(4.6, 2.6))
plot_fit(hist, res, ax=ax)
fig.savefig("figs/one_fit.pdf")
plt.close(fig)
open("figs/one_fit_summary.txt", "w").write(str(res) + "\n")

# 3. recovery: 20 histograms
results = [fit(simulate(0.3, 0.0009, cond, edges, n=100_000, seed=s), cond) for s in range(20)]
D = np.array([r.D_x for r in results]); Derr = np.array([r.D_x_err for r in results])
S = np.array([r.s_bar for r in results]); Serr = np.array([r.s_bar_err for r in results])
row = np.arange(1, 21)
fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), sharey=True)
axes[0].errorbar(D, row, xerr=Derr, fmt="o", color="k", ms=3.5, elinewidth=1)
axes[0].axvline(0.3, color="tab:blue", label="true value")
axes[0].set_xlabel(r"$D_x$ ($\mu$m$^4$/s)"); axes[0].set_ylabel("histogram")
axes[1].errorbar(100 * S, row, xerr=100 * Serr, fmt="o", color="k", ms=3.5, elinewidth=1)
axes[1].axvline(0.09, color="tab:blue")
axes[1].set_xlabel(r"$\bar s$ (%)")
axes[0].set_yticks([1, 5, 10, 15, 20])
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.05), frameon=False)
fig.savefig("figs/recovery.pdf")
plt.close(fig)
print("inside 1 error bar:", int(np.sum(np.abs(D - 0.3) < Derr)), int(np.sum(np.abs(S - 0.0009) < Serr)))
print("inside 2 error bars:", int(np.sum(np.abs(D - 0.3) < 2 * Derr)), int(np.sum(np.abs(S - 0.0009) < 2 * Serr)))
print(res)
