"""One plotting helper: a histogram with the model on top of it."""
import numpy as np
import matplotlib.pyplot as plt


def log_diameter_axis(ax, lo, hi):
    """Log-scaled x axis with plain tick labels (1, 2, 5, 10, ...) between lo and hi."""
    ax.set_xscale("log")
    ticks = [t for t in (m * 10.0**e for e in range(-2, 4) for m in (1, 2, 5)) if lo / 1.1 <= t <= hi * 1.1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()


def plot_fit(hist, result=None, ax=None):
    """Plot the fraction of droplets in each bin, with the true and fitted model if available.

    Parameters
    ----------
    hist : Histogram
    result : FitResult, optional
        If given, the fitted model is drawn as a dashed line.
    ax : matplotlib Axes, optional

    Returns
    -------
    the matplotlib Axes, so you can add to the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.8))
    x = hist.centers
    ax.plot(x, hist.fractions, "o", ms=3, color="0.3", label=f"data ({hist.n:,} droplets)")
    if hist.true_fractions is not None:
        ax.plot(x, hist.true_fractions, "-", lw=3, color="tab:blue", alpha=0.45, label="true model")
    if result is not None:
        ax.plot(x, result.fractions, "--", lw=1.6, color="tab:orange", label="fitted model")
    log_diameter_axis(ax, hist.edges[0], hist.edges[-1])
    ax.set_xlabel("droplet diameter (µm)")
    ax.set_ylabel("fraction of droplets in bin")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    return ax
