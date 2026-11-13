"""dsdfit: estimate D_x and s_bar from droplet size histograms.

The functions you need:

    Conditions(T, p, d_dry, H, alpha_c, alpha_T, fallout_factor)    physical conditions
    welas_bins(cutoff_diameter=1.0)                 bin edges of the WELAS instrument
    Histogram(edges, counts)                        droplet counts in size bins
    load_histogram(path), save_histogram(h, path)   read / write histograms as CSV
    model_fractions(D_x, s_bar, cond, edges)        the model's predicted size distribution
    simulate(D_x, s_bar, cond, edges, n, seed)      a synthetic histogram of n droplets
    model_evolution(D_x, s_bar, cond, edges, times) droplet numbers over time (pulse or empty start)
    fit(hist, cond)                                 estimates of D_x and s_bar, with error bars
    predicted_D_x(sigma_s, tau, cond)               a^2 sigma_s^2 tau, to compare with the fit
    plot_fit(hist, result)                          plot a histogram and the fitted model

Conventions: sizes you type are DIAMETERS in micrometers; D_x is in um^4/s;
s_bar is a fraction (0.0009 means 0.09%).
"""
from .conditions import Conditions
from .histogram import Histogram, load_histogram, save_histogram, welas_bins
from .simulate import model_fractions, simulate
from .evolve import model_evolution
from .fit import fit, FitResult
from .compare import predicted_D_x
from .plots import plot_fit

__version__ = "0.3.0"
__all__ = ["Conditions", "Histogram", "load_histogram", "save_histogram", "welas_bins",
           "model_fractions", "simulate", "model_evolution", "fit", "FitResult", "predicted_D_x", "plot_fit"]
