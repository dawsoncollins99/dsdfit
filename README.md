# dsdfit

Estimate how much turbulence broadens cloud droplet sizes, from a histogram of droplet sizes.

`dsdfit` fits a steady Fokker–Planck model of droplet growth to a droplet size histogram. It returns two numbers, each with an error bar:

- **D_x**: how strongly fluctuations in supersaturation spread droplet sizes apart (µm⁴/s);
- **s_bar**: the mean supersaturation, as a fraction (0.0009 means 0.09%).

It can also make synthetic histograms from the same model, so you can test how well the method works before you trust it on real or simulated data.

**Start with the guide:** [`docs/guide.pdf`](docs/guide.pdf). It explains the physics, the method and the code, and suggests what to try.

## Install

```bash
git clone https://github.com/dawsoncollins99/dsdfit.git
cd dsdfit
pip install -e ".[notebook]"
```

This installs `dsdfit` with numpy, scipy, matplotlib and Jupyter. Then open `examples/example.ipynb`.

`examples/codt_first_look.ipynb` fits droplet size distributions from C-ODT simulations and compares the results with C-ODT's supersaturation statistics. The C-ODT output is not included in this repository, and reading it needs `netCDF4` (`pip install -e ".[codt]"`).

## Quick start

```python
from dsdfit import Conditions, welas_bins, simulate, fit, plot_fit

cond  = Conditions()            # temperature, pressure, dry particle size, chamber height
edges = welas_bins()            # bin edges of the WELAS instrument (diameters, µm)

hist = simulate(D_x=0.3, s_bar=0.0009, cond=cond, edges=edges, n=100_000, seed=1)
result = fit(hist, cond)
print(result)                   # estimates, error bars and a fit check
plot_fit(hist, result)
```

## The functions

| Function | What it does |
|---|---|
| `Conditions(T, p, d_dry, H, alpha_c, alpha_T, fallout_factor)` | physical conditions; computes the model constants |
| `welas_bins(cutoff_diameter=1.0)` | bin edges of the WELAS instrument |
| `Histogram(edges, counts)` | droplet counts in size bins (the data format) |
| `load_histogram(path)`, `save_histogram(hist, path)` | read and write histograms as CSV |
| `model_fractions(D_x, s_bar, cond, edges)` | the model's predicted fraction of droplets in each bin |
| `simulate(D_x, s_bar, cond, edges, n, seed)` | a synthetic histogram of `n` droplets |
| `model_evolution(D_x, s_bar, cond, edges, times, start)` | expected droplet numbers in each bin over time, after a pulse (`start="pulse"`) or from an empty chamber (`start="empty"`) |
| `fit(hist, cond)` | estimates of `D_x` and `s_bar`, with error bars |
| `predicted_D_x(sigma_s, tau, cond)` | `a² σ_s² τ`, to compare with a fitted `D_x` |
| `plot_fit(hist, result)` | plot a histogram with the true and fitted model |

**Conventions:** sizes you type are droplet **diameters in µm**; `D_x` is in µm⁴/s; `s_bar` is a fraction.

**Slower growth of small droplets (optional).** Giving the accommodation coefficients, for example `Conditions(alpha_c=0.04, alpha_T=1.0)` (C-ODT's values), slows the growth of small droplets by a factor r/(r + ℓ), with a kinetic length ℓ ≈ 1.5 µm at 288 K. By default there is no such correction (ℓ = 0). With it, the model works with y = (r + ℓ)² instead of r², and `D_x ≈ a² σ_s² τ` still applies. Section 2 of the guide explains it.

**Faster fallout (optional).** The model assumes droplets are spread evenly in height, so droplets of radius r fall out at rate k r². If they are removed faster (for example because they crowd toward the floor), multiply that rate by a fallout factor: a number, `Conditions(fallout_factor=1.3)`, or a function of radius r in µm, `Conditions(fallout_factor=lambda r: 1 + 0.005 * r**2)`. The default is 1. A histogram cannot reveal a wrong fallout rate, and an error in it goes straight into both estimates, so it is worth measuring.

## Histogram format

A CSV file with one row per bin. Bins must be contiguous, and counts must be whole numbers of independent droplets:

```
# comment lines start with '#'
lower_diameter_um,upper_diameter_um,count
1.0233,1.06038,88
1.06038,1.0988,75
...
```

`data/example_histogram.csv` is an example. Section 4 of the guide explains the format in detail, including the pitfalls of building histograms from simulation output.

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Credits

The finite-volume solver follows Steve Krueger's MATLAB implementation of the steady Fokker–Planck model with Köhler growth and Scharfetter–Gummel fluxes.

## License

MIT; see `LICENSE`.
