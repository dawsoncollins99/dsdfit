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
| `Conditions(T, p, d_dry, H)` | physical conditions; computes the model constants |
| `welas_bins(cutoff_diameter=1.0)` | bin edges of the WELAS instrument |
| `Histogram(edges, counts)` | droplet counts in size bins (the data format) |
| `load_histogram(path)`, `save_histogram(hist, path)` | read and write histograms as CSV |
| `model_fractions(D_x, s_bar, cond, edges)` | the model's predicted fraction of droplets in each bin |
| `simulate(D_x, s_bar, cond, edges, n, seed)` | a synthetic histogram of `n` droplets |
| `fit(hist, cond)` | estimates of `D_x` and `s_bar`, with error bars |
| `predicted_D_x(sigma_s, tau, cond)` | `a² σ_s² τ`, to compare with a fitted `D_x` |
| `plot_fit(hist, result)` | plot a histogram with the true and fitted model |

**Conventions:** sizes you type are droplet **diameters in µm**; `D_x` is in µm⁴/s; `s_bar` is a fraction.

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
