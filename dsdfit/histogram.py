"""Histograms of droplet sizes: the one data format dsdfit works with.

THE FORMAT
----------
A histogram is two arrays:

    edges   bin edges, as droplet DIAMETERS in micrometers (um).
            Strictly increasing. There is one more edge than there are bins.
    counts  the number of droplets in each bin.
            Whole numbers (0, 1, 2, ...), one per bin.

Each count must be a number of *independent* droplets: every droplet is counted
once. (A normalized distribution, a concentration, or the same droplets counted
in many snapshots is NOT a valid set of counts; see the guide.)

On disk, a histogram is a CSV file with one row per bin:

    # optional comment lines start with '#'
    lower_diameter_um,upper_diameter_um,count
    1.0236,1.0603,412
    1.0603,1.0984,455
    ...

Bins must be contiguous: each bin's upper edge equals the next bin's lower edge.
"""
import warnings
import numpy as np

CSV_HEADER = "lower_diameter_um,upper_diameter_um,count"


class Histogram:
    """Droplet counts in size bins.

    Parameters
    ----------
    edges : array of length K+1
        Bin edges as droplet diameters [um], strictly increasing.
    counts : array of length K
        Number of droplets in each bin (whole numbers, not negative).

    Useful attributes
    -----------------
    edges, counts      the data
    n                  total number of droplets
    centers            middle of each bin (geometric mean of its edges) [um]
    fractions          counts / n, the fraction of droplets in each bin
    truth              for simulated histograms: the D_x and s_bar used to make them
    true_fractions     for simulated histograms: the exact model fractions (no noise)
    """

    def __init__(self, edges, counts, *, truth=None, true_fractions=None):
        edges = np.asarray(edges, dtype=float)
        counts = np.asarray(counts, dtype=float)

        if edges.ndim != 1 or counts.ndim != 1:
            raise ValueError("Histogram: edges and counts must be 1-D arrays.")
        if len(edges) != len(counts) + 1:
            raise ValueError(f"Histogram: there must be one more edge than counts "
                             f"(got {len(edges)} edges and {len(counts)} counts).")
        if len(counts) < 4:
            raise ValueError("Histogram: need at least 4 bins.")
        if not np.all(np.isfinite(edges)) or np.any(edges <= 0):
            raise ValueError("Histogram: edges must be positive, finite diameters in um.")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("Histogram: edges must be strictly increasing.")
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("Histogram: counts must be finite and not negative.")
        if np.any(np.abs(counts - np.round(counts)) > 1e-9):
            raise ValueError("Histogram: counts must be whole numbers of droplets. If you have a "
                             "normalized distribution or a concentration, convert it to the number "
                             "of independent droplets it represents (see the guide).")
        n = int(round(counts.sum()))
        if n == 0:
            raise ValueError("Histogram: all counts are zero.")
        if n < 10_000:
            warnings.warn(f"Histogram has only {n} droplets. dsdfit assumes plenty of droplets "
                          "(about 10,000 or more); with fewer, error bars are less reliable.",
                          stacklevel=2)

        self.edges = edges
        self.counts = np.round(counts).astype(np.int64)
        self.truth = dict(truth) if truth is not None else None
        self.true_fractions = None if true_fractions is None else np.asarray(true_fractions, float)

    @property
    def n(self):
        """Total number of droplets."""
        return int(self.counts.sum())

    @property
    def n_bins(self):
        return len(self.counts)

    @property
    def centers(self):
        """Middle of each bin [um diameter] (geometric mean of the two edges)."""
        return np.sqrt(self.edges[:-1] * self.edges[1:])

    @property
    def fractions(self):
        """Fraction of droplets in each bin (sums to 1)."""
        return self.counts / self.n

    def __repr__(self):
        return (f"Histogram({self.n_bins} bins, {self.edges[0]:.3g}-{self.edges[-1]:.3g} um diameter, "
                f"n = {self.n} droplets)")


def save_histogram(hist, path, comment=None):
    """Write a histogram to a CSV file in the dsdfit format."""
    lines = []
    if comment:
        lines += ["# " + line for line in str(comment).splitlines()]
    lines.append(CSV_HEADER)
    for lo, hi, c in zip(hist.edges[:-1], hist.edges[1:], hist.counts):
        lines.append(f"{lo:.6g},{hi:.6g},{int(c)}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_histogram(path):
    """Read a histogram from a CSV file in the dsdfit format (see the top of this module)."""
    rows = []
    header_seen = False
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not header_seen:
                if line.replace(" ", "") != CSV_HEADER:
                    raise ValueError(f"{path}, line {line_no}: expected the header '{CSV_HEADER}', "
                                     f"got '{line}'.")
                header_seen = True
                continue
            parts = line.split(",")
            if len(parts) != 3:
                raise ValueError(f"{path}, line {line_no}: expected 3 comma-separated values, got '{line}'.")
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                raise ValueError(f"{path}, line {line_no}: could not read numbers from '{line}'.") from None
    if not header_seen:
        raise ValueError(f"{path}: no header line '{CSV_HEADER}' found.")
    if not rows:
        raise ValueError(f"{path}: no bins found.")
    data = np.array(rows)
    lower, upper, counts = data[:, 0], data[:, 1], data[:, 2]
    if np.any(upper <= lower):
        raise ValueError(f"{path}: each bin's upper diameter must be larger than its lower diameter.")
    gaps = np.abs(lower[1:] - upper[:-1]) > 1e-6 * upper[:-1]
    if np.any(gaps):
        i = int(np.nonzero(gaps)[0][0])
        raise ValueError(f"{path}: bins must be contiguous, but bin {i + 1} ends at {upper[i]:g} um "
                         f"and bin {i + 2} starts at {lower[i + 1]:g} um.")
    edges = np.append(lower, upper[-1])
    return Histogram(edges, counts)


def welas_bins(cutoff_diameter=1.0):
    """Bin edges of the WELAS instrument [um diameter].

    The WELAS sorts droplets into 118 bins spaced evenly in log(diameter) between
    0.6 and 40 um. Droplets smaller than ``cutoff_diameter`` are not counted
    reliably, so bins whose lower edge is below the cutoff are dropped.
    With the default cutoff of 1 um, 103 bins remain.
    """
    edges = 0.6 * (40.0 / 0.6) ** (np.arange(119) / 118.0)
    first = int(np.nonzero(edges[:-1] >= cutoff_diameter - 1e-12)[0][0])
    return edges[first:]
