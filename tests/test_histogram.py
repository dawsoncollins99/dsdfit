import numpy as np
import pytest
from dsdfit import Histogram, load_histogram, save_histogram, welas_bins


def test_welas_bins():
    e = welas_bins()
    assert len(e) == 104                       # 103 bins above 1 um
    assert e[0] >= 1.0 and e[-1] == pytest.approx(40.0)
    assert len(welas_bins(cutoff_diameter=2.0)) < len(e)


def test_basic_attributes():
    h = Histogram([1, 2, 3, 4, 5, 6], [10000, 20000, 30000, 20000, 20000])
    assert h.n == 100000 and h.n_bins == 5
    assert np.allclose(h.fractions.sum(), 1.0)
    assert np.allclose(h.centers[0], np.sqrt(2.0))


@pytest.mark.parametrize("edges,counts", [
    ([1, 2, 3, 4, 5], [1, 2, 3]),             # wrong number of edges
    ([1, 3, 2, 4, 5, 6], [1, 2, 3, 4, 5]),    # edges not increasing
    ([1, 2, 3, 4, 5, 6], [1, 2, -3, 4, 5]),   # negative count
    ([1, 2, 3, 4, 5, 6], [0.1, 0.2, 0.3, 0.2, 0.2]),   # a normalized distribution, not counts
    ([1, 2, 3], [5, 5]),                      # too few bins
])
def test_bad_histograms_raise(edges, counts):
    with pytest.raises(ValueError):
        Histogram(edges, counts)


def test_small_n_warns():
    with pytest.warns(UserWarning):
        Histogram([1, 2, 3, 4, 5, 6], [10, 20, 30, 20, 20])


def test_csv_round_trip(tmp_path):
    h = Histogram(welas_bins(), np.arange(103) * 200)
    path = tmp_path / "h.csv"
    save_histogram(h, path, comment="test file")
    h2 = load_histogram(path)
    assert np.allclose(h2.edges, h.edges, rtol=1e-5)
    assert np.array_equal(h2.counts, h.counts)


def test_csv_errors(tmp_path):
    bad_header = tmp_path / "a.csv"
    bad_header.write_text("lo,hi,n\n1,2,5\n")
    with pytest.raises(ValueError, match="header"):
        load_histogram(bad_header)
    gap = tmp_path / "b.csv"
    gap.write_text("lower_diameter_um,upper_diameter_um,count\n1,2,5000\n2,3,5000\n3.5,4,5000\n4,5,5000\n")
    with pytest.raises(ValueError, match="contiguous"):
        load_histogram(gap)
