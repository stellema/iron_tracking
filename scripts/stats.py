# -*- coding: utf-8 -*-
"""

Notes:

Example:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Sat Jun 24 01:04:50 2023

"""
# -*- coding: utf-8 -*-
"""Statistical tests.

Example:

Notes:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Mon Aug  1 15:01:41 2022

"""
import math
import numpy as np
# import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
# import statsmodels.stats.weightstats as wtd


def precision(var):
    """Determine the precision to print based on the number of digits.

    Values greater than ten: the precision will be zero decimal places.
    Values less than ten but greater than one: print one decimal place.
    Values less than one: print two decimal places.


    Args:
        var (DataArray): Transport dataset

    Returns:
        p (list): The number of decimal places to print
    """
    # List for the number of digits (n) and decimal place (p).
    n, p = 1, 1

    tmp = abs(var.item())
    n = int(math.log10(tmp)) + 1
    if n == 1:

        p = 1 if tmp >= 1 else 2
    elif n == 0:
        p = 2
    elif n == -1:
        p = 3
    return p


def format_pvalue_str(p):
    """Format significance p-value string to correct decimal places.

    p values greater than 0.01 rounded to two decimal places.
    p values between 0.01 and 0.001 rounded to three decimal places.
    p values less than 0.001 are just given as 'p>0.001'
    Note that 'p=' will also be included in the string.
    """
    if p <= 0.001:
        sig_str = 'p<0.001'
    elif p <= 0.01 and p >= 0.001:
        sig_str = 'p=' + str(np.around(p, 3))
    else:
        if p <= 0.05:
            sig_str = 'p<' + str(np.around(p, 3))
        else:
            sig_str = 'p=' + str(np.around(p, 3))

    return sig_str


def test_signifiance(x, y):
    tdim = x.dims[0]
    def resample(ds):
        return ds.resample({tdim: "Y"}).mean(tdim, keep_attrs=True)
    x, y = x.dropna(tdim, 'all'), y.dropna(tdim, 'all')
    x, y = resample(x), resample(y)
    t, p = stats.wilcoxon(x, y)
    p = format_pvalue_str(p)
    return p


def weighted_bins_fd(ds, weights):
    """Weighted Freedman Diaconis Estimator bin width (number of bins).

    Bin width:
        h = 2 * IQR(values) / cubroot(values.size)

    Number of bins:
        nbins = (max(values) - min(values)) / h

    """
    # Sort data and weights.
    ind = np.argsort(ds).values
    d = ds[ind]
    w = weights[ind]

    # Interquartile range.
    pct = 1. * w.cumsum() / w.sum() * 100  # Convert to percentiles?
    iqr = np.interp([25, 75], pct, d)
    iqr = np.diff(iqr)

    # Freedman Diaconis Estimator (h=bin width).
    h = 2 * iqr / np.cbrt(ds.size)
    h = h[0]

    # Input data max/min.
    data_range = [ds.min().item(),  ds.max().item()]

    # Numpy conversion from bin width to number of bins.
    nbins = int(np.ceil(np.diff(data_range) / h)[0])
    return h, nbins, data_range


def get_min_weighted_bins(ds, weights):
    # Find number of bins based on combined hist/proj data range.
    h0, _, r0 = weighted_bins_fd(ds[0], weights[0])
    h1, _, r1 = weighted_bins_fd(ds[1], weights[1])

    # Data min & max of both datasets.
    r = [min(np.floor([r0[0], r1[0]])), max(np.ceil([r0[1], r1[1]]))]

    # Number of bins for combined data range (use smallest bin width).
    bins = int(np.ceil(np.diff(r) / min([h0, h1])))
    return bins


def get_source_transit_mode(ds, var, exp, zone, where='mode'):

    plt.ioff()  # Supress figure.
    ds = [ds.isel(exp=i).sel(zone=zone).dropna('traj', 'all') for i in [0, 1]]
    bins = get_min_weighted_bins([d[var] for d in ds], [d.u for d in ds])
    dz = ds[exp]

    plot = sns.histplot(x=dz[var], weights=dz.u, bins=bins, kde=True,
                        kde_kws=dict(bw_adjust=0.5))
    hist = plot.get_lines()[-1]
    x, y = hist.get_xdata(), hist.get_ydata()

    if where == 'mode':
        xx = [x[np.argmax(y) + i] for i in [0, 1]]
    else:
        xx = x[sum(np.cumsum(y) < sum(y) * where)]
    plt.ion()  # End supress figure.
    return xx

# # Man whitney test (histogram diff) returns zero while ttest_ind p = 0.158.
# # r sample sizes above ~20, approximation using the normal distribution is fairly good
# binned data -mwu=value=0.00035 ttest_ind=0.158
# Whitney U test can have inflated type I error rates even in large samples (especially if the variances of two populations are unequal and the sample sizes are different)
# Brunner-Munzel (n <50 samples?)and the Fligner–Policello test

# Kolmogorov–Smirnov test
if __name__ == "__main__":
    from fncs import source_dataset
    from tools import idx
    import matplotlib.pyplot as plt

    z = 1
    var = 'age'
    lon = 190
    ds = source_dataset(lon, sum_interior=True)
    dx = [ds.sel(zone=z).isel(exp=i).dropna('traj') for i in range(2)]
    dv = [dx[i][var].sortby(dx[i][var]) for i in range(2)]
    weights = [d.u / 1948 for d in dx]
    bins = get_min_weighted_bins(dv, weights)

    xx, yy = [], []

    for exp in range(2):
        n = dx[exp].names.item()
        fig, ax = plt.subplots(figsize=(7, 9))
        ax = sns.histplot(x=dv[exp], weights=weights[exp], bins=bins, ax=ax,
                          kde=True, kde_kws=dict(bw_adjust=0.5))
        ax.set_xlim(0, 2000)
        # hist = ax.get_lines()[-1]
        # xx.append(hist.get_xdata())
        # yy.append(hist.get_ydata())
        yy.append(np.array([h.get_height() for h in ax.patches]))
        plt.clf()


    stats.mannwhitneyu(*dv, use_continuity=False)

    stats.brunnermunzel(*dv)
    stats.ttest_ind(*dv, equal_var=False)
    stats.ks_2samp(*dv)
    stats.ks_2samp(*dv, alternative='less')
    stats.epps_singleton_2samp(*dv)
    wtd.ttest_ind(*dv, weights=tuple(weights), usevar='unequal')

    stats.ttest_ind(*yy)
    stats.ttest_ind(*yy, equal_var=False)
    stats.mannwhitneyu(*yy)
    stats.wilcoxon(*yy)
    stats.wilcoxon(*yy, zero_method='pratt')
    stats.wilcoxon(*yy, zero_method='zsplit')


    stats.mannwhitneyu(*dv)
