
import numpy as np
import pandas as pd
import polars as pl


def get_first_column_enforce_rank_1_tensor(d) -> np.ndarray:
    """
    if a data frame or ndarray, get the first column
    assert if not a 1 dimensional object
    """
    if isinstance(d, pl.DataFrame):
        assert d.shape[1] == 1
        d = d[:, 0]
    elif isinstance(d, pd.DataFrame):
        assert d.shape[1] == 1
        d = d.iloc[:, 0].values
    elif isinstance(d, np.ndarray):
        assert len(d.shape) == 1
    assert not isinstance(d, str)  # common error, string imitates iterable
    return np.asarray(d)


def _treat_zeros(bins, *, epsilon: float = 1e-6):
    epsilon = float(epsilon)
    treated = np.maximum(bins, epsilon)
    return treated / np.sum(treated)


def _calc_PSI(actual_bins, baseline_bins, *, epsilon: float = 1e-6):
    """
    Population stability index. 
    PSI = Sum [ (%actual - %baseline) * ln (%actual/%baseline) ]
    """
    epsilon = float(epsilon)
    pactual = _treat_zeros(actual_bins, epsilon=epsilon)
    pbase = _treat_zeros(baseline_bins, epsilon=epsilon)
    
    bindiff = pactual - pbase
    binlog = np.log(pactual/pbase)

    return np.sum(bindiff * binlog)


def _calc_KL(p_bins, q_bins, *, epsilon: float = 1e-6):
    """
    Calculate DKL(P || Q).
    See https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
    :param p_bins: vector of p probabilities.
    :param q_bins: vector of q probabilities.
    """
    epsilon = float(epsilon)
    assert np.min(p_bins) >= 0
    assert np.min(q_bins) >= 0
    assert np.abs(np.sum(p_bins) - 1) < 1e-6
    assert np.abs(np.sum(q_bins) - 1) < 1e-6
    dkl = np.sum(p_bins * np.log2(np.maximum(p_bins, epsilon) / np.maximum(q_bins, epsilon)))
    return dkl


def jensen_shannon_divergence(p_data, q_data, *, num: int = 11, epsilon: float = 1e-6):
    """
    Calculate the Jensen Shannon Divergence between a 
    reference data set and a new data set using percentile bins.
    Ref: https://en.wikipedia.org/wiki/Jensen–Shannon_divergence .
    """
    num = int(num)
    epsilon = float(epsilon)
    p_data = get_first_column_enforce_rank_1_tensor(p_data)
    q_data = get_first_column_enforce_rank_1_tensor(q_data)
    # build bins off joint data
    bins = np.percentile(
        list(p_data) + list(q_data),
        np.linspace(start=0, stop=100, num=num, endpoint=True)
        )
    p_counts, p_bin_edges = np.histogram(p_data, bins)
    q_counts, q_bin_edges = np.histogram(q_data, bins)
    p_dist = p_counts / np.sum(p_counts)
    q_dist = q_counts / np.sum(q_counts)
    m_dist = (p_dist + q_dist) / 2
    result = (_calc_KL(p_dist, m_dist, epsilon=epsilon) + _calc_KL(q_dist, m_dist, epsilon=epsilon)) / 2
    return result


class DistCalculator :
    """
    Calculate divergences from a reference data set.
    """
    def __init__(self, reference_data, *, num: int = 11):
        """
        calculate the reference bins and the reference distribution

        :param reference_data: data for reference distribution
        :param num: bin boundaries are np.linspace(start=0, stop=100, num=num, endpoint=True). defaults to 11, which produces deciles.
        """
        num = int(num)
        assert num > 2
        reference_data = get_first_column_enforce_rank_1_tensor(reference_data)
        # decile bins
        qbins = np.percentile(
            reference_data, 
            np.linspace(start=0, stop=100, num=num, endpoint=True)
            )
        # makes fewest reserved zero bins, so similar new data doesn't look surprising
        qbins[0] = -np.Inf
        qbins[-1] = np.Inf
        
        rcounts, bin_edges = np.histogram(reference_data, qbins)
        self.qbins = qbins
        self.reference_dist = rcounts / np.sum(rcounts)
    
    def PSI(self, dataset):
        """
        Calculate the Population Stability Index between a 
        reference data set and a new data set using percentile bins.

        General rule of thumb:
        PSI > 0.1: "small (but noticeable) change"; PSI > 0.25: "significant change"
        Your mileage may vary

        Discussion:
        https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4249&context=dissertations
        http://www.stat.wmich.edu/naranjo/PSI.pdf
        """
        dataset = get_first_column_enforce_rank_1_tensor(dataset)
        dcounts, bin_edges = np.histogram(dataset, self.qbins)
        ddist = dcounts / np.sum(dcounts)

        return _calc_PSI(ddist, self.reference_dist)
