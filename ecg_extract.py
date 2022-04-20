import torch
from transforms import PowerSpec
from ecg_dataset import EcgDataset
import math
import numpy as np

# All functions in this class must take an ecg parameter for type (tensor, numpy or torch) and shape=(8 or 12, 5000),
# and lead parameter. Each lead signal is a row

def relative_power_ratio(ecg, lead):
    """
    Section 4.1 Signal Quality, equation (1) at DOI: 10.1049/htl.2016.0020

    Return a ratio of PowerSpectrum area across Hz range: 1 -  0 to 1Hz / 0 to 40 Hz

    :param ecg: (tensor) shape=(8 or 12, 5000)
    :param lead: int lead id
    :return: float
    """
    if isinstance(ecg, np.ndarray):
        ecg = torch.from_numpy(ecg)
    signal = ecg[lead]
    P = PowerSpec()
    trfm = P(signal)
    return 1 - trfm[:10].sum() / trfm[:400].sum()


def entropy_of_hist(ecg, lead):
    """
    Section 2.2.1 at DOI: 10.1016/j.future.2020.10.024

    Get the 40 bin density histogram of the signal, take the entropy of bin heights

    :param ecg: (tensor) shape=(8 or 12, 5000)
    :param lead: int lead id
    :return: float
    """
    if isinstance(ecg, np.ndarray):
        ecg = torch.from_numpy(ecg)
    signal = ecg[lead]
    vals, bins = torch.histogram(signal, bins=40, density=True)
    res = -torch.sum(torch.log2(vals) * vals).item()
    return res


def curve_length(ecg, lead):
    """
    Section 2.2.1 at DOI: 10.1016/j.future.2020.10.024

    Compute the curve length of a given signal, the sum of all adjacent amplitude differences.

    :param ecg: (tensor) shape=(8 or 12, 5000)
    :param lead: int lead id
    :return: float
    """
    if isinstance(ecg, np.ndarray):
        ecg = torch.from_numpy(ecg)
    signal = ecg[lead]
    diffs = torch.diff(signal)
    return torch.sqrt(diffs*diffs + 1).sum()


def autocorr_similarity(ecg, lead):
    """
    From Section II and Table I at DOI: 10.1109/EMBC.2012.6346633

    Split the signal in quarters, autocorrelate each quarter (giving ACFs), compare the ACFs using cosine similarlity,
    sum all the pairwise similarity values.

    :param ecg: (tensor) shape=(8 or 12, 5000)
    :param lead: int lead id
    :return: float
    """
    if isinstance(ecg, np.ndarray):
        ecg = torch.from_numpy(ecg)

    cosSim = torch.nn.CosineSimilarity(dim=0)

    # ALL CONST! Don't change
    seg_size = 1250
    segs = 4
    nlags = 50

    ACFs = torch.zeros(segs, nlags+1)
    signal = ecg[lead]

    for i in range(segs):
        segment = signal[i*seg_size:(i+1)*seg_size]
        demeaned = segment - segment.mean()
        Frf = torch.fft.fft(demeaned, n=2560)
        acov = torch.fft.ifft(Frf * torch.conj(Frf))[:seg_size] / (1250 * torch.ones(1250))
        acov = acov.real
        acf = acov[:nlags+1] / acov[0]
        ACFs[i] = acf

    pairwiseM = torch.zeros(4, 4)

    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        A1 = ACFs[i]
        A2 = ACFs[j]
        similarity = cosSim(A1, A2)
        if not (0 <= similarity <= 1):
            similarity = min(similarity, 1.00)
        theta = math.acos(similarity)
        pairwiseM[i, j] = theta
        pairwiseM[j, i] = theta
    return torch.sum(pairwiseM, dim=1).sum().item()


def skew(ecg, lead):
    """
    From Section 2.1 at  https://doi.org/10.1098/rsif.2022.0012

    Compute the centralized skewness of the signal

    :param ecg: (tensor) shape=(8 or 12, 5000)
    :param lead: int lead id
    :return: float
    """
    if isinstance(ecg, np.ndarray):
        ecg = torch.from_numpy(ecg)

    signal = ecg[lead]
    mu = torch.mean(signal)
    sd = torch.std(signal)
    signal -= mu
    signal /= sd
    pows = torch.pow(signal, 3)
    return pows.sum() / signal.shape[0]


def kurtosis(ecg, lead):
    """
    From Section 2.1 at  https://doi.org/10.1098/rsif.2022.0012

    Compute the centralized kurtosis of the signal

    :param ecg: (tensor) shape=(8 or 12, 5000)
    :param lead: int lead id
    :return: float
    """
    if isinstance(ecg, np.ndarray):
        ecg = torch.from_numpy(ecg)

    signal = ecg[lead]
    mu = torch.mean(signal)
    sd = torch.std(signal)
    signal -= mu
    signal /= sd
    pows = torch.pow(signal, 4)
    return pows.sum() / signal.shape[0]


def snr(ecg, lead):
    """
    From Section 2.1 at  https://doi.org/10.1098/rsif.2022.0012

    Compute the signal to noise ratio index
    ratio of variance / abs(signal) variance

    :param ecg: (tensor) shape=(8 or 12, 5000)
    :param lead: int lead id
    :return: float
    """
    if isinstance(ecg, np.ndarray):
        ecg = torch.from_numpy(ecg)

    signal = ecg[lead]
    return torch.var(signal) / torch.var(np.abs(signal))


def hos(ecg, lead):
    """
    From Section 2.1 at  https://doi.org/10.1098/rsif.2022.0012

    hos = Higher Order Statistics

    :param ecg: (tensor) shape=(8 or 12, 5000)
    :param lead: int lead id
    :return: float
    """
    return abs(skew(ecg, lead)) * (kurtosis(ecg, lead) / 5.0)
