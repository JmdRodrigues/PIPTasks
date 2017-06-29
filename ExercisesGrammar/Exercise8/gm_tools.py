from pylab import *
from scipy.signal import medfilt
from D_gmtools.PeakFinder import detect_peaks
import numpy as np
import novainstrumentation as ni


def stat_white(s):
    return (s - np.mean(s)) / np.std(s)


def normalization(signal, newMin, newMax, xmin=None, xmax=None):
    if xmin is None:
        xmin = np.min(signal)
    if xmax is None:
        xmax = np.max(signal)
    return (signal - xmin) * (newMax - newMin) / (xmax - xmin) + newMin


def merge_chars(string_matrix):
    """
    Function performs the merge of the strings generated with each method. The function assumes
    that each string is organized in the StringMatrix argument as a column.
    The function returns the merged string.
    """
    col = np.size(string_matrix, axis=0)

    lines = np.size(string_matrix, axis=1)
    Str = ""
    for l in range(0, lines):
        for c in range(0, col):
            Str += str(string_matrix[c][l])

    return Str


def convert2str(s, f=2, thr=0.25, win_size=51):
    s1 = (s >= np.mean(s) * thr) * 1.

    # Quantization of the derivative
    ds1 = normalization(np.diff(s), -1, 1)
    ds1 = np.clip(np.around(ds1 * f), -1, 1)
    ds1 = medfilt(ds1, win_size)

    x = np.empty(len(ds1)+1, dtype='str')
    x[find(ds1 == -1)] = '-'
    x[find(ds1 == 0)] = '_'
    x[find(ds1 == 1)] = '+'

    # Merge strings in matrix
    Mat = [s1[:-1].astype(int), x[1:]]
    Str = merge_chars(Mat)

    return Str, x

def mergeChars(StringMatrix):
    col = np.size(StringMatrix, axis=0)
    lines = len(StringMatrix[0])
    Str = ""
    for l in range(0, lines):
        for c in range(0, col):
            Str += str(StringMatrix[c][l])

    return Str


def get_magnitude(s):
    return sqrt(s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2)

def PiecewiseLinearFit(t, s, fs, winSize, ord=1):
    ss = np.array([])
    it = len(s)//winSize + 1
    for i in range(0, it):
        p1 = np.poly1d(np.polyfit(t[i*winSize:(i+1)*winSize], s[i*winSize:(i+1)*winSize], deg=ord))
        ss = np.append(ss, p1(t[i*winSize:(i+1)*winSize]))

    return ss

def R_R(t, s, X):
    RR = np.zeros(len(s))
    RR2 = []
    peaks = detect_peaks(s, mph=0.2*max(s))
    drr = np.diff(np.diff(peaks))
    drr = drr / max(drr)
    for i in range(0, len(peaks)):
        print(peaks[i])
        if (i == 0):
            RR[0:peaks[i]] = [0 for a in range(0, peaks[i])]
        elif(i<len(peaks)-1):

            RR[peaks[i-1]:peaks[i]] = [drr[i-1] for a in range(peaks[i-1], peaks[i])]
    dRR = np.array([RR>0]).astype(int)[0]
    return RR, dRR


def R_F_Amp(Signal, sup1, sup2):
    # detect all valleys
    val = detect_peaks(Signal, valley=True)
    # final pks array
    pks = []
    # array of amplitude of rising with size of signal
    risingH = np.zeros(len(Signal))
    # array of amplitude of falling with size of signal
    dropH = np.zeros(len(Signal))
    # array of amplitude of rising with size pks size
    Rise = np.array([])
    # array of amplitude of falling with size pks size
    Fall = np.array([])

    for i in range(0, len(val) - 1):
        # piece of signal between two successive valleys
        wind = Signal[val[i]:val[i + 1]]
        # find peak between two minimums
        pk = detect_peaks(wind, mph=0.1 * max(Signal))

        # if peak is found:
        if (len(pk) > 0):
            # append peak position
            pks.append(val[i] + pk)
            # calculate rising amplitude
            risingH[val[i]:val[i + 1]] = [wind[pk] - Signal[val[i]] for a in range(val[i], val[i + 1])]
            Rise = np.append(Rise, (wind[pk] - Signal[val[i]]) > sup1)
            # calculate dropping amplitude
            dropH[val[i]:val[i + 1]] = [wind[pk] - Signal[val[i + 1]] for a in range(val[i], val[i + 1])]
            Fall = np.append(Fall, (wind[pk] - Signal[val[i + 1]]) > sup2)

    risingH = np.array(risingH > sup1).astype(int)
    dropH = np.array(risingH > sup2).astype(int)
    Rise = Rise.astype(int)
    Fall = Fall.astype(int)
    return risingH, dropH, Rise, Fall, pks

def powerFunction(x):
    PWR = np.sum(np.power(x, 2))
    return PWR

def powerArray(x, fs, window_len=50, window='hanning'):
    output = np.zeros(len(x))

    win = eval('np.' + window + '(window_len)')

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x

    WinRange = int(window_len / 2)

    sig = np.r_[x[WinRange:0:-1], x, x[-1:len(x) - WinRange:-1]]

    for i in range(int(WinRange), len(sig) - int(WinRange)):
        output[i-int(WinRange)] = powerFunction(sig[i - WinRange:WinRange + i]*win)

    return output
