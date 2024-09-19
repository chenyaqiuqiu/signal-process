import numpy as np
import pandas as pd
from scipy.stats import kurtosis

# 均值
def means(signal):
    return signal.mean(axis=0)  # 1.均值

# 方差
def var(signal):
    return signal.var(axis=0)  # 2.方差

# 最大值
def max(signal):
    return signal.max(axis=0)  # 3.最大值

# 最小值
def min(signal):
    return signal.min(axis=0)  # 4.最小值

#标准差
def std(signal):
    return signal.std()

# 偏度
def skew(signal):
    data = pd.Series(signal)
    return data.skew()

# 翘度
def kurt(signal):
    data = pd.Series(signal) 
    return data.kurt()

# 峰峰值
def peak2peak(signal):
    peak_to_peak = signal.max() - signal.min()
    return peak_to_peak

# RMS 均方根
def rms(signal):
    return np.sqrt(np.mean(signal**2))

# 峰值因子
def peak_factor(signal):
    # 计算峰值
    peak_value = np.max(np.abs(signal))
    # 计算均方根值
    rms_value = np.sqrt(np.mean(signal**2))
    # 计算峰值因子
    if rms_value != 0:
        return peak_value / rms_value
    else:
        return np.inf  # 避免除以零
    
# 波形因子
def waveform_factor(signal):
    # 计算均值
    mean_value = np.mean(np.abs(signal))
    # 计算均方根值
    rms_value = np.sqrt(np.mean(signal**2))
    # 计算波形因子
    if rms_value != 0:
        return mean_value / rms_value
    else:
        return np.inf  # 避免除以零
    
# 脉冲因子
def impulse_factor(signal):
    # 计算峰值
    peak_value = np.max(np.abs(signal))
    # 计算平均值
    mean_value = np.mean(np.abs(signal))
    # 计算脉冲因子
    if mean_value != 0:
        return peak_value / mean_value
    else:
        return np.inf  # 避免除以零
    
# 裕度因子
def crest_factor(signal):
    # 计算均方根值
    rms_value = np.sqrt(np.mean(signal**2))
    # 计算峰值
    peak_value = np.max(np.abs(signal))
    # 计算裕度因子
    if peak_value != 0:
        return rms_value / peak_value
    else:
        return 0  # 避免除以零
    
# 峭度因子
def kurtosis_factor(signal):
    # 计算峭度
    return kurtosis(signal, fisher=True)  # fisher=True 返回标准化的峭度，减去3


if __name__ == '__main__':
    Fs = 10000  # 采样频率
    f1 = 390  # 信号频率1
    f2 = 2000  # 信号频率2
    t = np.linspace(0, 1, Fs)  # 生成 1s 的时间序列
    # 给定信号
    signals = 2 * np.sin(2 * np.pi * f1 * t) + 5 * np.sin(2 * np.pi * f2 * t)

    print("means: ", means(signals))
    print("var: ", var(signals))
    print("max: ", max(signals))
    print("min: ", min(signals))
    print("std: ", std(signals))
    print("skew: ", skew(signals))
    print("kurt: ", kurt(signals))
    print("p2p: ", peak2peak(signals))
    print("rms: ", rms(signals))
    print("peak_factor: ", peak_factor(signals))
    print("form_factor: ", waveform_factor(signals))
    print("impulse_factor: ", impulse_factor(signals))
    print("crest_factor: ", crest_factor(signals))
    print("kurtosis_factor: ", kurtosis_factor(signals))


