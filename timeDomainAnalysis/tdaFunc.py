import numpy as np
import pandas as pd

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
    peak_value = signal.abs().max()
    peak_factor = peak_value / rms
    return peak_factor

# 波形因子
def form_factor(signal):
    # 计算均方根（RMS）
    rms = np.sqrt(np.mean(signal**2))
    # 计算平均绝对值
    mean_absolute_value = np.mean(signal.abs())
    # 计算波形因子
    form_factor = rms / mean_absolute_value
    return form_factor

# 脉冲因子
def impulse_factor(signal):
    # 计算均方根（RMS）
    rms = np.sqrt(np.mean(signal**2))
    # 计算最大绝对值
    max_absolute_value = signal.abs().max()
    # 计算脉冲因子
    impulse_factor = max_absolute_value / rms
    return impulse_factor

# 裕度因子
def crest_factor(signal):
    # 计算均方根（RMS）
    rms = np.sqrt(np.mean(signal**2))
    # 计算最大绝对值
    max_absolute_value = signal.abs().max()
    # 计算裕度因子
    crest_factor = max_absolute_value / rms
    return crest_factor

# 峭度因子
def kurt_factor(signal):
    data = pd.Series(signal)
    kurt_factor = data.kurt() / signal.std()
    return kurt_factor

'''
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
print("form_factor: ", form_factor(signals))
print("impulse_factor: ", impulse_factor(signals))
print("crest_factor: ", crest_factor(signals))
'''

