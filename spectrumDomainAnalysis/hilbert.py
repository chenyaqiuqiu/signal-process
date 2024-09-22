'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks

# 生成信号
fs = 1000  # 采样频率
t = np.linspace(0, 1, fs)  # 时间向量
signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t)  # 衰减正弦波信号

# 计算解析信号
analytic_signal = hilbert(signal)
upper_envelope = np.abs(analytic_signal)  # 上包络线

# 计算下包络线
# 找到峰值位置
peaks, _ = find_peaks(signal)
troughs, _ = find_peaks(-signal)  # 寻找信号的下波谷

# 生成下包络线
lower_envelope = np.interp(t, t[peaks], signal[peaks])  # 线性插值上包络线
lower_envelope = np.interp(t, t[troughs], signal[troughs])  # 线性插值下包络线

# 绘制信号和包络线
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label='Signal', color='blue')
plt.plot(t, upper_envelope, label='Upper Envelope', color='red', linestyle='--')
plt.plot(t, lower_envelope, label='Lower Envelope', color='green', linestyle='--')
plt.title('Signal and its Upper and Lower Envelopes')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# 生成随机信号
np.random.seed(0)
t = np.linspace(0, 10, 1000)
signal = np.random.normal(0, 1, t.shape) * np.sin(2 * np.pi * 0.5 * t)

# 计算希尔伯特变换
analytic_signal = hilbert(signal)
envelope = np.abs(analytic_signal)

# 绘制随机信号和包络线
plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Random Signal')
plt.plot(t, envelope, label='Envelope', color='red')
plt.title('Random Signal and its Envelope using Hilbert Transform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()