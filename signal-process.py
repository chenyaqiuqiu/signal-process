from spectrumDomainAnalysis import fft
import numpy as np
import matplotlib.pyplot as plt

def test():
    Fs = 10000  # 采样频率
    f1 = 390  # 信号频率1
    f2 = 2000  # 信号频率2
    t = np.linspace(0, 1, Fs)  # 生成 1s 的时间序列
    # 给定信号
    y = 2 * np.sin(2 * np.pi * f1 * t) + 5 * np.sin(2 * np.pi * f2 * t)
    # 第一步，对没有添加噪声的信号进行FFT，验证分析是否正确
    x, result = fft.FFT(Fs, y)
 
    # 绘图
    fig1 = plt.figure(figsize=(16, 9))
    plt.title('original data')
    plt.plot(t, y)
    plt.xlabel('time/s')
    plt.ylabel('Amplitude')
    plt.grid()
 
    fig2 = plt.figure(figsize=(16, 9))
    plt.title('FFT')
    plt.plot(x, result)
    plt.xlabel('Frequency/Hz')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

test()