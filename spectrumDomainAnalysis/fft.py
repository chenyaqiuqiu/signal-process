
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt

def FFT(Fs, data):
    """
    对输入信号进行FFT
    :param Fs:  采样频率
    :param data:待FFT的序列
    :return:
    """
    L = len(data)  # 信号长度
    N = np.power(2, np.ceil(np.log2(L)))  # 下一个最近二次幂，也即N个点的FFT
    result = np.abs(fft(x=data, n=int(N))) / L * 2  # N点FFT
    axisFreq = np.arange(int(N / 2)) * Fs / N  # 频率坐标
    result = result[range(int(N / 2))]  # 因为图形对称，所以取一半
    return axisFreq, result

if __name__ == '__main__':
    Fs = 10000  # 采样频率
    f1 = 390  # 信号频率1
    f2 = 2000  # 信号频率2
    t = np.linspace(0, 1, Fs)  # 生成 1s 的时间序列
    # 给定信号
    y = 2 * np.sin(2 * np.pi * f1 * t) + 5 * np.sin(2 * np.pi * f2 * t)
    # 第一步，对没有添加噪声的信号进行FFT，验证分析是否正确
    x, result = FFT(Fs, y)
 
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
