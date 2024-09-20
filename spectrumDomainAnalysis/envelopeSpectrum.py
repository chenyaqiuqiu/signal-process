import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

def envelope_spectrum(data, fs):
    '''
    param data: 输入数据，1维array
    param fs: 采样频率
    '''
    #----去直流分量----#
    data = data - np.mean(data)
    #----做希尔伯特变换----#
    xt = data
    ht = fftpack.hilbert(xt)
    at = np.sqrt(xt**2+ht**2)   # 获得解析信号at = sqrt(xt^2 + ht^2)
    am = np.fft.fft(at)         # 对解析信号at做fft变换获得幅值
    am = np.abs(am)             # 对幅值求绝对值（此时的绝对值很大）
    am = am/len(am)*2
    am = am[0: int(len(am)/2)]  # 取正频率幅值
    freq = np.fft.fftfreq(len(at), d=1 / fs)  # 获取fft频率，此时包括正频率和负频率
    freq = freq[0:int(len(freq)/2)]  # 获取正频率

def plt_envelope_spectrum(data, fs, xlim=None, vline= None):
    '''
    fun: 绘制包络谱图
    param data: 输入数据，1维array
    param fs: 采样频率
    param xlim: 图片横坐标xlim，default = None
    param vline: 图片垂直线，default = None
    '''
    #----去直流分量----#
    data = data - np.mean(data)
    #----做希尔伯特变换----#
    xt = data
    ht = fftpack.hilbert(xt)
    at = np.sqrt(xt**2+ht**2)   # 获得解析信号at = sqrt(xt^2 + ht^2)
    am = np.fft.fft(at)         # 对解析信号at做fft变换获得幅值
    am = np.abs(am)             # 对幅值求绝对值（此时的绝对值很大）
    am = am/len(am)*2
    am = am[0: int(len(am)/2)]  # 取正频率幅值
    freq = np.fft.fftfreq(len(at), d=1 / fs)  # 获取fft频率，此时包括正频率和负频率
    freq = freq[0:int(len(freq)/2)]  # 获取正频率

    # 测试绘图
    plt.plot(freq, am)
    if vline:  # 是否绘制垂直线
        plt.vlines(x=vline, ymax=0.2, ymin=0, colors='r')  # 高度y 0-0.2，颜色红色
    if xlim: # 图片横坐标是否设置xlim
        plt.xlim(0, xlim)  
    plt.xlabel('freq(Hz)')    # 横坐标标签
    plt.ylabel('amp(m/s2)')   # 纵坐标标签


if __name__ == '__main__':
    print()
    ## 需要CWRU数据集

