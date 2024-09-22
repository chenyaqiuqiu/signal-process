import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

from signals import readCwru

def interval_num_count(data, low, high):
    '''
    fun: 统计一维数据data落入某一个区间[low, high]内的数量
    param low: 区间下限
    param high: 区间上限
    return count_num: 落入某一个区间[low, high]内的数量
    '''
    count_num = 0
    for i in range(len(data)):
        if data[i]>low and data[i]<high:
            count_num += 1
    return count_num

# 计算概率密度 按照N进行分割
def plt_amp_prob_density_fun(input, n):
    '''
    fun: 绘制幅值概率密度函数
    param data: 输入数据，1维array
    param n: 分割成段数的数量
    return: 绘制幅值概率密度函数
    '''
    max_value = np.abs( input[np.argmax( np.abs(input) )] ) #
    count_num = []
    for i in range(n):
        interval = max_value*2/n              # 区间长度为interval_len
        low = -max_value + i*interval         # 区间下限
        high = -max_value + (i+1)*interval    # 区间上限
        count = interval_num_count(data=input, low=low, high=high)  # 统计落入该区间的幅值个数
        count_num.append(count)
    plt.bar(x=range( len(count_num) ), height=count_num)  # 绘制柱状图
    plt.show()

# 包络谱函数
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

def plt_envelope_spectrum(data, fs, xlim=None, vline=None):
    '''
    fun: 绘制包络谱图
    param data: 输入数据，1维array
    param fs: 采样频率
    param xlim: 图片横坐标xlim，default = None  横坐标采样长度
    param vline: 图片垂直线，default = None  故障特征频率
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
    plt.show()

if __name__ == '__main__':
    data = readCwru.read_outerBall()
    plt_envelope_spectrum(data = data, fs=12000, xlim=300, vline=None)

    data = readCwru.read_outerBall()
    plt_envelope_spectrum(data = data, fs=12000, xlim=300, vline=None)
