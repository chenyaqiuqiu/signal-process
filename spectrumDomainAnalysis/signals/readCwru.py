
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fftpack, stats

userFilePath='/Users/ma/phmCodes/signal-process/spectrumDomainAnalysis/signals/cwru-bearing-data/'
def data_acquision(FilePath):
    """
    fun: 从cwru mat文件读取加速度数据
    param file_path: mat文件绝对路径
    return accl_data: 加速度数据，array类型
    """
    data = scio.loadmat(FilePath)  # 加载mat数据
    data_key_list = list(data.keys())  # mat文件为字典类型，获取字典所有的键并转换为list类型
    accl_key = data_key_list[3]  # 获取'X108_DE_time'
    accl_data = data[accl_key].flatten()  # 获取'X108_DE_time'所对应的值，即为振动加速度信号,并将二维数组展成一维数组
    return accl_data

def read_outerRace3():
    file_path=userFilePath + '1730_12k_0.007-OuterRace3.mat'
    return data_acquision(file_path)

def read_innerRace():
    file_path = userFilePath + '1730_12k_0.007-InnerRace.mat'
    return data_acquision(file_path)


def read_outerBall():
    file_path = userFilePath + '1730_12k_0.014-Ball.mat'
    return data_acquision(file_path)

def read_normal():
    file_path = userFilePath + '1730_48k_Normal.mat'
    return data_acquision(file_path)

