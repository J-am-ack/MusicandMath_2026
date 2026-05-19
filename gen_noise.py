import numpy as np
from scipy.io import wavfile
from scipy import signal

sample_rate = 44100
duration = 10
n = sample_rate * duration
white = np.random.randn(n)          # 1. 先生成白噪声

# 粉色噪声：一次累积和 + 高通去直流
pink = signal.sosfilt(
    signal.butter(4, 20, btype='high', fs=sample_rate, output='sos'),
    np.cumsum(white)
)
pink = pink / np.max(np.abs(pink)) * 0.9

# 棕色噪声：两次累积和 + 高通去直流（低频更猛）
brown = signal.sosfilt(
    signal.butter(4, 20, btype='high', fs=sample_rate, output='sos'),
    np.cumsum(np.cumsum(white))
)
brown = brown / np.max(np.abs(brown)) * 0.9

# 保存
wavfile.write('src/pink.wav', sample_rate, (pink * 32767).astype(np.int16))
wavfile.write('src/brown.wav', sample_rate, (brown * 32767).astype(np.int16))