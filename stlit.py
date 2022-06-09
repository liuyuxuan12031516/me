import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from python_speech_features import mfcc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import numpy
import scipy.io.wavfile
from matplotlib import pyplot as plt
from scipy.fftpack import dct
from scipy.fft import fft

st.title('言语语音可视化平台')

import streamlit as st
import base64
from pathlib import Path
import tempfile


uploaded_file = st.file_uploader("选择待上传的xlsx文件", accept_multiple_files = False, type=["xlsx","xls"])
if st.button("点击"):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file.read())
        st.dataframe(df)
        


# uploaded_file1 = st.file_uploader("Choose a WAV file", type="wav")
# uploaded_file2 = st.file_uploader("Choose a WAV file", type="wav")

# if uploaded_file1 is not None:
#     sample_rate,signal=scipy.io.wavfile.read(uploaded_file)

# # file = st.sidebar.selectbox(
# #         'file',
# #         ['D:/桌面/CGZ-speaker1-a-T1.wav','D:/桌面/ZQ-speaker1-a-T4.wav'])

# 'file:',uploaded_file1
# option = st.sidebar.selectbox(
#         'option',
#         ['time domain','mfcc','frequency domain'])
# 'option:', option

# if option == 'time domain':
#     fig1 = plt.figure()
#     plt.plot(signal)
#     plt.title(option)
#     st.pyplot(fig1)

# elif option == 'mfcc':
#     feature_mfcc = mfcc(signal, samplerate=sample_rate,nfft=1103,winfunc=np.hamming)
#     fig2 = plt.figure()
#     mfcc_data= np.swapaxes(feature_mfcc, 0 ,1)
#     plt.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
#     plt.title(option)
#     st.pyplot(fig2)

# else:
#     fig3 = plt.figure()
#     x, sr = librosa.load(file, sr=16000)
#     print(len(x))
#     ft = fft(x)
#     print(len(ft), type(ft), np.max(ft), np.min(ft))
#     magnitude = np.absolute(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
#     frequency = np.linspace(0, sr, len(magnitude))  # (0, 16000, 121632)
#     # plot spectrum，限定[:40000]
#     # plt.figure(figsize=(18, 8))
#     plt.plot(frequency[:40000], magnitude[:40000])  # magnitude spectrum
#     plt.title(option)
#     plt.xlabel("Hz")
#     plt.ylabel("Magnitude")
#     st.pyplot(fig3)



