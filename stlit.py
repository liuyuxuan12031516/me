# import librosa
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import datetime
import random
from python_speech_features import mfcc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import numpy
import scipy.io.wavfile
from matplotlib import pyplot as plt
from scipy.fftpack import dct
from scipy.fft import fft

st.set_page_config(page_title="言语语音可视化平台", page_icon=":rainbow:", layout="wide", initial_sidebar_state="auto")
st.title('言语语音可视化平台:heart:')
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True
else:
    st.session_state.first_visit = False
# 初始化全局配置
if st.session_state.first_visit:
    st.session_state.date_time = datetime.datetime.now() + datetime.timedelta(
        hours=8)  # Streamlit Cloud的时区是UTC，加8小时即北京时间
    st.balloons()
    st.write('Start time:', st.session_state.date_time.date(), st.session_state.date_time.time())
# 	st.snow()

option = st.sidebar.selectbox(
        'type',
        ['xlsx','xls','wav','mp3'])
uploaded_file = st.file_uploader("选择待上传的文件", accept_multiple_files=False, type=[option])


if uploaded_file is not None:
    if option == 'wav':

        st.write(uploaded_file.type)
        sample_rate, signal = scipy.io.wavfile.read(uploaded_file)


        'file:', uploaded_file
        option2 = st.selectbox(
            'option',
            ['time domain', 'mfcc', 'frequency domain','specgram'])
        'option:', option2

        if option2 == 'time domain':
            fig1 = plt.figure(figsize=(18, 8))
            plt.plot(signal)
            plt.title(option2)
            st.pyplot(fig1)

        elif option2 == 'mfcc':
            feature_mfcc = mfcc(signal, samplerate=sample_rate, nfft=1103, winfunc=np.hamming)
            fig2 = plt.figure(figsize=(18, 8))
            mfcc_data = np.swapaxes(feature_mfcc, 0, 1)
            plt.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
            plt.title(option2)
            st.pyplot(fig2)

        elif option2 == 'frequency domain':
            fig3 = plt.figure(figsize=(18, 8))
            x, sr = signal, sample_rate
            print(len(x))
            ft = fft(x)
            print(len(ft), type(ft), np.max(ft), np.min(ft))
            magnitude = np.absolute(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
            frequency = np.linspace(0, sr, len(magnitude))  # (0, 16000, 121632)
            # plot spectrum，限定[:40000]
            # plt.figure()
            plt.plot(frequency, magnitude)  # magnitude spectrum
            plt.title(option)
            plt.xlabel("Hz")
            plt.ylabel("Magnitude")
            st.pyplot(fig3)

        elif option2 == 'specgram':
            fig3 = plt.figure(figsize=(18, 8))
            framelength = 0.025
            fs = sample_rate
            # NFFT点数=0.025*fs
            framesize = int(framelength * fs)
            print("NFFT:", framesize)
            plt.specgram(signal, NFFT=framesize, Fs=fs, window=np.hanning(M=framesize))
            plt.ylabel('Frequency')
            plt.xlabel('Time(s)')
            plt.title('Spectrogram')
            st.pyplot(fig3)

            # fftdata = np.fft.fft(waveData[0, :])
            # fftdata = abs(fftdata)
            # hz_axis = np.arange(0, len(fftdata))
            # plt.figure()
            # plt.plot(hz_axis, fftdata, c='b')
            # plt.xlabel('hz')
            # plt.ylabel('am')
            # plt.show()




    else:
        df = pd.read_excel(uploaded_file.read())
        st.dataframe(df)

# uploaded_file1 = st.file_uploader("Choose a WAV file", type="wav")
# uploaded_file2 = st.file_uploader("Choose a WAV file", type="wav")

# if uploaded_file1 is not None:
#     sample_rate,signal=scipy.io.wavfile.read(uploaded_file)

# # file = st.sidebar.selectbox(
# #         'file',
# #         ['D:/桌面/CGZ-speaker1-a-T1.wav','D:/桌面/ZQ-speaker1-a-T4.wav'])
