import librosa
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
import seaborn as sns

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

        audio_file = uploaded_file
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')

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

    else:
        df = pd.read_excel(uploaded_file.read())
        st.dataframe(df.head())
        option3 = st.multiselect(
            'Variable',
            ['f0','f1','f2','f3'])
        # st.write(option3)

        times = [13, 25, 38, 50, 62, 75, 88]
        time_dict = {}
        s = 0
        for i in times:
            time_dict[i] = s
            s += 1
        # st.write(time_dict)

        if len(option3) >= 1:
            col1,col2,col3 = st.columns((1,1,4))

            time0 = col1.checkbox('13')
            time1 = col1.checkbox('25')
            time2 = col1.checkbox('38')
            time3 = col1.checkbox('50')
            time4 = col1.checkbox('62')
            time5 = col1.checkbox('75')
            time6 = col1.checkbox('88')
            time = [time0,time1,time2,time3,time4,time5,time6]

            for i in option3:
                columns = [j for j in df.columns.values if i in j]

            col_list = []
            for i in range(7):
                if time[i]:
                    col_list.append(columns[time_dict[times[i]]])


            s = col2.multiselect('group:',['vowel','region','sex','age'])
            if len(s) != 0:

                col2.write(s)
                grouplist = [i for i in s]
                dff = df.groupby(by=grouplist).mean()
                dff = dff.loc[:,col_list]

                col3.dataframe(dff)

                fig = plt.figure(figsize=(18,8))
                for i in range(len(dff)):
                    plt.plot(dff.loc[dff.index[i]], label=dff.index[i])
                plt.legend(loc="upper right")
                plt.show()
                st.pyplot(fig)




