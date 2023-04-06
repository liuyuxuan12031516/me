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
import datetime
from sklearn import neighbors
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import time
import numpy as np
import matplotlib.pyplot as plt

class_le = LabelEncoder()
knn = neighbors.KNeighborsClassifier()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
gnb = GaussianNB()
svc = svm.SVC()
lg = LogisticRegression()

model_list = ['knn','dtc','gnb','lg','svc','rfc']
model_dict = {}
s = 0
for i in ['K近鄰', '決策樹', '樸素貝葉斯', '邏輯回歸','支持向量機','隨機森林']:
    model_dict[i] = model_list[s]
    s+=1

def model_create(i):
    return eval(model_dict[i])

def model_run(model,trainX,trainY):
    start = time.time()
    model = model.fit(trainX, trainY)  # 用训练集数据训练模型
    return model.score(trainX, trainY),time.time() - start

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
    ['xlsx_machine_learning','xlsx', 'xls', 'wav', 'mp3'])
if option == 'xlsx_machine_learning':
    uploaded_file = st.file_uploader("选择待上传的文件", accept_multiple_files=False, type=['xlsx'])
else:
    uploaded_file = st.file_uploader("选择待上传的文件", accept_multiple_files=False, type=[option])

def load_data():
    return pd.read_excel(uploaded_file.read())

introduction = {}
introduction['K近鄰'] = 'KNN(K-Nearest Neighbor)是最简单的机器学习算法之一，可以用于分类和回归，是一种监督学习算法。它的思路是这样，如果一个样本在特征空间中的K个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。也就是说，该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。'
introduction['決策樹'] = '决策树（Decision Tree）是在已知各种情况发生概率的基础上，通过构成决策树来求取净现值的期望值大于等于零的概率，评价项目风险，判断其可行性的决策分析方法，是直观运用概率分析的一种图解法。 由于这种决策分支画成图形很像一棵树的枝干，故称决策树。'
introduction['樸素貝葉斯'] = '朴素贝叶斯分类器，在机器学习中是一系列以假设特征之间强（朴素）独立下运用贝叶斯定理为基础的简单概率分类器。'
introduction['邏輯回歸'] = 'Logistic Regression 虽然被称为回归，但其实际上是分类模型，并常用于二分类。Logistic Regression 因其简单、可并行化、可解释强深受工业界喜爱。Logistic 回归的本质是：假设数据服从这个分布，然后使用极大似然估计做参数的估计。'
introduction['支持向量機'] = '支持向量机(SVM)是一项功能强大的分类和回归技术，可最大化模型的预测准确度，而不会过度拟合训练数据。 SVM特别适用于分析预测变量字段非常多的数据。 SVM的工作原理是将数据映射到高维特征空间，这样即使数据不是线性可分，也可以对该数据点进行分类。'
introduction['隨機森林'] = '随机森林是一种集成算法（Ensemble Learning），它属于Bagging类型，通过组合多个弱分类器，最终结果通过投票或取均值，使得整体模型的结果具有较高的精确度和泛化性能。 其可以取得不错成绩，主要归功于“随机”和“森林”，一个使它具有抗过拟合能力，一个使它更加精准。'

if option == 'xlsx_machine_learning':
    variable = st.multiselect(
        'Variable',
        ['f0', 'f1', 'f2', 'f3'])
    target = st.multiselect(
        'Target',
        ['speaker', 'vowel', 'community', 'region', 'sex', 'age'])
    models = st.multiselect(
        'Model',
        ['K近鄰', '決策樹', '樸素貝葉斯', '邏輯回歸', '支持向量機', '隨機森林'])

if uploaded_file is not None:
    if (option == 'wav')|(option == 'mp3'):

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

    elif (option == 'xlsx')|(option == 'xls'):
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

    elif option == 'xlsx_machine_learning':
        data = None
        if st.button('加载数据以及运行模型'):
            data = load_data()
            st.dataframe(data.head())

            if len(variable)!=0:
                feature_list = []
                for v in variable:
                    if feature_list is None:
                        feature_list = [i for i in data.columns if v in i]
                    else:
                        feature_list += [i for i in data.columns if v in i]
                st.write('特征列：')
                st.text(feature_list)
                feature = data[feature_list]

            if len(models)!=0:
                st.write('模型選擇：')
                # st.text(model)
                for i in models:
                    st.text(introduction[i])

            if len(target)!=0:
                label = data[target]
                st.write('分類數：')
                # st.dataframe(label)
                st.text(label.nunique())
                label = class_le.fit_transform(label.values)

            # if st.button('運行模型：'):
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)
            num = 1
            score_list = []
            delta_list = []
            for i in models:
                model = model_create(i)
                score,delta = model_run(model,feature.values,label)
                score_list.append(round(score * 100, 2))
                delta_list.append(round(delta, 2))
                with eval('col'+str(num)):
                    num+=1
                    # st.write(i)
                    st.markdown(str(i)+'(%)')
                    st.metric(label='準確率'+'(%)',value=round(score*100,2))
                    # st.text('消耗的時間:')
                    st.metric(label='消耗的時間:(s)',value=round(delta,2))

            score_list = np.array(score_list)
            delta_list = np.array(delta_list)
            fig = plt.figure(figsize=(5, 2))
            plt.plot(score_list)
            plt.xticks(ticks=[i for i in range(len(models))],labels=[model_dict[i] for i in models])
            # plt.legend(loc="upper right")
            plt.title('Model Score Comparison')
            plt.xlabel('Model')
            plt.ylabel('Accuracy')
            plt.show()
            st.pyplot(fig)
