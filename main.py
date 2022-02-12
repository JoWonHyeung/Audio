import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn import mixture
from librosa.feature.spectral import mfcc


# 1. 경로 설정
audio_path = "C:\\Users\\user0425\\Desktop\\DigitalSound\\"

file1 = "M1.wav"
file2 = "M2.wav"
file3 = "F1.wav"
file4 = "F2.wav"

test_file = "TEST.wav" ################# 이쪽 부분에 TEST FILE명을 적어주시면 됩니다. #########################

y1, sr1 = librosa.load(audio_path + file1, sr = 16000)
y2, sr2 = librosa.load(audio_path + file2, sr = 16000)
y3, sr3 = librosa.load(audio_path + file3, sr = 16000)
y4, sr4 = librosa.load(audio_path + file4, sr = 16000)

# 2. MFCC 추출

mfcc1 = librosa.feature.mfcc(y = y1, sr = sr1, n_mfcc=24, n_fft = 512)
mfcc2 = librosa.feature.mfcc(y = y2, sr = sr2, n_mfcc=24, n_fft = 512)
mfcc3 = librosa.feature.mfcc(y = y3, sr = sr3, n_mfcc=24, n_fft = 512)
mfcc4 = librosa.feature.mfcc(y = y4, sr = sr4, n_mfcc=24, n_fft = 512)


#3. GMM Model 구축

gmm1 = mixture.GaussianMixture(n_components=5, covariance_type='full')
gmm1.fit(mfcc1.T)

gmm2 = mixture.GaussianMixture(n_components=5, covariance_type='full')
gmm2.fit(mfcc2.T)

gmm3 = mixture.GaussianMixture(n_components=5, covariance_type='full')
gmm3.fit(mfcc3.T)

gmm4 = mixture.GaussianMixture(n_components=5, covariance_type='full')
gmm4.fit(mfcc4.T)

#4. Test case의 MFCC 추출
test_y, test_sr = librosa.load(audio_path + test_file, sr = 16000)

test_mfcc = librosa.feature.mfcc(y = test_y, sr = test_sr, n_mfcc=24, n_fft = 512)

#5. 점수 측정
predict1 = gmm1.score(test_mfcc.T)
predict2 = gmm2.score(test_mfcc.T)
predict3 = gmm3.score(test_mfcc.T)
predict4 = gmm4.score(test_mfcc.T)

predicts = [predict1 , predict2, predict3, predict4]
target = ['M1','M2','F1','F2']

max = -sys.maxsize
max_index = ""
target_res = ""
i = 0

# 6. 확률이 가장 큰 값을 추출하고 해당하는 target을 text파일에 저장할 target을 저장한다
for p in predicts:
    if max < p:
        max = p
        target_res = target[i]
    i += 1
    
f = open(audio_path + "result.txt","a")

f.write(target_res + "\n")

f.close()
    


