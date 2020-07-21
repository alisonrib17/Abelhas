import librosa, librosa.display, IPython.display as ipd
import pickle
import pandas as pd
import numpy as np
import os
import csv
from PIL import Image
import pathlib
import csv
import matplotlib.pyplot as plt
import IPython
from scipy.io import wavfile
import scipy.signal
import noisereduce as nr

plt.rcParams['figure.figsize'] = (14,5)

N_FFT = 1024         # Número de posições na frequência para Fast Fourier Transform
HOP_SIZE = 1024      # Número de quadros de áudio entre colunas STFT
SR = 44100           # Frequência de amostragem
N_MELS = 40          # Parâmetros de filtros Mel   
WIN_SIZE = 1024      # Número de amostras em cada janela STFT
WINDOW_TYPE = 'hann' # The windowin function
FEATURE = 'mel'      # Feature representation
FMIN = 1400
N_MFC = 128			 # Número de features


audio = f'/home/alison/Documentos/Projeto/especies/Melipona_bicolor/9.WAV'
tabela = f'/home/alison/Documentos/Projeto/TabelasAudiosSeparados/Melipona_bicolor/9.txt'

tabela = pd.read_table(tabela, sep='\t')
print(tabela.shape[0])

bee, sr = librosa.load(audio, mono=True)

#plt.subplot(1,1,1)
#librosa.display.waveplot(bee, sr=sr)
#plt.show()

start_time = tabela['Begin Time (s)']
end_time  = tabela['End Time (s)']
annotation = tabela['Annotation']
segmentos = []
'''
for i in range(8):
	start = float(start_time[i])
	end = float(end_time[i])

	segmentos.append((start, end, annotation[i]))

	start_index = librosa.time_to_samples(start)
	end_index = librosa.time_to_samples(end)

	required_slice = bee[start_index:end_index]
	#required_mfcc = librosa.feature.mfcc(y=required_slice, sr=sr, n_fft=N_FFT, hop_length=HOP_SIZE, n_mels=N_MELS, htk=True, fmin=FMIN, fmax=sr/2.0)
	D_bee = librosa.stft(required_slice, n_fft=N_FFT, hop_length=HOP_SIZE, window=WINDOW_TYPE, win_length=WIN_SIZE)
	stft_bee = np.abs(D_bee)**2

	required_mfcc = librosa.feature.mfcc(y=None, S=stft_bee, sr=SR, n_fft=N_FFT, hop_length=HOP_SIZE, n_mels=N_MELS, htk=True, fmin=FMIN, fmax=sr/2.0)

	to_append = f'{start}'
	to_append += f' {end}'
	to_append += f' {annotation[i]}'

	for e in required_mfcc:
		to_append += f' {np.mean(e)}'

	file = open('/home/alison/Documentos/Projeto/datasets/dataset_mfcc_test2.csv', 'a', newline='')
	with file:
		writer = csv.writer(file)
		writer.writerow(to_append.split())

	#print(start,end,annotation[i])

'''
#D_bee = librosa.stft(bee, n_fft=N_FFT, hop_length=HOP_SIZE, window=WINDOW_TYPE, win_length=WIN_SIZE) #Cria o espetograma
#stft_bee = np.abs(D_bee)**2

#mfcc_bee = librosa.feature.mfcc(y=None, S=stft_bee, sr=SR, n_fft=N_FFT, hop_length=HOP_SIZE, n_mels=N_MELS, htk=True, fmin=FMIN, fmax=sr/2.0)

#plt.subplot(3,1,1)
#librosa.display.waveplot(bee, sr=SR)
#plt.subplot(3,1,2)
#librosa.display.specshow(librosa.core.amplitude_to_db(stft_bee), sr=SR, hop_length=HOP_SIZE, x_axis='time', y_axis='linear')
#plt.subplot(3,1,3)
#librosa.display.specshow(librosa.core.amplitude_to_db(mfcc_bee, ref=1.0), sr=SR, hop_length=HOP_SIZE, x_axis='time', y_axis='mel')
#plt.show()