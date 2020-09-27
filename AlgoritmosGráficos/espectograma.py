#import the pyplot and wavfile modules 

import matplotlib.pyplot as plot
import librosa, librosa.display, IPython.display as ipd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

N_FFT = 1024         # Número de posições na frequência para Fast Fourier Transform
HOP_SIZE = 1024      # Número de quadros de áudio entre colunas STFT
SR = 44100           # Frequência de amostragem
N_MELS = 40          # Parâmetros de filtros Mel   
WIN_SIZE = 1024      # Número de amostras em cada janela STFT
WINDOW_TYPE = 'hann' # The windowin function
FEATURE = 'mel'      # Feature representation
#FMIN = 1400
N_MFCC = 41			 # Número de features


dataset = f'AU_118.WAV'
y, sr = librosa.load(dataset, mono=True)

table = f'AU_118.txt'
table = pd.read_table(table, sep='\t')

start_time = table['Begin Time (s)']
end_time  = table['End Time (s)']
low_freq = table['Low Freq (Hz)']

start = float(start_time[1])
end = float(end_time[1])
FMIN = float(low_freq[1])

start_index = librosa.time_to_samples(start)
end_index = librosa.time_to_samples(end)

required_slice = y[start_index:end_index]

D_bee = librosa.stft(required_slice, n_fft=N_FFT, hop_length=HOP_SIZE, window=WINDOW_TYPE, win_length=WIN_SIZE) #Cria o espetograma
stft_bee = np.abs(D_bee)

mfcc_bee = librosa.feature.mfcc(y=required_slice, S=stft_bee, sr=SR, n_fft=N_FFT, hop_length=HOP_SIZE, n_mels=N_MELS, htk=True, fmin=FMIN, fmax=sr/2.0)

plt.figure(figsize=(12,5))
#plt.figure(1)
#plt.subplot(211)
#librosa.display.waveplot(required_slice, sr=SR)

#plot.ylabel('Amplitude')

#plt.subplot(212)
#librosa.display.specshow(librosa.core.amplitude_to_db(stft_bee), sr=SR, hop_length=HOP_SIZE, x_axis='time', y_axis='linear', cmap='jet')
librosa.display.specshow(librosa.core.amplitude_to_db(mfcc_bee, ref=1.0), sr=SR, hop_length=HOP_SIZE, x_axis='time', y_axis='mel')
#plot.xlabel('Time (s)')
#plot.ylabel('Frequency (Hz)')
plt.colorbar(format='%+2.0f dB')

#plt.savefig("specVooFeature.pdf")
plt.savefig("specFlorFeature.pdf")
#plot.show()