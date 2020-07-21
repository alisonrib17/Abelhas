import os
import csv
import pickle
import librosa
import pathlib
import pandas as pd
import numpy as np

N_FFT = 1024         # Número de posições na frequência para Fast Fourier Transform
HOP_SIZE = 1024      # Número de quadros de áudio entre colunas STFT
SR = 44100           # Frequência de amostragem
N_MELS = 40          # Parâmetros de filtros Mel   
WIN_SIZE = 1024      # Número de amostras em cada janela STFT
WINDOW_TYPE = 'hann' # The windowin function
FEATURE = 'mel'      # Feature representation
FMIN = 1400
N_MFCC = 41			 # Número de features


def extrai_features():
	header = 'filename'
	for i in range(1, N_MFCC):
		header += f' mfcc{i}'
	
	header += ' BeginTime'
	header += ' EndTime'
	header += ' Annotation'
	header += ' label'
	header = header.split()

	file = open('/home/alison/Documentos/Projeto/datasets/dataset_mfcc.csv', 'w', newline='')
	with file:
		writer = csv.writer(file)
		writer.writerow(header)

	especies = ['Auglochloropsis_bradiocephalis', 'Augchloropsis_sp1', 'Auglochloropsis_sp1', 'Auglochloropsis_sp2', 'Pseudoalglochloropsis_graminea',
 		'Bombus_morio', 'Bombus_atractus', 'Centris_trigonoides', 'Melipona_quadrifasciata', 'Melipona_bicolor', 'Xylocopa_suspecta',
		'Xylocopa_nigrocincta', 'Exomalopsis_analis', 'Exomalopsis_minor', 'Centris_fuscata', 'Centris_tarsata', 'Eulaema_nigrita', 'Exomalopis_analis']

	for g in especies:
		for filename in os.listdir(f'/home/alison/Documentos/Projeto/especies/{g}'):			
			songname = f'/home/alison/Documentos/Projeto/especies/{g}/{filename}'
			y, sr = librosa.load(songname, mono=True)

			table_name = os.path.splitext(filename)[0] + ".txt"
			table = f'/home/alison/Documentos/Projeto/TabelasAudiosSeparados/{g}/{table_name}'
			table = pd.read_table(table, sep='\t')
			size = int(table.shape[0])

			start_time = table['Begin Time (s)']
			end_time  = table['End Time (s)']
			annotation = table['Annotation']

			for i in range(size):
				to_append = f'{filename}'
				start = float(start_time[i])
				end = float(end_time[i])

				start_index = librosa.time_to_samples(start)
				end_index = librosa.time_to_samples(end)

				required_slice = y[start_index:end_index]

				required_mfcc = librosa.feature.mfcc(y=required_slice, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_SIZE, n_mels=N_MELS, htk=True, fmin=FMIN, fmax=sr/2.0)

				for e in required_mfcc:
					to_append += f' {np.mean(e)}'
				
				to_append += f' {start}'
				to_append += f' {end}'
				to_append += f' {annotation[i]}'
				to_append += f' {g}'

				file = open('/home/alison/Documentos/Projeto/datasets/dataset_mfcc.csv', 'a', newline='')
				with file:
					writer = csv.writer(file)
					writer.writerow(to_append.split())

if __name__ == '__main__':
	extrai_features()