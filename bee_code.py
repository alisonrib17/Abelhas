import os
import csv
import pickle
import librosa
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Binarizer
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2


def extrai_features():
	#header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
	header = 'filename'
	for i in range(1, 129):
		header += f' mfcc{i}'
	header += ' label'
	header = header.split()

	file = open('/home/alison/Documentos/Projeto/datasets/dataset_mfcc.csv', 'w', newline='')
	with file:
		writer = csv.writer(file)
		writer.writerow(header)

	especies = ['Auglochloropsis_bradiocephalis', 'Augchloropsis_sp1', 'Auglochloropsis_sp1', 'Auglochloropsis_sp2', 'Pseudoalglochloropsis_graminea',
 		'Bombus_morio', 'Bombus_atractus', 'Centris_trigonoides', 'Melipona_quadrifasciata', 'Melipona_bicolor', 'Xylocopa_suspecta',
		'Xylocopa_nigrocincta', 'Exomalopsis_analis', 'Exomalopsis_minor']

	for g in especies:
		for filename in os.listdir(f'/home/alison/Documentos/Projeto/especies/{g}'):
			songname = f'/home/alison/Documentos/Projeto/especies/{g}/{filename}'
			y, sr = librosa.load(songname, duration=10)
			to_append = f'{filename}'
			#rmse = librosa.feature.rmse(y=y)
			#chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
			#spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
			#spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
			#rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
			#zcr = librosa.feature.zero_crossing_rate(y)
			mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
			#melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_SIZE, n_mels=N_MELS, htk=True, fmin=FMIN, fmax=sr/2)

			#to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'

			for e in mfcc:
				to_append += f' {np.mean(e)}'
			to_append += f' {g}'
			file = open('/home/alison/Documentos/Projeto/datasets/dataset_mfcc.csv', 'a', newline='')
			with file:
				writer = csv.writer(file)
				writer.writerow(to_append.split())

def salva_modelo(modelo, nome_arq):
	filename = '/home/alison/Documentos/Projeto/modelos/' + nome_arq
	pickle.dump(modelo, open(filename, 'wb'))

def algoritmos(op, matriz, classes):
	if op == "svm":
		pass

	return prediction

def preprocessamento():
	data = pd.read_csv('/home/alison/Documentos/Projeto/datasets/dataset_mfcc.csv', sep=',')
	#data.head()
	data = data.drop(['filename'],axis=1)

	especies_list = data.iloc[:, -1]
	encoder = LabelEncoder()
	classes = encoder.fit_transform(especies_list)

	matriz = np.array(data.iloc[:, :-1])

	return matriz, classes

def main():
	#extrai_espectograma()
	#extrai_features()
	matriz, classes = preprocessamento()

	algoritmo = "svm"
	#algoritmo = "lr"
	#algoritmo = "rf"
	#algoritmo = "dtree"
	#algoritmo = "ens"
	#algoritmo = "lstm"
	#algoritmo = "bilstm"
	#algoritmo = "cnn"
	pred, y_test = algoritmos(algoritmo, matriz, classes)

	print("Acurácia...: %.4f" %(metrics.accuracy_score(y_test, pred) * 100))
	print("Precision..: %.4f" %(metrics.precision_score(y_test, pred, average='macro') * 100))
	print("Recall.....: %.4f" %(metrics.recall_score(y_test, pred, average='macro') * 100))
	print("F1-Score...: %.4f" %(metrics.f1_score(y_test, pred, average='macro') * 100))
	print()
	#print(metrics.classification_report(y_test, pred, especies,digits=4))
	#print(pd.crosstab(y_test, pred, rownames=['True'], colnames=['Predicted'], margins=True))

if __name__ == '__main__':
	main()