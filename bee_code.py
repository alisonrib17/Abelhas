import os
import csv
import sys
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
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
import warnings
warnings.filterwarnings('ignore')

def salva_modelo(modelo, nome_arq):
	filename = '/home/alison/Documentos/Projeto/modelos/' + nome_arq
	pickle.dump(modelo, open(filename, 'wb'))

def algoritmos(op, matriz, classes, zumbido):
	if op == "svm":
		tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-3,1e-4], 'C': [0.001, 0.1, 0.01, 1, 10], 'decision_function_shape': ['ovo']},
							{'kernel': ['poly'], 'gamma': [1e-2,1e-3,1e-4], 'C': [0.001, 0.1, 0.01, 1, 10], 'decision_function_shape': ['ovo']},
							{'kernel': ['sigmoid'], 'gamma': [1e-2,1e-3,1e-4], 'C': [0.001, 0.1, 0.01, 1, 10], 'decision_function_shape': ['ovo']}, 
							{'kernel': ['linear'], 'C': [0.001, 0.1, 0.01, 1, 10], 'decision_function_shape': ['ovo']}]

		scores = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

		for score in scores:
			modelo = GridSearchCV(SVC(), tuned_parameters, scoring=score)
			
			kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

			prediction = cross_val_predict(modelo, matriz, classes, cv=kfold)

		modelname = "modelo_svm_" + zumbido + ".sav"
		salva_modelo(modelo, modelname)

	elif op == "lr":
		tuned_parameters = [{'penalty': ['l1', 'l2']},
							{'C': [0.001, 0.1, 1, 10, 100]}]

		scores = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

		for score in scores:
			modelo = GridSearchCV(LogisticRegression(), tuned_parameters, scoring=score)

			kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

			prediction = cross_val_predict(modelo, matriz, classes, cv=kfold)

		modelname = "modelo_lr_" + zumbido + ".sav"
		salva_modelo(modelo, modelname)

	elif op == "dtree":
		tuned_parameters = {"criterion": ["gini", "entropy"],
							"min_samples_split": [2, 10],
							"max_depth": [2, 5, 10]
							}

		scores = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

		for score in scores:
			modelo = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, scoring=score)

			kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

			prediction = cross_val_predict(modelo, matriz, classes, cv=kfold)

		modelname = "modelo_dtree_" + zumbido + ".sav"
		salva_modelo(modelo, modelname)

	elif op == "rf":
		tuned_parameters = {'n_estimators': [100, 200],
							'max_features': ['auto', 'sqrt', 'log2']}
		
		scores = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

		for score in scores:
			modelo = GridSearchCV(RandomForestClassifier(), tuned_parameters, scoring=score)

			kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

			prediction = cross_val_predict(modelo, matriz, classes, cv=kfold)

		modelname = "modelo_rf_" + zumbido + ".sav"
		salva_modelo(modelo, modelname)

	elif op == "ens":
		tuned_parameters = {'lr__C': [0.001, 0.1, 1, 10, 100],
							'svc__C': [0.001, 0.1, 0.01, 1, 10]}

		svc = SVC()
		rf = RandomForestClassifier()
		lr = LogisticRegression()

		modelos = [('svc', svc), ('rf', rf), ('lr', lr)]

		votingclf = VotingClassifier(estimators=modelos, voting='hard')

		scores = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

		for score in scores:
			modelo = GridSearchCV(votingclf, tuned_parameters, scoring=score)

			kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

			prediction = cross_val_predict(modelo, matriz, classes, cv=kfold)


		modelname = "modelo_ensemble_" + zumbido + ".sav"
		salva_modelo(modelo, modelname)

	elif op == "lstm":
		pass
	elif op == "bilstm":
		pass
	elif op == "cnn":
		pass        
	else:
		print("Opção errada!")

	return prediction

def read_dataset(zumbido):
	data = pd.read_csv('/home/alison/Documentos/Projeto/datasets/dataset_mfcc.csv', sep=',')
	
	if zumbido == "voo":
		data = data[data['Annotation'] == 'voo']
	if zumbido == "flor":
		data = data[data['Annotation'] != 'voo']
	
	data = data.drop(['filename', 'Annotation'],axis=1)

	especies_list = data.iloc[:, -1]
	encoder = LabelEncoder()
	classes = encoder.fit_transform(especies_list)
	standard = StandardScaler()
	matriz = standard.fit_transform(np.array(data.iloc[:, :-1]))

	return matriz, classes

def main(algoritmo, zumbido):
	matriz, classes = read_dataset(zumbido)
	
	pred = algoritmos(algoritmo, matriz, classes, zumbido)

	print("Acurácia...: %.4f" %(metrics.accuracy_score(classes, pred) * 100))
	print("Precision..: %.4f" %(metrics.precision_score(classes, pred, average='macro') * 100))
	print("Recall.....: %.4f" %(metrics.recall_score(classes, pred, average='macro') * 100))
	print("F1-Score...: %.4f" %(metrics.f1_score(classes, pred, average='macro') * 100))
	print()
	#print(metrics.classification_report(classes, pred, especies,digits=4))
	#print(pd.crosstab(classes, pred, rownames=['True'], colnames=['Predicted'], margins=True))

if __name__ == '__main__':

	args = sys.argv[1:]

	if len(args) >= 2:

		algoritmo = args[0]
		zumbido = args[1]
		main(algoritmo, zumbido)

	else:
		sys.exit('''
			Parâmetros incoerentes!
		''')