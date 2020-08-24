import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv('/home/alison/Documentos/Projeto/datasets_generos/dataset_mfcc.csv', sep=',')
generos = list(data['label'])
num_bins = 8

lista = []
for genero in generos:
	if genero == 'Augchloropsis':
		lista.append('E1')
	elif genero == 'Bombus':
		lista.append('E2')
	elif genero == 'Centris':
		lista.append('E3')
	elif genero == 'Eulaema':
		lista.append('E4')
	elif genero == 'Exomalopis':
		lista.append('E5')
	elif genero == 'Melipona':
		lista.append("E6")
	elif genero ==  'Pseudoalglochloropsis':
		lista.append('E7')
	elif genero == 'Xylocopa':
		lista.append('E8')

#data = data.drop(['label'],axis=1)
#data['label'] = lista

p = data['label'].hist(bins=num_bins, grid=False, color="#e84d60", label=generos)
plt.xlabel("Gêneros", fontsize=15)
plt.ylabel("Quantidade de Amostras",fontsize=15)
plt.show(p)