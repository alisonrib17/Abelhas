import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv('/home/alison/Documentos/Projeto/datasets_especies/dataset_mfcc.csv', sep=',')
especies = list(data['label'])
num_bins = 15

lista = []
for especie in especies:
	if especie == 'Auglochloropsis_bradiocephalis':
		lista.append('E1')
	elif especie == 'Auglochloropsis_sp1':
		lista.append('E2')
	elif especie == 'Auglochloropsis_sp2':
		lista.append('E3')
	elif especie == 'Pseudoalglochloropsis_graminea':
		lista.append('E4')
	elif especie == 'Bombus_morio':
		lista.append("E5")
	elif especie ==  'Bombus_pauloensis':
		lista.append('E6')
	elif especie == 'Centris_trigonoides':
		lista.append('E7')
	elif especie == 'Melipona_quadrifasciata':
		lista.append('E8')
	elif especie == 'Melipona_bicolor':
		lista.append('E9')
	elif especie == 'Xylocopa_suspecta':
		lista.append('E10')
	elif especie ==	'Xylocopa_nigrocincta':
		lista.append('E11')
	elif especie == 'Exomalopsis_analis':
		lista.append('E12')
	elif especie == 'Exomalopsis_minor':
		lista.append('E13')
	elif especie == 'Centris_tarsata':
		lista.append('E14')
	elif especie == 'Eulaema_nigrita':
		lista.append('E15')

data = data.drop(['label'],axis=1)
data['label'] = lista

p = data['label'].hist(bins=num_bins, grid=False, color="#e84d60", label=lista)
plt.xlabel("Espécies", fontsize=15)
plt.ylabel("Quantidade de Amostras",fontsize=15)
plt.show(p)