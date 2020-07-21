import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv('/home/alison/Documentos/Projeto/datasets/dataset_mfcc.csv', sep=',')
especies = list(data['label'])
num_bins = 18



lista = []
for especie in especies:
	if especie == 'Auglochloropsis_bradiocephalis':
		lista.append('E1')
	elif especie == 'Augchloropsis_sp1':
		lista.append('E2')
	elif especie == 'Auglochloropsis_sp1':
		lista.append('E3')
	elif especie == 'Auglochloropsis_sp2':
		lista.append('E4')
	elif especie == 'Pseudoalglochloropsis_graminea':
		lista.append('E5')
	elif especie == 'Bombus_morio':
		lista.append("E6")
	elif especie ==  'Bombus_atractus':
		lista.append('E7')
	elif especie == 'Centris_trigonoides':
		lista.append('E8')
	elif especie == 'Melipona_quadrifasciata':
		lista.append('E9')
	elif especie == 'Melipona_bicolor':
		lista.append('E10')
	elif especie == 'Xylocopa_suspecta':
		lista.append('E11')
	elif especie ==	'Xylocopa_nigrocincta':
		lista.append('E12')
	elif especie == 'Exomalopsis_analis':
		lista.append('E13')
	elif especie == 'Exomalopsis_minor':
		lista.append('E14')
	elif especie == 'Centris_fuscata':
		lista.append('E15')
	elif especie == 'Centris_tarsata':
		lista.append('E16')
	elif especie == 'Eulaema_nigrita':
		lista.append('E17')
	elif especie == 'Exomalopis_analis':
		lista.append('E18')

data = data.drop(['label'],axis=1)
data['label'] = lista

p = data['label'].hist(bins=18, grid=False, color="#e84d60", label=lista)
plt.xlabel("Espécies", fontsize=15)
plt.ylabel("Quantidade de Amostras",fontsize=15)
plt.show(p)