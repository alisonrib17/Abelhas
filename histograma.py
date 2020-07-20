import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/home/alison/Documentos/Projeto/datasets/dataset_mfcc.csv', sep=',')
especies = list(data['label'])
num_bins = 18

fig, ax = plt.subplots()

n, bins, patches = ax.hist(especies, num_bins, orientation='horizontal')

#plt.xticks(rotation = 30)
plt.show()