import os
import librosa
import pathlib
import matplotlib.pyplot as plt

#Extraindo espectograma de cada áudio
def extrai_espectograma():
	cmap = plt.get_cmap('inferno')
	plt.figure(figsize=(8,8))

	especies = ['Auglochloropsis_bradiocephalis', 'Augchloropsis_sp1', 'Auglochloropsis_sp1', 'Auglochloropsis_sp2', 'Pseudoalglochloropsis_graminea',
 	'Bombus_morio', 'Bombus_atractus', 'Centris_trigonoides', 'Melipona_quadrifasciata', 'Melipona_bicolor', 'Xylocopa_suspecta',
 	'Xylocopa_nigrocincta', 'Exomalopsis_analis', 'Exomalopsis_minor']

	for g in especies:
		pathlib.Path(f'/home/alison/Documentos/Projeto/img_data/{g}').mkdir(parents=True, exist_ok=True)
		for filename in os.listdir(f'/home/alison/Documentos/Projeto/especies/{g}'):
			dataset = f'/home/alison/Documentos/Projeto/especies/{g}/{filename}'
			y, sr = librosa.load(dataset, mono=True, duration=5)
			plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
			plt.axis('off');
			plt.savefig(f'/home/alison/Documentos/Projeto/img_data/{g}/{filename[:-3].replace(".", "")}.png')
			plt.clf()

if __name__ == '__main__':
	extrai_espectograma()