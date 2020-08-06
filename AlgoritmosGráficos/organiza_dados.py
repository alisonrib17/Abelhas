import pandas as pd
import os

genero = ['Augchloropsis', 'Bombus', 'Centris', 'Eulaema', 'Exomalopis', 'Melipona', 'Pseudoalglochloropsi', 'Xylocopa']

especies = ['Auglochloropsis_bradiocephalis', 'Augchloropsis_sp1', 'Auglochloropsis_sp1', 'Auglochloropsis_sp2', 'Pseudoalglochloropsis_graminea',
 		'Bombus_morio', 'Bombus_atractus', 'Centris_trigonoides', 'Melipona_quadrifasciata', 'Melipona_bicolor', 'Xylocopa_suspecta',
		'Xylocopa_nigrocincta', 'Exomalopsis_analis', 'Exomalopsis_minor', 'Centris_fuscata', 'Centris_tarsata', 'Eulaema_nigrita', 'Exomalopis_analis']

def por_genero():
	for g in genero:
		for filename in os.listdir(f'/home/alison/Documentos/Projeto/gênero/{g}'):			
			
			songname = f'/home/alison/Documentos/Projeto/gênero/{g}/{filename}'

			table_name = os.path.splitext(filename)[0] + ".txt"
			table = f'/home/alison/Documentos/Projeto/TabelasAudiosSeparadosGênero2/{g}/{table_name}'
			table = pd.read_table(table, sep='\t')
			
			tabela_geral = f'/home/alison/Documentos/Projeto/Tabela_geral_Actualizada2.xlsx'
			tabela_geral = pd.read_excel(tabela_geral, sep='\t')

			tabela_geral = tabela_geral[tabela_geral['Genero_Abelha'] == g]
			tabela_geral = tabela_geral[tabela_geral['Audio'] == filename]

			table['peso'] = list(tabela_geral['Peso'])
			table['tamanho torax'] = list(tabela_geral['Tamanho torax'])

			end = f'/home/alison/Documentos/Projeto/TabelasAudiosSeparadosGênero2/{g}/{table_name}'
			table.to_csv(end, sep='\t')

def por_especie():
	for e in especies:
		for filename in os.listdir(f'/home/alison/Documentos/Projeto/especies/{e}'):		

			songname = f'/home/alison/Documentos/Projeto/especies/{e}/{filename}'

			table_name = os.path.splitext(filename)[0] + ".txt"
			table = f'/home/alison/Documentos/Projeto/TabelasAudiosSeparados2/{e}/{table_name}'
			table = pd.read_table(table, sep='\t')
			
			tabela_geral = f'/home/alison/Documentos/Projeto/Tabela_geral_Actualizada2.xlsx'
			tabela_geral = pd.read_excel(tabela_geral, sep='\t')

			tabela_geral = tabela_geral[tabela_geral['Especie_Abeja'] == e]
			tabela_geral = tabela_geral[tabela_geral['Audio'] == filename]

			table['peso'] = list(tabela_geral['Peso'])
			table['tamanho torax'] = list(tabela_geral['Tamanho torax'])
			
			end = f'/home/alison/Documentos/Projeto/TabelasAudiosSeparados2/{e}/{table_name}'
			table.to_csv(end, sep='\t')

if __name__ == '__main__':
	#por_genero()
	por_especie()