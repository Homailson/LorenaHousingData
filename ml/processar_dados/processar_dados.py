import pandas as pd
import re

# Carregar os dados dos dois arquivos CSV
df_bairros = pd.read_csv('ml/dados/neighborhood.csv')  # Arquivo de bairros
df_bairros['bairro'] = df_bairros['bairro'].replace('Parque do Taboão', 'Área Rural de Lorena') #trocando nome de coluna para casar com o outro dataframe
df_imoveis = pd.read_csv('ml/dados/imoveis.csv')  # Arquivo de imóveis
df_imoveis['local'] = df_imoveis['local'].replace('Loteamento Lorena Village', 'Loteamento Village das Palmeiras') #Loteamento Village das Palmeiras e Lorena Village são o mesmo bairro
df_imoveis['local'] = df_imoveis['local'].replace('Santa Lucrécia', 'Área Rural de Lorena') #trocando nome de coluna para casar com o outro dataframe
df_imoveis['local'] = df_imoveis['local'].replace(r'(?<!Parque )Mondesir', 'Parque Mondesir', regex=True)
# Realizar a junção dos dois DataFrames com base no bairro
df_completo = pd.merge(df_imoveis, df_bairros, left_on='local', right_on='bairro', how='left')

# Removendo a coluna local
df_completo = df_completo.drop(columns=['local'])

# Lista para ordenação das colunas
nova_ordem = ["bairro",
        "area",
        "quartos",
        "banheiros",
        "vagas",
        "hospital",
        "gas_station",
        "school",
        "supermarket",
        "restaurant",
        "pharmacy",
        "preco"
        ]

# Dataframe com as colunas em nova ordem
df_completo = df_completo[nova_ordem]

# Mostrar o DataFrame resultante
print(df_completo.head())

# Salvar o DataFrame resultante em um novo arquivo CSV
df_completo.to_csv('ml/saida/imoveis_com_bairros.csv', index=False)

print("Arquivo salvo como 'imoveis_com_bairros.csv'")

