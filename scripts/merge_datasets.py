import pandas as pd
import re
import os

# Define caminho base como a pasta pai de "scripts"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def ingest_amenidades(arquivo_amenidades):
    df_bairros = pd.read_csv(f'{arquivo_amenidades}')
    df_bairros['bairro'] = df_bairros['bairro'].\
        replace(
            'Parque do Taboão', 
            'Área Rural de Lorena'
            ) 
    
    return df_bairros


def ingest_scraping(arquivo_scraping):
    df_imoveis = pd.read_csv(f'{arquivo_scraping}')
    df_imoveis['local'] = df_imoveis['local'].\
        replace(
            'Loteamento Lorena Village', 
            'Loteamento Village das Palmeiras'
            )
    df_imoveis['local'] = df_imoveis['local'].\
        replace(
            'Santa Lucrécia', 
            'Área Rural de Lorena'
            ) 
    df_imoveis['local'] = df_imoveis['local'].\
        replace(
            r'(?<!Parque )Mondesir', 
            'Parque Mondesir', 
            regex=True
            )
    
    return df_imoveis



def merging_files(amenidades,scraping):
    # Realizar a junção dos dois DataFrames com base no bairro
    df_completo = pd.merge(scraping, 
                           amenidades, 
                           left_on='local', 
                           right_on='bairro', 
                           how='left'
                           )
    

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

    return df_completo

def salvar_df_completo(df, caminho_saida):
    df.to_csv(caminho_saida, index=False)


if __name__ == "__main__":
    #caminhos dos arquivos
    caminho_amenidades = os.path.join(BASE_DIR, 'dados', 'raw', 'amenidades.csv')
    caminho_scraping = os.path.join(BASE_DIR, 'dados', 'raw', 'scraping.csv')
    caminho_saida = os.path.join(BASE_DIR, 'dados', 'processed', 'imoveis_com_amenidades.csv')

    #obtendo dados ingeridos
    df_bairros = ingest_amenidades(caminho_amenidades)
    df_imoveis = ingest_scraping(caminho_scraping)

    #mesclando dados
    df_completo = merging_files(df_bairros, df_imoveis)
    salvar_df_completo(df_completo, caminho_saida)

    #salvando dados no caminho de saída
    print(f"Arquivo salvo como '{caminho_saida}'")