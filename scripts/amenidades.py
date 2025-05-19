import requests
import concurrent.futures
import os
import pandas as pd
from tqdm import tqdm
import argparse
import time

# Constantes
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise ValueError("API_KEY não configurada. Defina a variável de ambiente 'API_KEY'.")

URL_GEOCODE = "https://maps.googleapis.com/maps/api/geocode/json"
URL_PLACES = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

# Tipos de amenidades
tipos = ['school', 'hospital', 'gas_station', 'supermarket', 'restaurant', 'pharmacy']

# Função para tratar requisições
def fazer_requisicao_segura(url, params=None, timeout=10):
    try:
        resposta = requests.get(url, params=params, timeout=timeout)
        resposta.raise_for_status()
        return resposta.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")
    return None

# Função para buscar coordenadas de 1 bairro
def buscar_coordenada(args):
    bairro, cidade = args
    params = {"address": f"{bairro}, {cidade}", "key": API_KEY}
    dados = fazer_requisicao_segura(URL_GEOCODE, params=params)
    if dados and dados.get('status') == 'OK' and dados['results']:
        location = dados['results'][0]['geometry']['location']
        return bairro, (location['lat'], location['lng'])
    else:
        print(f"Erro ao obter localização de {bairro}: {dados.get('status', 'Sem status') if dados else 'Sem resposta'}")
        return bairro, None

# Função para buscar lugares em 1 coordenada
def buscar_amenidades(nome_lat_lon):
    nome, (lat, lon) = nome_lat_lon
    amenidades = {}
    for tipo in tipos:
        params = {"location": f"{lat},{lon}", "radius": 1000, "type": tipo, "key": API_KEY}
        dados = fazer_requisicao_segura(URL_PLACES, params=params)
        amenidades[tipo] = 1 if dados and dados.get('status') == 'OK' else 0
        time.sleep(0.1)  # Evitar ultrapassar limite da API
    return {'bairro': nome, **amenidades}

# Função principal para obter todas coordenadas
def obter_lat_lon(bairros, cidade):
    coordenadas = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        resultados = list(tqdm(executor.map(buscar_coordenada, [(bairro, cidade) for bairro in bairros]),
                               total=len(bairros),
                               desc="Buscando coordenadas"))
        for bairro, coord in resultados:
            if coord:
                coordenadas[bairro] = coord
    return coordenadas

# Função principal para buscar todos os lugares
def buscar_lugares(coordenadas):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        resultados = list(tqdm(executor.map(buscar_amenidades, coordenadas.items()),
                               total=len(coordenadas),
                               desc="Buscando amenidades"))
    return resultados

# Função para salvar CSV
def salvar_csv(dados, caminho_saida):
    df = pd.DataFrame(dados)
    df.to_csv(caminho_saida, index=False)
    print(f"Arquivo salvo em {caminho_saida}")

# Função principal
def main(caminho_saida):
    bairros = [
        "Centro", "Vila Nunes", "Vila São Roque", "Santo Antônio", "Bairro da Cruz",
        "Parque das Rodovias", "Cidade Industrial", "Residencial Vila Rica",
        "Loteamento Jardim Primavera", "Vila dos Comerciários I", "Parque Mondesir",
        "Ponte Nova", "Vila Passos", "Jardim Novo Horizonte", "Vila Hepacare",
        "Vila Brito", "Nova Lorena", "Aterrado", "Cabelinha", "Vila Geny", "Olaria",
        "Portal das Palmeiras", "Vila Santa Edwiges", "Vila Portugal",
        "Loteamento Village das Palmeiras", "Loteamento Lorena Village",
        "Jardim Margarida", "Parque do Taboão", "Loteamento Colinas de Lorena", "Cecap"
    ]

    cidade = 'Lorena, São Paulo, Brasil'
    
    coordenadas = obter_lat_lon(bairros, cidade)
    informacoes = buscar_lugares(coordenadas)
    salvar_csv(informacoes, caminho_saida)

# Rodar com argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Busca de bairros e amenidades")
    parser.add_argument('--saida', type=str, default='dados/neighborhood.csv', help="Arquivo de saída CSV")
    args = parser.parse_args()

    main(args.saida)