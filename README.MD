# Projeto de Predição de Preços de Imóveis com Machine Learning

## 📍 Objetivo Geral
Criar um modelo de machine learning que prediz o valor de venda e aluguel de imóveis em **Lorena-SP**, considerando tanto as características do imóvel quanto os **benefícios da região onde ele está localizado**, como proximidade de escolas, hospitais, supermercados, etc.

---

## 🔹 Etapas do Projeto

### 📅 1. Coleta de Dados (Scraping)
- Site alvo: ZAP Imóveis (ou similar)
- Dados a serem coletados:
  - Tipo de imóvel
  - Preço de venda/aluguel
  - Área (m²)
  - Quartos, banheiros, vagas
  - Bairro ou endereço
  - Latitude/Longitude (se disponível na página)
- Ferramentas: `requests`, `BeautifulSoup`, `Selenium`
- Caso não haja lat/lon, usar geocodificação reversa com Google Maps

### 🌍 2. Geolocalização
- Usar **Google Maps Geocoding API** para transformar endereço em lat/lon
- Usar **Google Places API** para verificar:
  - Distância até escolas, hospitais, supermercados, academias, farmácias
  - Número de estabelecimentos por tipo em um raio X
- Dados se tornam novas features para o modelo

### 🤓 3. Limpeza e Preparação dos Dados
- Tratar valores nulos, converter tipos, normalizar valores
- Codificação de variáveis categóricas (`pandas.get_dummies`, `LabelEncoder`)
- Remoção de outliers e imputação de valores ausentes

### 📊 4. Análise Exploratória (EDA)
- Ver correlações, mapas de calor, gráficos de dispersão
- Análise por bairro, tipo de imóvel, faixa de preço

### 🤖 5. Modelagem Preditiva
- Modelos sugeridos:
  - `LinearRegression`
  - `RandomForestRegressor`
  - `XGBoost`
- Avaliação:
  - MAE, RMSE, R²
- Cross-validation para validação robusta

### 📍 6. Predição em Regiões Próximas
- Predizer valores em regiões com características semelhantes onde ainda não há dados
- Pode ser aplicado para imóveis na planta

### 🚀 7. Interface ou API (Opcional)
- Criar uma API (Flask ou FastAPI) que recebe dados de um imóvel e retorna a predição
- Ou criar um painel com Dash/Streamlit com filtros e mapa interativo

---

## 🔑 APIs Necessárias

| API                     | Finalidade                            |
|-------------------------|----------------------------------------|
| Google Maps Geocoding   | Obter lat/lon a partir de endereço     |
| Google Places API       | Buscar estabelecimentos próximos       |
| Google Distance Matrix  | (Opcional) Calcular tempo/distância    |

Necessário gerar uma **chave de API do Google Cloud** e ativar os serviços desejados.

---

## ✅ Conclusão
Com esse planejamento, é possível executar um pipeline completo de coleta, enriquecimento e modelagem para predição de preços imobiliários com base em dados estruturados e geográficos.

Se precisar, posso também montar um template inicial de scraping, pipeline de features, ou um notebook para os testes. É só pedir!