# 🏠 Análise Inteligente do Mercado Imobiliário de Lorena-SP

Projeto completo que automatiza a coleta, tratamento e análise de dados do mercado imobiliário de **Lorena - SP**. O objetivo é gerar insights geográficos e preditivos que auxiliam na compreensão e precificação dos imóveis da cidade, combinando técnicas de web scraping, análise de dados e machine learning.

---

## 🔍 Visão Geral

Este projeto integra dados do site **Viva Real** usando **web scraping com Selenium**, enriquece as informações com dados geográficos da **Google Places API**, e aplica técnicas de ciência de dados para analisar e prever preços imobiliários locais.

O resultado é uma base robusta que permite entender não só o preço dos imóveis, mas também como a proximidade a escolas, mercados e outros pontos de interesse impacta o valor.

---

## 🚀 Objetivos do Projeto

- Automatizar a coleta de dados reais do mercado imobiliário local  
- Tratar e limpar os dados para garantir qualidade  
- Enriquecer a base com informações geográficas relevantes  
- Realizar análise exploratória para identificar padrões e correlações  
- Construir modelos preditivos para estimar preços dos imóveis com boa acurácia  
- Facilitar tomadas de decisão para compradores, vendedores e corretores

---

## ⚙️ Tecnologias e Ferramentas

- 🐍 Python 3 — linguagem principal do projeto  
- 🕸️ Selenium — para web scraping automatizado e robusto  
- 📊 Pandas — limpeza e manipulação de dados  
- 📈 Matplotlib / Seaborn — visualização e análise exploratória (EDA)  
- 🗺️ Google Places API — coleta de dados de amenidades próximas (escolas, mercados, hospitais, etc.)  
- 🤖 Scikit-learn — modelagem e machine learning para previsão de preços  

---

## 📌 Estrutura do Projeto

- **`scripts/`** — códigos para scraping e processamento inicial  
- **`notebooks/`** — análises exploratórias e experimentos com machine learning  
- **`dados/`** — datasets brutos e tratados  
- **`app/`** — código para integração e execução do pipeline completo  
- **`saidas/eda_graphs/`** — gráficos e visualizações geradas  
- **`utils/`** — funções auxiliares reutilizáveis  

---

## 📋 Passo a passo do fluxo de trabalho

1. **Coleta**: Extração automatizada dos anúncios do Viva Real via Selenium.  
2. **Tratamento**: Limpeza, padronização e organização dos dados com Pandas.  
3. **Enriquecimento**: Consulta à Google Places API para mapear amenidades ao redor dos imóveis.  
4. **Análise Exploratória**: Exploração visual e estatística para entender as variáveis e relações entre elas.  
5. **Modelagem**: Treinamento e avaliação de modelos preditivos (regressão) para estimar preços.  

---

## 🎯 Resultados e Insights

- Descoberta de correlações importantes entre proximidade a pontos de interesse e preço dos imóveis  
- Visualizações claras para identificar bairros mais valorizados  
- Modelos capazes de prever preços com boa precisão, auxiliando decisões no mercado local  

---

## 📖 Como usar este projeto

1. Clone o repositório:  
   ```bash
   git clone https://github.com/Homailson/LorenaHousingData.git
   cd LorenaHousingData


2. Instale as dependências:
   ```bash
   pip install -r requirements.txt

3. Configure as credenciais da Google Places API

4. Execute os scripts conforme a ordem no README para coleta, tratamento e análise

## 🚀 Próximos passos (em desenvolvimento)
* Adicionar visualizações interativas com Plotly ou Folium

* Desenvolver uma interface web para exploração dos dados e resultados

* Automatizar todo o pipeline para execução periódica e atualização de dados

* Implementar testes unitários e documentação detalhada das funções
