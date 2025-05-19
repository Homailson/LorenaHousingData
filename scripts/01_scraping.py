# pip install selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import csv
import time


# Configurações do driver
def configurar_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")  # Abre o navegador maximizado
    # options.add_argument("--headless")  # Se quiser headless
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(60)
    return driver


# Verificando bloqueios de Captcha
def lidar_com_captcha(driver):
    try:
        WebDriverWait(driver, 10).until(
            EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//iframe[contains(@src, 'captcha')]"))
        )
        print("CAPTCHA detectado. Por favor resolva manualmente...")
        while True:
            try:
                driver.switch_to.default_content()
                # Se conseguir sair do frame, assume que foi resolvido
                break
            except Exception:
                time.sleep(2)
    except TimeoutException:
        print("Nenhum CAPTCHA detectado.")


# Extraindo cards dos imóveis da página
def get_cards(driver, tempo_de_rolagem):
    while True:
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(tempo_de_rolagem)
        altura_atual = driver.execute_script("return window.scrollY + window.innerHeight")
        altura_doc = driver.execute_script("return document.body.scrollHeight")
        if altura_atual >= altura_doc:
            print("Fim da rolagem da página")
            break
    cards = driver.find_elements(By.CSS_SELECTOR, "[data-cy='rp-property-cd']")
    return cards


# Extraindo os dados dos cards
def extrair_dados(cards):
    dados_imoveis = []
    for card in cards:
        dados = {}
        campos = {
            "area": "rp-cardProperty-propertyArea-txt",
            "quartos": "rp-cardProperty-bedroomQuantity-txt",
            "banheiros": "rp-cardProperty-bathroomQuantity-txt",
            "vagas": "rp-cardProperty-parkingSpacesQuantity-txt",
            "preco": "rp-cardProperty-price-txt",
            "local": "rp-cardProperty-location-txt"
        }

        for chave, seletor in campos.items():
            try:
                elemento = card.find_element(By.CSS_SELECTOR, f"[data-cy='{seletor}']")
                texto = elemento.text.strip()

                # Agora fazemos a limpeza
                if chave == "area":
                    texto = texto.replace("Tamanho do imóvel\n", "")
                    texto = texto.replace("m²", "").strip()
                elif chave == "quartos":
                    texto = texto.replace("Quantidade de quartos\n", "")
                elif chave == "banheiros":
                    texto = texto.replace("Quantidade de banheiros\n", "")
                elif chave == "vagas":
                    texto = texto.replace("Quantidade de vagas de garagem\n", "")
                elif chave == "local":
                    texto = texto.split("\n")[-1].strip()
                    texto = texto.split(",")[0].strip()
                    texto = texto.replace("Cruz", "Bairro da Cruz") # quando o bairro for Cruz, troca por bairro da Cruz
                elif chave == "preco":
                    texto = texto.split("\n")[0]  # Pega só a primeira linha
                    texto = texto.replace("R$", "").replace(".", "").replace("A partir de  ", "").strip()  # Remove 'R$' e espaços


                dados[chave] = texto
            except NoSuchElementException:
                dados[chave] = None

        dados_imoveis.append(dados)
    return dados_imoveis


def salvar_dados(dados, nome_arquivo):
    colunas = ["area", "quartos", "banheiros", "vagas", "preco", "local"]
    with open(nome_arquivo, mode='w', newline='', encoding='utf-8') as arquivo_csv:
        writer = csv.DictWriter(arquivo_csv, fieldnames=colunas)
        writer.writeheader()
        writer.writerows(dados)
    print(f"Dados salvos no arquivo {nome_arquivo} com sucesso!")


def main():
    driver = configurar_driver()
    url = "https://www.vivareal.com.br/venda/sp/lorena/"
    driver.get(url)

    lidar_com_captcha(driver)

    todos_os_imoveis = []

    pagina_atual = 1
    while True:
        cards = get_cards(driver, tempo_de_rolagem=4)
        print(f"Total de imóveis carregados na página {pagina_atual}: {len(cards)}")
        
        # Extrai os dados
        dados = extrair_dados(cards)
        todos_os_imoveis.extend(dados)
        
        # Agora tenta clicar no botão da próxima página
        try:
            pagina_atual += 1
            botao_pagina = driver.find_element(By.CSS_SELECTOR, f"button[data-testid='button-page-{pagina_atual}']")
            botao_pagina.click()
            time.sleep(10)
            
        except NoSuchElementException:
            print("Não há mais páginas para navegar. Encerrando.")
            break
        except TimeoutException:
            print("Elementos da nova página não carregaram a tempo. Encerrando.")
            break

    # Fecha o navegador
    driver.quit()

    # Exibe os dados coletados
    for i, imovel in enumerate(todos_os_imoveis, 1):
        print(f"{i}. {imovel}")

    salvar_dados(todos_os_imoveis, "imoveis_lorena_sp.csv")


if __name__ == "__main__":
    main()