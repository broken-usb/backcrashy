import os
import sys
import numpy        as np

from neural_network import RedeNeural
from data           import carregar_dados
from predictor      import mostrar_testes, modo_interativo

ARQUIVO_PESOS  = "pesos.npz"
ARQUIVO_TREINO = "dados.csv"
ARQUIVO_TESTE  = "testes.csv"
EPOCAS = 15000
TAXA   = 0.01

np.random.seed(42)

# verifica se os CSVs estao na pasta
for arq in [ARQUIVO_TREINO, ARQUIVO_TESTE]:
    if not os.path.exists(arq):
        print(f"Erro: '{arq}' nao encontrado! Coloque os CSVs na mesma pasta que o main.py.")
        sys.exit(1)

# 7 entradas: idade, ano, sexo, airbag, cinto, delta-v, frontal
rede = RedeNeural(entradas=7, ocultos=12, saidas=1, taxa=TAXA)

if os.path.exists(ARQUIVO_PESOS):
    print("\n--- MODO DE TESTE ---")

    max_idade, max_ano = rede.carregar(ARQUIVO_PESOS)

    # compatibilidade com pesos antigos sem normalizacao salva
    if max_idade is None:
        print("Aviso: normalizacao nao encontrada, recalculando a partir do treino...")
        _, _, max_idade, max_ano = carregar_dados(ARQUIVO_TREINO)

    X_teste, y_teste, _, _ = carregar_dados(ARQUIVO_TESTE, max_idade=max_idade, max_ano=max_ano)

    print("\nO que voce quer fazer?")
    print("  1 - Testes automaticos (com dados reais do CSV)")
    print("  2 - Inserir dados manualmente")
    print("  3 - Os dois")
    opcao = input("Escolha (1/2/3): ").strip()

    if opcao in ("1", "3"):
        print(f"\nQuantos casos de cada grupo voce quer testar?")
        print(f"(O CSV de teste tem {len(X_teste)} amostras no total)")
        while True:
            try:
                n = int(input("Quantidade por grupo (vivos e mortos): "))
                if n >= 1: break
                print("Digite pelo menos 1.")
            except ValueError:
                print("Digite um numero inteiro.")
        mostrar_testes(rede, X_teste, y_teste, max_idade, max_ano, n=n)

    if opcao in ("2", "3"):
        modo_interativo(rede, max_idade, max_ano)

    if opcao not in ("1", "2", "3"):
        print("Opcao invalida, encerrando.")

else:
    print("\n--- MODO DE TREINAMENTO ---")

    X_treino, y_treino, max_idade, max_ano = carregar_dados(ARQUIVO_TREINO, balancear=True)
    rede.treinar(X_treino, y_treino, EPOCAS)
    rede.salvar(ARQUIVO_PESOS, max_idade, max_ano)

    print("\nTreinamento concluido! Rode o programa de novo para testar.")