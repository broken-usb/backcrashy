import os
import numpy    as np
from data       import DVCAT_MAP

# cores para facilitar na hora de mostrar os resultados
VERDE    = '\033[1;92m'
VERMELHO = '\033[1;91m'
AMARELO  = '\033[1;93m'
CIANO    = '\033[1;96m'
CINZA    = '\033[1;90m'
RESET    = '\033[m'

if os.name == 'nt':  # desativa as cores no Windows
    '''
    quando fui testar o projeto em uma VM do Windows
    ele não estáva conseguindo mostrar as cores no Terminal
    exibindo o código delas ao invez de mudar a cor
    então optei para desligar as cores no Windows
    '''
    VERDE = VERMELHO = AMARELO = CIANO = CINZA = RESET = ''

SEP  = f'{CIANO} {"=-"*26}{RESET}'  #separador padrao
DVCAT_REVERSO = {v: k for k, v in DVCAT_MAP.items()}

# opcoes de Delta-V
DVCAT_OPCOES = {
    "1": "1-9km/h",
    "2": "10-24",
    "3": "25-39",
    "4": "40-54",
    "5": "55+",
}


def _linha(campo, valor, cor_valor=''):
    # imprime uma linha de dado no formato: campo | valor
    print(f' {AMARELO}{campo:<18}{RESET} | {cor_valor}{valor}{RESET if cor_valor else ""}')


def mostrar_testes(rede, X_teste, y_teste, max_idade, max_ano, n=5):
    idx_vivos  = np.where(y_teste.flatten() == 0)[0][:n]
    idx_mortos = np.where(y_teste.flatten() == 1)[0][:n]
    indices    = np.concatenate((idx_vivos, idx_mortos))

    previsoes = rede.prever(X_teste[indices])
    acertos   = 0

    print(f'\n{CIANO} {"TESTANDO A REDE COM DADOS REAIS":^50}{RESET}')
    print(SEP)

    for i, idx in enumerate(indices):
        x = X_teste[idx]

        # desnormaliza os valores pra mostrar os originais
        idade   = round(x[0] * max_idade)
        ano     = round(x[1] * max_ano)
        sexo    = "Feminino"  if x[2] == 1 else "Masculino"
        airbag  = "Sim"       if x[3] == 1 else "Nao"
        cinto   = "Sim"       if x[4] == 1 else "Nao"
        dvcat   = DVCAT_REVERSO.get(round(x[5] * 4), "?")
        frontal = "Sim"       if x[6] == 1 else "Nao"

        real     = "Morto" if y_teste[idx][0] == 1 else "Vivo"
        prob     = float(previsoes[i][0])
        previsto = "Morto" if prob >= 0.5 else "Vivo"
        acertou  = real == previsto
        acertos += int(acertou)

        cor_real = VERMELHO if real == "Morto" else VERDE

        print(f'\n{SEP}')
        print(f' Caso {i+1:02d} | Real: {cor_real}{real}{RESET}')
        print(SEP)
        _linha("Idade",           f"{idade} anos")
        _linha("Sexo",            sexo)
        _linha("Ano do veiculo",  str(ano))
        _linha("Airbag",          airbag,  VERDE if airbag == "Sim" else VERMELHO)
        _linha("Cinto",           cinto,   VERDE if cinto  == "Sim" else VERMELHO)
        _linha("Delta-V",         f"{dvcat} km/h")
        _linha("Colisao frontal", frontal)

        if acertou:
            print(f'{VERDE} --> ACERTO | Previsto: {previsto} ({prob*100:.1f}%){RESET}')
        else:
            print(f'{VERMELHO} --> ERRO   | Previsto: {previsto} ({prob*100:.1f}%){RESET}')

    total = len(indices)
    cor   = VERDE if acertos == total else (AMARELO if acertos >= total * 0.7 else VERMELHO)
    print(f'\n{SEP}')
    print(f'{cor} {"RESULTADO FINAL":^50}{RESET}')
    print(f'{cor} {f"Acertos: {acertos}/{total}  ({acertos/total*100:.1f}%)":^50}{RESET}')
    print(SEP)


def modo_interativo(rede, max_idade, max_ano):
    acertos_manual = 0
    erros_manual   = 0
    inventados     = 0

    print(f'\n{CIANO} {"MODO INTERATIVO":^50}{RESET}')
    print(SEP)

    while True:
        resp = input(f'\n{AMARELO}Deseja analisar um ocupante manualmente? (s/n): {RESET}').strip().lower()
        if resp != "s":
            break

        try:
            print(f'\n{CIANO} {"DADOS DO ACIDENTE":^50}{RESET}')
            print(SEP)

            print(f'\n {AMARELO}[1] Idade do ocupante{RESET}')
            print(f' {CINZA}Quantos anos tinha a pessoa no veiculo.{RESET}')
            idade = float(input(f' {AMARELO}Digite (1 a {int(max_idade)}): {RESET}'))

            print(f'\n {AMARELO}[2] Ano do veiculo{RESET}')
            print(f' {CINZA}Ano de fabricacao do carro.{RESET}')
            ano = float(input(f' {AMARELO}Digite (ex: 2005, max. {int(max_ano)}): {RESET}'))

            print(f'\n {AMARELO}[3] Sexo do ocupante{RESET}')
            sexo_str = input(f' {AMARELO}M = Masculino  |  F = Feminino: {RESET}').strip().upper()
            sexo = 0 if sexo_str == "M" else 1

            print(f'\n {AMARELO}[4] Airbag{RESET}')
            print(f' {CINZA}O veiculo possuia airbag? (independente de ter disparado){RESET}')
            airbag_str = input(f' {AMARELO}S = Sim  |  N = Nao: {RESET}').strip().upper()
            airbag = 1 if airbag_str == "S" else 0

            print(f'\n {AMARELO}[5] Cinto de seguranca{RESET}')
            print(f' {CINZA}O ocupante usava cinto no momento do impacto?{RESET}')
            cinto_str = input(f' {AMARELO}S = Sim  |  N = Nao: {RESET}').strip().upper()
            cinto = 1 if cinto_str == "S" else 0

            print(f'\n {AMARELO}[6] Faixa de velocidade do impacto (Delta-V){RESET}')
            print(f' {CINZA}NAO e a velocidade antes do acidente.{RESET}')
            print(f' {CINZA}E a variacao de velocidade sofrida durante o impacto.{RESET}')
            print(f' {CINZA}  1 - Leve    ( 1 a  9 km/h){RESET}')
            print(f' {CINZA}  2 - Baixo   (10 a 24 km/h){RESET}')
            print(f' {CINZA}  3 - Medio   (25 a 39 km/h){RESET}')
            print(f' {CINZA}  4 - Grave   (40 a 54 km/h){RESET}')
            print(f' {CINZA}  5 - Severo  (55+ km/h){RESET}')
            dvcat_num = input(f' {AMARELO}Digite o numero (1 a 5): {RESET}').strip()

            if dvcat_num not in DVCAT_OPCOES:
                raise ValueError(f"'{dvcat_num}' invalido. Digite um numero de 1 a 5.")
            dvcat_norm = DVCAT_MAP[DVCAT_OPCOES[dvcat_num]] / 4.0

            print(f'\n {AMARELO}[7] Tipo de colisao{RESET}')
            print(f' {CINZA}A colisao foi frontal (de frente)?{RESET}')
            frontal_str = input(f' {AMARELO}S = Frontal  |  N = Lateral/Traseira: {RESET}').strip().upper()
            frontal = 1 if frontal_str == "S" else 0

            # monta o vetor e faz a previsao
            X_manual = np.array([[
                idade / max_idade,
                ano   / max_ano,
                sexo, airbag, cinto,
                dvcat_norm, frontal
            ]])

            prob      = float(rede.prever(X_manual)[0][0])
            resultado = "MORTO" if prob >= 0.5 else "SOBREVIVEU"
            cor_res   = VERMELHO if resultado == "MORTO" else VERDE

            # barra de probabilidade visual
            cheios = round(prob * 20)
            barra  = VERMELHO + "#" * cheios + CINZA + "." * (20 - cheios) + RESET

            print(f'\n{CIANO} {"RESULTADO":^50}{RESET}')
            print(SEP)
            _linha("Previsao",       resultado, cor_res)
            _linha("Prob. de obito", f"[{barra}] {prob*100:.1f}%")
            print(SEP)

            # pergunta se a rede acertou
            print(f'\n {AMARELO}A previsao foi correta?{RESET}')
            print(f' {CINZA}  1 - Sim, acertou!{RESET}')
            print(f' {CINZA}  2 - Nao, errou{RESET}')
            print(f' {CINZA}  3 - Nao sei, inventei o acidente{RESET}')
            feedback = input(f' {AMARELO}Escolha (1/2/3): {RESET}').strip()

            if feedback == "1":
                acertos_manual += 1
                print(f' {VERDE}Que bom! A rede acertou :){RESET}')
            elif feedback == "2":
                erros_manual += 1
                print(f' {VERMELHO}Que pena, a rede errou :({RESET}')
            else:
                inventados += 1
                print(f' {CINZA}Sem problema, inventar e otimo pra testar!{RESET}')

        except ValueError as e:
            print(f' {VERMELHO}Erro: {e}{RESET}')

    # resumo da sessao
    casos_reais = acertos_manual + erros_manual
    if casos_reais > 0 or inventados > 0:
        print(f'\n{CIANO} {"RESUMO DA SESSAO":^50}{RESET}')
        print(SEP)
        if casos_reais > 0:
            _linha("Acertos reais", f"{acertos_manual}/{casos_reais}", VERDE)
            _linha("Erros reais",   f"{erros_manual}/{casos_reais}",   VERMELHO)
        if inventados > 0:
            _linha("Inventados", str(inventados), CINZA)
        print(SEP)

    print(f'\n{CINZA} Encerrando. Ate logo!{RESET}\n')