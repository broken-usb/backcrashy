import pandas   as pd
import numpy    as np

# velocidade do impacto com categorias ordenadas do menor pro maior risco
DVCAT_MAP = {
    "1-9km/h": 0,
    "10-24":   1,
    "25-39":   2,
    "40-54":   3,
    "55+":     4,
}

def carregar_dados(caminho, balancear=False, max_idade=None, max_ano=None):
    print(f"Carregando '{caminho}'...")
    df = pd.read_csv(caminho)

    df = df[["ageOFocc", "yearVeh", "sex", "airbag", "seatbelt", "dvcat", "frontal", "dead"]].copy()

    # converte texto pra numero
    df["sex"]      = df["sex"].map({"m": 0, "f": 1})
    df["airbag"]   = df["airbag"].map({"none": 0, "airbag": 1})
    df["seatbelt"]  = df["seatbelt"].map({"none": 0, "belted": 1})
    df["dead"]     = df["dead"].map({"alive": 0, "dead": 1})
    df["dvcat"]    = df["dvcat"].map(DVCAT_MAP)

    antes = len(df)
    df.dropna(inplace=True) # remove linhas com dados faltando
    if len(df) < antes:
        print(f"  Aviso: {antes - len(df)} linha(s) removida(s) por dados faltando.")

    df = df.astype(float)

    # balanceia mortos e vivos pra nao viciar o treino (so usa no treino)
    if balancear:
        mortos = df[df["dead"] == 1]
        vivos  = df[df["dead"] == 0].sample(n=len(mortos), random_state=42)
        df = pd.concat([mortos, vivos]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  Balanceado: {len(mortos)} mortos e {len(vivos)} vivos")

    # normaliza pra valores entre 0 e 1
    # se vier max de fora usa ele
    if max_idade is None: max_idade = float(df["ageOFocc"].max())
    if max_ano   is None: max_ano   = float(df["yearVeh"].max())

    df["ageOFocc"] = df["ageOFocc"] / max_idade
    df["yearVeh"]  = df["yearVeh"]  / max_ano
    df["dvcat"]    = df["dvcat"]    / 4.0

    X = df[["ageOFocc", "yearVeh", "sex", "airbag", "seatbelt", "dvcat", "frontal"]].values
    y = df[["dead"]].values

    print(f"  {len(df)} amostras carregadas.")
    return X, y, max_idade, max_ano