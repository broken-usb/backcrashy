import numpy    as np

class RedeNeural:
    def __init__(self, entradas, ocultos, saidas, taxa=0.1):
        self.taxa = taxa
        self.W1 = np.random.randn(entradas, ocultos) * 0.1 # pesos entre entrada e camada oculta
        self.b1 = np.zeros((1, ocultos))                   # bias da camada oculta
        self.W2 = np.random.randn(ocultos, saidas) * 0.1   # pesos entre camada oculta e saida
        self.b2 = np.zeros((1, saidas))                    # bias da saida

    def sigmoide(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoide(self, x):
        return x * (1 - x)

    def treinar(self, X, y, epocas):
        print(f"Iniciando treinamento com {epocas} epocas...")
        for epoca in range(epocas):
            # forward propagation
            A1 = self.sigmoide(np.dot(X, self.W1) + self.b1)
            A2 = self.sigmoide(np.dot(A1, self.W2) + self.b2)

            # backpropagation
            erro   = y - A2
            delta2 = erro * self.derivada_sigmoide(A2)
            delta1 = delta2.dot(self.W2.T) * self.derivada_sigmoide(A1)

            # atualiza pesos
            self.W2 += A1.T.dot(delta2) * self.taxa
            self.b2 += np.sum(delta2, axis=0, keepdims=True) * self.taxa
            self.W1 += X.T.dot(delta1) * self.taxa
            self.b1 += np.sum(delta1, axis=0, keepdims=True) * self.taxa

            if epoca % 1000 == 0 or epoca == epocas - 1:
                erro_medio = np.mean(np.abs(erro))
                acuracia   = np.mean((A2 >= 0.5).astype(int) == y) * 100
                print(f"  Epoca {epoca:5d} | Erro: {erro_medio:.4f} | Acuracia: {acuracia:.2f}%")

    def prever(self, X):
        A1 = self.sigmoide(np.dot(X, self.W1) + self.b1)
        return self.sigmoide(np.dot(A1, self.W2) + self.b2)

    def salvar(self, arquivo, max_idade, max_ano):
        np.savez(arquivo, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 max_idade=max_idade, max_ano=max_ano)
        print(f"Pesos salvos em '{arquivo}'!")

    def carregar(self, arquivo):
        dados     = np.load(arquivo)
        self.W1   = dados["W1"]
        self.b1   = dados["b1"]
        self.W2   = dados["W2"]
        self.b2   = dados["b2"]
        max_idade = float(dados["max_idade"]) if "max_idade" in dados else None
        max_ano   = float(dados["max_ano"])   if "max_ano"   in dados else None
        print(f"Pesos carregados de '{arquivo}'!")
        return max_idade, max_ano