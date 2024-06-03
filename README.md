### This READ.md template was written based on this [repository](https://github.com/FernandoSchett/github_readme_template).

<div align="center" style="line-height: 0.5">

# Relatório SESAB
## Etapa A/Cenário 1
## Entrega 04/06

</div>

<table style="border: none;">
  <tr>
    <td align="left" style="padding-right: 90px; border: none;">
      <a href="link_for_website">
        <img height="125em" src="./assets/SESAB.png" border="0" />
      </a>
    </td>
    <td align="right" style="padding-left: 90px; border: none;">
      <a href="link_for_website">
        <img height="125em" src="./assets/ufba-logo.png" border="0" />
      </a>
    </td>
  </tr>
</table>


<div align="center">

## Integrantes do Grupo:
Adrielly Silva Ferreira de Oliveira,
Danilo Oliveira Andrade,
Diego Quadros dos Santos Dias,
Gabriel Sinzinio Bonfim Cruz,
Guilherme Castro Hora Fontes,
Gustavo Jorge Novaes Silva,
João Victor Leahy de Melo.

</div>

## Objetivo da Etapa A/Cenário 1:
Este relatório contempla uma das entregas que devem ser realizadas para a matéria <div align="center">

**ADML - 43: ACCS: Oficina de Projetos em Inteligência Artificial**

</div>

Este relatório em específico especifica a criação de um modelo de aprendizado de máquina que, a partir dos dados disponibilizados pela SESAB, deve ser capaz de responder a seguinte pergunta:

<div align="center">

**Este indivíduo foi diagnosticado com dengue com base nos dados disponíveis?**

</div>

Para tanto, seguimos algumas regras estabelecidas pelos professores coordenadores da matéria:

- **Pré-Processamento Livre**
- **Algoritmos com hiperparâmetros definidos**
  - *K-Nearest Neighbors (KNN)*
    - Número de Vizinhos (K): Comece com K=5, podendo ser ajustado por meio de validação cruzada.
    - Métrica de Distância: Euclidiana.
    - Peso das amostras: Uniforme.
  - *Árvore de Decisão*
    - Critério de Divisão: Entropia.
    - Profundidade Máxima: Sem limite inicial, ajustável com validação cruzada.
    - Número Mínimo de Amostras para Divisão de um Nó: 2.
    - Número Mínimo de Amostras em um Nó Folha: 1
  - *Rede Neural (Multilayer Perceptron - MLP)*
    - Número de Camadas Ocultas:  2 camadas.
    - Número de Neurônios em Cada Camada Oculta: 100 neurônios.
    - Função de Ativação: ReLU.
    - Solver (Algoritmo de Otimização): Adam.
    - Taxa de Aprendizado: 0.001.
    - Número de Épocas: 200.
    - Tamanho do Lote: 32.
  - *Logistic Regression (Regressão Logística)*
    - Solver: lbfgs.
    - Penalidade: L2.
    - Máximo de Iterações: 100.
  - *Random Forest (Floresta Aleatória)*
    - Número de Árvores na Floresta: 100.
    - Critério de Divisão: Entropia.

Dessa forma, os entregáveis para 04/06 são:
- **Estatística descritiva com distribuição dos dados, padrões e outliers;**
- **Divisão dos conjuntos de treino, teste e validação;** 
- **Validação dos algoritmos com métricas de P, R e F1;**
- **Deploy em um container e conexão com dados;**

## Etapas do Trabalho:

### Pré-Processamento:

Para o pré-processamento, utilizamos a linguagem Python e as suas bibliotecas:

- **SEABORN**
- **MATPLOTLIB**
- **SCIKIT-LEARN**
- **NUMPY**
- **PANDAS**

Para o pré-processamento fizemos algumas etapas:

- **Retirada de Dados Ausentes**
Quando fomos analisar os dados, percebemos que haviam muitas colunas com dados faltantes. A ausência desses dados seria prejudicial na elaboração do modelo e por isso resolvemos,incialmente, retirar as colunas que possuem mais de 30% de dados faltantes. Após isso, resolvemos excluir as linhas que possuiam dados faltantes para que elas também não influenciem negativamente os resultados. Após as operações, a tabela continha 307.753 linhas e 52 colunas.As operações realizadas foram as seguintes:

```python
#Retiramos do DataFrame todas as colunas que possuem mais de 30% dos dados ausentes
threshold = len(data) * 0.7
data_cleaned = data.dropna(thresh=threshold, axis=1)

#Removemos todas as linhas que possuem algum dado ausente
data_cleaned = data_cleaned.dropna()
```
- **Seleção das Colunas que Possuem Relevância para Diagnóstico de Dengue**
Mesmo após a remoção das colunas que possuiam quantidades de dados faltantes, ainda tinhamos 52 colunas, que são muitas features para treinamento de modelo e muitas delas com informações desnecessárias para o diagnóstico de dengue, tais como: município de residência, id da notificação e afins. Por isso, optamos por selecionar apenas as colunas que continham informações relevantes para o diagnóstico. Essas colunas são:

```python
columns_to_keep = ['FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 'VOMITO', 'NAUSEA', 
                       'DOR_COSTAS', 'CONJUNTVIT', 'ARTRITE', 'ARTRALGIA', 'PETEQUIA_N', 
                       'LEUCOPENIA', 'LACO', 'DOR_RETRO', 'DIABETES', 'HEMATOLOG', 
                       'HEPATOPAT', 'RENAL', 'HIPERTENSA', 'ACIDO_PEPT', 'AUTO_IMUNE', 
                       'CS_SEXO', 'SG_UF', 'ID_MN_RESI', 'CLASSI_FIN']

# Filtrar o DataFrame para manter apenas as colunas desejadas
data_cleaned = df[columns_to_keep]
```

Não conseguimos enviar a planilha dos dados pré-processados para o repositório do GitHub devido a sua extensão. Por isso os dados não se encontram no repositório, mas se seguirem o código descrito acima, chegarão aos mesmos dados encontrados pelo grupo.

Segue em anexo os histogramas do pré processamento:

<div align="center">
  <table style="border: none; margin: auto; background-color: white;">
    <tr>
      <td align="center" style="border: none;">
        <a href="link_for_website">
          <img height="1550em" src="./assets/histogramas.png" border="0" />
        </a>
      </td>
    </tr>
  </table>
</div>

Caso a visualização esteja tendo problemas de visualização devido ao modo escuro do GitHub, suegerimos que clique com o botão direito na imagem e escolha abrir a imagem em uma nova guia, onde ela será exibida corretamente.

### Criação do Modelo:

- A seção de criação do modelo começa com a importação das bibliotecas necessárias. As bibliotecas importadas incluem pandas e numpy para manipulação de dados, sklearn para modelagem de aprendizado de máquina, tqdm para barras de progresso e loguru para registro de logs.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,f1_score
from tqdm import tqdm
from loguru import logger as log
```

- Em seguida, a função compute_scores é definida para calcular as métricas F1, precisão e recall.

```python
def compute_scores(y_test,y_pred):
	f1 = f1_score(y_test,y_pred)
	precision = precision_score(y_test,y_pred)
	recall = recall_score(y_test,y_pred)
	return f1,precision,recall
```
- A função run_pipeline é onde a maior parte do trabalho de modelagem é feito. Primeiro, os dados são lidos do arquivo CSV e armazenados em um DataFrame pandas. Em seguida, um dicionário vazio chamado results é criado para armazenar os resultados das métricas de avaliação.


```python

def run_pipeline(): 
	N_ITERS = 1
	N_FOLDS = 5
	df = pd.read_csv("../dengue_pre_processed.csv")

	results = {
		"model_name":[],
		"iteration":[],
		"fold":[],
		"F1": [],
		"Recall": [],
		"Precision":[]
	}
```
- O conjunto de dados é então dividido em recursos (X) e alvo (y). O alvo é a coluna "CLASSI_FIN", que é o que o modelo tentará prever.

```python
	X,y = df.drop("CLASSI_FIN",axis=1).to_numpy(),df["CLASSI_FIN"].to_numpy()
```
- Um loop é iniciado para realizar a validação cruzada estratificada. A validação cruzada estratificada é uma técnica que garante que cada fold da validação cruzada tenha a mesma proporção de observações que o conjunto de dados completo.

- Dentro deste loop, vários modelos de aprendizado de máquina são definidos em uma lista chamada models. Cada modelo é uma tupla contendo o nome do modelo e a instância do modelo com os hiperparâmetros definidos.

- Um loop interno é então iniciado para treinar e testar cada modelo em cada fold da validação cruzada. O modelo é treinado no conjunto de treinamento e as previsões são feitas no conjunto de teste. As métricas F1, precisão e recall são calculadas e armazenadas no dicionário results.

- Finalmente, os resultados são convertidos em um DataFrame pandas e salvos em um arquivo CSV. O DataFrame é agrupado pelo nome do modelo e as médias das métricas são calculadas e salvas em outro arquivo CSV.

```python

	for i in tqdm(range(N_ITERS)):
		cv = StratifiedKFold(n_splits=N_FOLDS,random_state=i,shuffle=True)
			
		models =[
			# ("KNN", KNeighborsClassifier(n_neighbors=5)),
			# ("Decision Tree", DecisionTreeClassifier(criterion='entropy',max_depth=None,min_samples_split=2,min_samples_leaf=1,random_state=i)),
			# ("Logistic Regression", LogisticRegression(penalty='l2', solver='lbfgs', max_iter=100,random_state=i)),
			# ("Random Forest", RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=i)),
			("Multilayer Perceptron", MLPClassifier(hidden_layer_sizes=(100, 100),activation='relu',solver='adam',learning_rate_init=0.001,max_iter=200,batch_size=32,random_state=i))				
		]

		for j,(train_index,test_index) in enumerate(cv.split(X,y)):

			log.info(f"iteration: {i} fold: {j}")
			X_train,y_train = X[train_index],y[train_index]
			X_test, y_test = X[test_index],y[test_index]
			
	
			for model_name,model in models:
				
				model.fit(X_train,y_train)
				y_pred = model.predict(X_test)
				print(y_test)
				print(y_pred)
				f1,precision, recall= compute_scores(y_test,y_pred)

				results['model_name'].append(model_name)
				results['iteration'].append(i)
				results['fold'].append(j)
				results['F1'].append(f1)
				results['Recall'].append(recall)
				results['Precision'].append(precision)
				log.info(f"{model_name}.......... f1: {f1}")

	df_raw = pd.DataFrame(results)

	df = df_raw.groupby(["model_name"]).mean().round(2).reset_index()
	df = df.drop(["iteration","fold"],axis= 1)

	df_raw.to_csv("results/dengue_results_by_fold.csv",index=False)
	df.to_csv("results/dengue_results.csv",index=False)

				
		

if __name__ == "__main__":
	run_pipeline()
```
## Resultados

Por fim, obtivemos os resultados dos algoritmos dos modelos que podem ser encontrados a seguir:

<div align="center">
  <table style="border: none; margin: auto;">
    <tr>
      <td align="center" style="border: none;">
        <a href="link_for_website">
          <img height="250em" src="./assets/precisao_modelos.PNG" border="0" />
        </a>
      </td>
    </tr>
  </table>
</div>


## Conclusão

- Neste relatório, apresentamos o desenvolvimento e avaliação de modelos de aprendizado de máquina para diagnosticar a dengue com base em dados fornecidos pela SESAB. Nossa abordagem seguiu um fluxo de trabalho estruturado, desde o pré-processamento dos dados até a validação cruzada e avaliação de desempenho dos modelos.

### Pré-Processamento
- No pré-processamento, lidamos com dados ausentes e selecionamos colunas relevantes para o diagnóstico da dengue. As operações de limpeza resultaram em um conjunto de dados com 307.753 linhas e 52 colunas, garantindo um dataset mais robusto para a modelagem.

### Modelagem
- Utilizamos várias técnicas de aprendizado de máquina, incluindo K-Nearest Neighbors (KNN), Árvore de Decisão, Rede Neural (Multilayer Perceptron - MLP), Regressão Logística e Random Forest. Para cada técnica, foram definidos hiperparâmetros específicos conforme as orientações fornecidas pelos coordenadores da matéria.

### Avaliação
- Avaliamos os modelos utilizando validação cruzada estratificada e métricas de F1, precisão e recall. Os resultados das avaliações foram detalhados e comparados, oferecendo uma visão clara do desempenho de cada abordagem.

### Resultados
- Os resultados dos modelos mostraram diferentes níveis de eficácia, destacando-se o desempenho da Rede Neural (MLP) em termos de precisão, recall e F1-score. Estes resultados indicam que a MLP pode ser uma abordagem promissora para o diagnóstico de dengue com base nos dados disponíveis.

## Conclusões Finais
- Os modelos desenvolvidos e os processos de pré-processamento implementados demonstram a eficácia do uso de técnicas de aprendizado de máquina para o diagnóstico de doenças como a dengue. A abordagem sistemática desde a limpeza de dados até a avaliação dos modelos garantiu resultados confiáveis e robustos.

- Para futuras iterações, recomenda-se explorar técnicas de ajuste de hiperparâmetros e incorporação de novos dados, além de testar outras arquiteturas de redes neurais para potencialmente melhorar ainda mais o desempenho do modelo.
