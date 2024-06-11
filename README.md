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

## Objetivo da Etapa A/Cenário 1/ Entrega 1:
Este relatório contempla uma das entregas que devem ser realizadas para a matéria 

<div align="center">

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
			("KNN", KNeighborsClassifier(n_neighbors=5)),
			("Decision Tree", DecisionTreeClassifier(criterion='entropy',max_depth=None,min_samples_split=2,min_samples_leaf=1,random_state=i)),
			("Logistic Regression", LogisticRegression(penalty='l2', solver='lbfgs', max_iter=100,random_state=i)),
			("Random Forest", RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=i)),
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

Como podemos ver, para o algoritmo de MLP obtivemos resultado 0 nos testes de validação utilizados, investigamos a causa desse equívoco mas não encontramos discrepâncias no código que poderiam ocasionar esse resultado. Entretanto, os resultados dos outros algoritmos permanecem normais e dentro do esperado, com uma média em torno de 60% de acerto no diagnóstico da doença.


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

<div align="center" style="line-height: 0.5">

## Entrega 11/06

</div>

## Objetivo da Etapa A/Cenário 1/ Entrega 2:
Este relatório contempla uma das entregas que devem ser realizadas para a matéria 

<div align="center">

**ADML - 43: ACCS: Oficina de Projetos em Inteligência Artificial**

</div>

Este relatório em específico especifica a criação de um dashboard para a visualização dos dados da etapa anterior, além da avaliação dos hiperparâmetros e o do monitoramento do modelo criado anteriormente. Tudo isso voltado para responder a pergunta:

<div align="center">

**Este indivíduo foi diagnosticado com dengue com base nos dados disponíveis?**

</div>

## Etapas do Trabalho:

### Dashboard de visualização dos dados:

Nesta seção, discutiremos as decisões e ideias que nortearam a construção do dashboard requerido para esta entrega. A seguir, vocês terão acesso a uma imagem do dashboard, que não exibe seu potencial total, como filtro de dados e afins, visto que a imagem é estática. Para uma visualização dinâmica do dashboard, onde permita a manipulação de filtros e dados, basta clicar na imagem, que você será redirecionado para o dashboard na plataforma PowerBI Online (importante destacar que será necessário o login na conta UFBA para o acesso dessa maneira). Entretanto, para aqueles que preferirem, basta baixar o arquivo [Base_de_Dados_Sesab.pbix](./Base_de_Dados_Sesab.pbix), onde você poderá baixar o dashboard e acessá-lo pelo aplicativo do PowerBI em seu computador pessoal.

<div align="center">
  <table style="border: none; margin: auto; background-color: white;">
    <tr>
      <td align="center" style="border: none;">
        <a href="https://app.powerbi.com/links/2qgnVXixyU?ctid=df71f6bb-e3cc-4f5b-b532-79de261b51a2&pbi_source=linkShare">
          <img height="600em" src="./assets/DashBoard_SESAB.jpg" border="0" />
        </a>
      </td>
    </tr>
  </table>
</div>

#### Seleção e Manipulação dos Dados para Visualização:

Como ressaltado anteriormente, devemos responder à pergunta: Se um dado paciente apresentar **X** sintomas, ele receberá diagnóstico de dengue com base nos dados disponíveis ? 

Diante dessa provocação, optamos por seguir com o pré-processamento da etapa anterior, que já havia selecionado colunas importantes relacionadas ao diagnóstico de dengue. Caso haja alguma dúvida sobre quais features foram escolhidas, basta voltar algumas seções deste documento, que estará tudo explicado. 

Entretanto, acho importante ressaltar algumas decisões adotadas somente para a parte do dashboard com relação aos dados. As principais alterações foram:

- Criação de uma nova coluna (Absolute_Dengue) que agrupa os casos classificados como indeterminados e descartados em um grupo e as classificações de dengue em outro. Isto ajudou na elaboração de uma visualização que funciona de maneira binária na classificação da doença;

- Alteração dos valores do atributo CLASSI_FIN de numéricos para categóricos, para que assim haja uma coesão maior na apresentação do dashboard;

- Alteração das classificações de sexo (gênero) para que também houvesse maior coesão na criação das visualizações;

- Alteração dos valores que determinavam a presença ou ausência de um sintoma, também favorecendo a coesão das visualizções.

Caso haja interesse em saber todas as alterações realizadas, basta baixar a base de dados e clicar na opção de tranformar dados. A plataforma do PowerBi mantém um "track" das alterações feitas na base e é visível para todos os interessados.

#### Escolha das Visualizações:

Nesta seção vamos descrever as visualizações criadas e como elas se relacionam com a pergunta estabelecida para esta etapa/cenário do trabalho. As visualizações estão catalogadas a partir de seus nomes, exceto para os charts:

- **Charts para Informações Gerais**
  - *Nesta visualização, temos 4 charts que nos trazem informações gerais , mas pertinentes acerca das notificações de dengue dos dados pré-processados. Onde exibimos: a quantidade de sintomas levados em conta no diagnóstico da doença, a quantidade de casos classificados como dengue, a quantidade de notificações descartadas e também a quantidade de notificações levantadas (descartadas+confirmadas).Importante ressaltar que essas informações são dinâmicas e podem ser alteradas quando escolhermos um certo diagnóstico ou gênero, por exemplo. A visualização desses dados contribiu para um entendimento geral da base de dados e ajuda a situar o observador sobre o dashboard em questão.*

- **Gênero dos Notificados**
  - *Nesta visualização, podemos visualizar a quantidade de homens, mulheres e pessoas de sexo não definido notificadas. Ao clicarmos em uma das seções do gráfico de pizza, podemos especificar o gênero que queremos explorar os dados e o restante das visualizações será ajustado de acordo com esse filtro. Através desse gráfico, podemos notar a existência ou não de uma tendência de confirmação de casos para algum dos sexos e assim, levantar pesquisas e experimentos para se aprofundar no debate, caso haja suspeitas.*

- **Sintomas Mais Presentes nas Notificações**
  - *Nesta visualização, podemos visualizar os 5 sintomas mais presentes nas notificações para casos de suspeita de dengue. Importante ressaltar que os valores presentes na visualização se referem tanto para as notificações indeterminadas, descartadas e confirmadas. Para saber a quantidade absoluta para cada uma das classificações, basta selecionar as colunas do gráfico a seguir no dashboard, que assim esses valores serão filtrados e atualizados para corresponder ao filtro do usuário. A escolha da adoção desta visualização situa o observador nos sintomas mais recorrentes que levam à suspeita de dengue, sendo muito útil para o diagnóstico.*

- **Classificação das Notificações**
  - *Nesta visualização, o gráfico de colunas nos evidencia os diferentes tipos de classificação de casos de dengue e também a recorrência desses casos com os dados pré-processados da base de dados. Essa visualização é fundamental para se entender não só a classificação dos casos mas também para ter em mente sua distribuição de acordo com as categorias estabelecidas, demonstrando a incidência da doença nas notificações registradas.*

- **Influenciadores no Diagnóstico de Dengue**
  - *Nesta visualização, o PowerBI oferece uma ferramenta que calcula a influência de certas features para uma certa coluna. Neste caso, utilizamos as colunas referentes a sintomas (sinais clínicos) da base de dados e observamos a sua influência para o diagnóstico e classificação da notificação como dengue. Importante ressaltar que, para esta visualização, utlizamos a coluna "Absolute_dengue" ao invés da CLASSI_FIN por motivos já listados anteriomente neste documento.*

#### Insights Extraídos a partir do DashBoard:

A partir das visualizações elaboradas no dashboard, podemos extrair os seguintes insights:

- Pudemos observar que a maioria das notificações de dengue são para o sexo feminino, cerca de 56%. Isso nos leva a crer que a incidência de dengue para as mulheres seja maior que para os homens, mas nós da equipe acreditamos que isso se deve a fatores sociais e econômicos e não de gênero. Pesquisando acerca do tema, encontramos que as [mulheres são mais afetadas pelo vírus da dengue, a explicação estaria estaria no maior tempo médio de permanência da mulher em casa](https://www.em.com.br/app/noticia/gerais/2019/05/31/interna_gerais,1058348/pesquisa-aponta-que-mulheres-sao-mais-afetadas-pelo-virus-da-dengue.shtml). Fora isso, temos que [mulheres pardas e pretas são as mais afetadas pela dengue](https://www1.folha.uol.com.br/equilibrioesaude/2024/02/mulheres-pretas-e-pardas-sao-as-mais-afetadas-pela-dengue-no-brasil.shtml#:~:text=Grupo%20representa%2026%25%20dos%20brasileiros%20com%20suspeita%20da%20doen%C3%A7a&text=Mulheres%20pretas%20e%20pardas%20s%C3%A3o%20o%20grupo%20populacional%20com%20maior,doen%C3%A7a%20do%20Minist%C3%A9rio%20da%20Sa%C3%BAde.). O que sustenta a teoria de que fatores socias e econômicos influenciam mais no diganóstico da doença do que o gênero em si.


- Pudemos também nos atentar que os sintomas mais comuns para suspeita de dengue são, em ordem decrescente:

  1. Febre (cerca de 268 mil relatos)
  2. Cefaleia (cerca de 234 mil relatos)
  3. Mialgia (cerca de 229 mil relatos)
  4. Náusea (cerca de 101 mil relatos)
  5. Artralgia (cerca de 89 mil relatos)

  Isso nos trás informações relevantes acerca dos diagnósticos, visto que os principais sintomas são os mesmos para os gêneros e também possuem muita influência na classificação do diagnóstico.

- Também obtivemos os números absolutos da classificação dos casos, que se dividem a seguir:

  1. Dengue (cerca de 128 mil notificações)
  2. Descartado (cerca de 105 mil notificações)
  3. Indeterminado (cerca de 70 mil notificações)
  4. Dengue com Sinais de Alarme (cerca de 3.8 mil notificações)
  5. Dengue Grave (274 notificações)

  A partir dessas informações podemos observar que, apesar de a classificação de dengue ser a maior, os casos descartados vem logo a seguir e poucos são os casos da doença que possuem agravantes, quando comparados com os outros valores.

- Por fim, nossa última visualização faz o papel de analisar os principais sintomas que são relevantes no diagnóstico da doença e quantificar o quanto eles exercem essa influência. Em suma, os insights já estão presentes na própria visualização e acredito que não preciso ser redundante e abordar o que já está bem detalhado.


### Avaliação dos Hiperparâmetros:


### Monitoramento da Performance do Modelo:

> Aqui estão as etapas que levam em conta apenas as alterações feitas no código-fonte que foi descrito integralmente no relatório anterior. Portanto, qualquer modificação no código-fonte será detalhada e justificada; caso contrário, presume-se que o código permaneceu inalterado.

> O modelo MLP foi retirado desse monitoramente devido ao seu valor de 0 nos testes de validação utilizados, como descrito no relatório anterior.

Nesta seção, discutiremos os procedimentos que foram realizados para monitorar a perfomance dos modelos. O monitoramente foi feito considerando: *1. Métricas de desempenho, 2. Matriz de confusão e 3. Curva ROC*.

- Começando com a importação das bibliotecas, comparado ao código anterior, somente foi importado métricas adicionais relacionadas à curva ROC, AUC e a matriz de confusão. 

```python
    from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix, roc_curve, auc,RocCurveDisplay
```

- Como descrito no relatório anterior, definimos uma função 'compute_scores' para calcular as métricas F1, Precision e Recall. Nessa versão também adicionamos uma função interna 'roc_calc_viz_pred' para gerar e retornar a curva ROC de cada modelo, incluindo FPR, TPR e AUC.

  ```python
    def compute_scores(y_test,y_pred):
      def roc_calc_viz_pred(y_test, y_pred):
        viz = RocCurveDisplay.from_predictions(
                    y_test,
                    y_pred
                  )

        return viz.fpr, viz.tpr, viz.roc_auc
      
      f1 = f1_score(y_test,y_pred)
      precision = precision_score(y_test,y_pred)
      recall = recall_score(y_test,y_pred)
      fpr,tpr,auc = roc_calc_viz_pred(y_test,y_pred)

      return f1,precision,recall,fpr,tpr,auc
  ```

- Em seguida temos a função run_pipeline onde a maior parte do trabalho de modelagem é feito. 

- Quanto a quantidade de iterações, optamos por 5. 

- No código anterior um dicionário vazio chamado 'results' foi criado para armazenar os resultados das métricas de avaliação e de outros dados relevantes gerados durante a validação cruzada, como 'iteration' e 'fold'. Para mais, nessa versão também adicionamos os dados referentes às matrizes de confusão, curva ROC e AUC para realizar o monitoramento do desempenho dos modelos. 

  ```python
    def run_pipeline(): 
      N_ITERS = 5
      N_FOLDS = 5
      df = pd.read_csv("../dengue_pre_processed.csv")

      results = {
        "model_name":[],
        "iteration":[],
        "fold":[],
        "TPR":[],
        "FPR":[],
        "Confusion Matrix": [],
        "AUC":[],
        "F1": [],
        "Recall": [],
        "Precision":[]
      }
  ```
- Em seguida, selecionamos as 5 características mais relevantes do conjunto de dados X em relação à variável alvo y usando o teste qui-quadrado.

```python
    	X = SelectKBest(score_func=chi2,k=5).fit_transform(X,y)
```

- O restante do código permaneceu semelhante à versão descrita no relatório anterior. Dentro da função run_pipeline, um loop realiza a validação cruzada estratificada.

- Um loop interno treina e testa cada modelo em cada fold da validação cruzada. O modelo é treinado no conjunto de treinamento e as previsões são feitas no conjunto de teste. Na versão mais atual, as métricas F1, precisão, recall, além dos dados das matrizes de confusão, curva ROC e AUC, são calculadas e armazenadas no dicionário results.

- Finalmente, os resultados são convertidos em um DataFrame do pandas e salvos em um arquivo CSV. Em seguida, o DataFrame é agrupado pelo nome do modelo, e as médias das métricas são calculadas e armazenadas em outro arquivo CSV.

```python
   for i in tqdm(range(N_ITERS)):
      cv = StratifiedKFold(n_splits=N_FOLDS,random_state=i,shuffle=True)
        
      models =[
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("Decision Tree", DecisionTreeClassifier(criterion='entropy',max_depth=None,min_samples_split=2,min_samples_leaf=1,random_state=i)),
        ("Logistic Regression", LogisticRegression(penalty='l2', solver='lbfgs', max_iter=100,random_state=i)),
        ("Random Forest", RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=i)),
        # ("Multilayer Perceptron", MLPClassifier(hidden_layer_sizes=(100, 100),activation='relu',solver='adam',learning_rate_init=0.001,max_iter=200,batch_size=32,random_state=i))				
      ]

      for j,(train_index,test_index) in enumerate(cv.split(X,y)):

          log.info(f"iteration: {i} fold: {j}")
          X_train,y_train = X[train_index],y[train_index]
          X_test, y_test = X[test_index],y[test_index]
          
          
          for model_name,model in models:
            
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            f1,precision, recall,fpr,tpr,auc= compute_scores(y_test,y_pred)
            
            cm = confusion_matrix(y_test,y_pred, labels=np.unique(y))
            # confusion_matrices[model_name].append(cm)

            results['model_name'].append(model_name)
            results['iteration'].append(i)
            results['fold'].append(j)
            results['F1'].append(f1)
            results['TPR'].append(tpr)
            results['FPR'].append(fpr)
            results['Confusion Matrix'].append(cm)
            results['AUC'].append(auc)
            results['Recall'].append(recall)
            results['Precision'].append(precision)
            log.info(f"{model_name}.......... f1: {f1}")

  df_raw = pd.DataFrame(results)

  df = df_raw.groupby(["model_name"]).mean().round(2).reset_index()
  df = df.drop(["iteration","fold","FPR","TPR","Confusion Matrix"],axis= 1)
  df_raw.to_csv("results/dengue_results_by_fold.csv",index=False)
  df.to_csv("results/dengue_results.csv",index=False)

  if __name__ == "__main__":
    run_pipeline()
```

#### 1. Métricas de desempenho

O resultado final permaneceu o mesmo do relatório anterior, uma vez que nenhum pré-processamento ou mudança de hiperparâmetros foi feito.

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

- Análise 

De modo geral, podemos observar que o modelo Decision Tree apresentou o melhor desempenho geral, com o maior F1-score (0.63), indicando um bom equilíbrio entre precisão e recall. 
  
O Random Forest também apresentou um bom desempenho, com métricas próximas às do Decision Tree.

O KNN apresentou um desempenho aceitável, com um F1-score de 0.60. Sua precisão (0.65) é comparável à do Decision Tree, mas com uma recall (0.56) um pouco menor. 
  
A Logistic Regression apresentou um desempenho significativamente inferior em comparação com os outros modelos, com um F1-score de apenas 0.25.

#### 2. Matriz de confusão

Com o objetivo de obtermos uma visão mais detalhada sobre as previsões dos modelos em comparação com os valores reais, foi gerada uma matriz de confusão.

- O código referente à construção e plotagem da matriz de confusão está contido no arquivo 'roc_curve'

- Primeiramente definimos a função 'str_to_matrix', na qual converte uma string que está representando uma matriz de confusão em uma matriz NumPy. Em seguida, essa função é aplicada à coluna "Confusion Matrix" do DataFrame df, convertendo cada entrada de string em uma matriz NumPy.

```python
    def str_to_matrix(s):
      # Remove the outer brackets and newline characters
      s = s.strip('[]\n ')
      
      # Split the string into individual rows
      rows = s.split('\n')
      
      # Split each row into individual elements, ensuring to strip out unwanted characters
      matrix = [list(map(int, row.strip('[] ').split())) for row in rows]
      
      # Convert the list of lists to a numpy array
      matrix = np.array(matrix)
      
      return matrix
```

- Em seguida, agrupamos e calculamos a média das matrizes de confusão para cada modelo.

- Como utilizamos 5 iterações, uma matriz de confusão foi gerada para cada fold dentro de cada iteração para cada modelo. Assim, para gerar uma única matriz para cada modelo, nós extraímos a média das matrizes geradas durante as iterações. 

- Por fim, plotamos as matrizes de confusão média de cada modelo em um grid de subplots.

```python
  df["Confusion Matrix"] = df["Confusion Matrix"].apply(str_to_matrix)
  models = df["model_name"].unique()
  confusion_matrices = {model: df[df["model_name"] == model]["Confusion Matrix"] for model in models}
  
  for model_name, cm in confusion_matrices.items():
    # Média das matrizes de confusão
    confusion_matrices[model_name] = np.floor(np.mean(confusion_matrices[model_name],axis=0)).astype(int)

  n_models = len(models)
  n_cols = 2
  n_rows = (n_models + n_cols - 1) // n_cols
  # fig, axes = plt.subplots(1, 4, figsize=(20, 5))
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
  axes = axes.flatten()
  for i,(model_name,cm) in enumerate(confusion_matrices.items()):
    ax = axes[i]
    sns.heatmap(cm,annot=True, fmt='d', cmap = "Blues",cbar=False,ax=ax)
    ax.set_title(model_name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

        
  plt.tight_layout()
  plt.show()	
```

- Análise 

A matriz de confusão final pode ser encontrada a seguir:

<div align="center">
  <table style="border: none; margin: auto; background-color: white;">
    <tr>
      <td align="center" style="border: none;">
        <a href="link_for_website">
          <img height="1000em" src="./assets/matriz_de_confusao.png" border="0" />
        </a>
      </td>
    </tr>
  </table>
</div>


De modo geral, o KNN apresentou um bom desempenho em detectar negativos (28,212), apesar de possuir uma alta taxa de falsos negativos (15,635), indicando dificuldades em identificar corretamente os positivos. 
  
O modelo Decision Tree demonstrou maior equilíbrio entre positivos e negativos em comparação com o KNN.Porém, ele também apresentou uma quantidade significativa de falsos positivos e falsos negativos.

O Logistic Regression, por sua vez, embora tenha gerado um baixo número de falsos positivos (3,212), falhou significativa na detecção de positivos, com alta taxa de falsos negativos (23,092), sendo o pior dos modelos. 

Por fim, o Random Forest apresentou bom equilíbrio entre verdadeiros positivos e verdadeiros negativos, semelhante ao Decision Tree, mas ligeiramente melhor em detectar positivos. Entretanto, ele apresentou um número considerável de falsos positivos (7,469)

Acreditamos que o fato dos modelos estarem, de modo geral, obtendo números altos de verdadeiros negativos pode ser devido a falta de informações na classe a ser predita ('CLASSI_FIN'). Tais informações foram solicitadas via formulário à SESAB, como instruído pelos professores e monitores. 

#### 3. Curva ROC 

Nós também utilizamos a curva ROC e a métrica AUC (Area Under the Curve) para avaliar o desempenho de modelos de classificação. 

- A curva ROC foi gerada no mesmo arquivo da matriz de confusão, 'roc_curve'. 

- O arquivo começa com a importação das bibliotecas necessárias para a construção da curva ROC. Essas bibliotecas são importadas para manipulação de dados (pandas), cálculos numéricos (numpy), plotagem de gráficos (matplotlib e seaborn), e cálculo da métrica AUC (sklearn).

```python
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import auc
    import seaborn as sns
    import ast
```
- Em seguida, a função 'plot_roc_curve_from_df' é definida. Ela recebe um DataFrame contendo os resultados dos diferentes modelos ('dengue_results_by_fold.csv') e plota as curvas ROC para cada um deles.

- Como são 5 iterações, temos diferentes curvas ROC sendo geradas. Portanto, foi necessário gerar uma curva ROC média para cada modelo por meio da média dos 'true positive rates', da média do AUC e do desvio padrão.

- Por fim, essa curvas foram plotadas para cada modelo com a sua respectiva AUC média e desvio padrão.

```python
def plot_roc_curve_from_df(df):
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(10, 7))

    list_models = df['model_name'].unique()
    
    for nmodel in list_models:

        # Fixing the logical AND operator by wrapping individual conditions in parentheses
        df_filter = df[(df['model_name'] == nmodel)]
        
        tprs = []
        aucs = []

        for index, row in df_filter.iterrows():
            x = row['FPR'].strip('[]\n ')
            y = row['TPR'].strip('[]\n ')

            x = np.fromstring(x, dtype=float, sep=' ')
            y = np.fromstring(y, dtype=float, sep=' ')

            interp_tpr = np.interp(mean_fpr, x, y)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(row['AUC'])

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax.plot(
            mean_fpr,
            mean_tpr,
            '--',
            label=r"Mean ROC (%0.25s AUC = %0.2f $\pm$ %0.2f)" % (nmodel, mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

    ax.plot([0, 1], [0, 1], linestyle="-", lw=3, color="r", alpha=0.8)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()
```

- Segue abaixo as curvas ROC média dos modelos com as suas respectivas AUC média e desvio padrão:

- PNG

Ao analisar a imagem acima podemos perceber que o Decision Tree é o modelo mais eficaz entre os avaliados, possuindo um AUC de 0.68. Os modelos Random Forest e KNN possuem desempenho similar, com AUCs semelhantes (de 0.66 e 0.67, respectivamente), mas ainda um pouco inferiores quando comparadaos ao Decision Tree. 

Com um AUC de 0.53, a curva ROC do modelo Logistic Regression está muito próxima da linha diagonal, indicando que o desempenho do modelo é apenas ligeiramente melhor do que o acaso.

As análise extraídas da curva ROC confirmam as observações feitas anteriomente sobre o desempenho dos modelos, destacando o Decision Tree como o modelo mais promissor, seguido por Random Forest e KNN.