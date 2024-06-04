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

def compute_scores(y_test,y_pred):
	f1 = f1_score(y_test,y_pred)
	precision = precision_score(y_test,y_pred)
	recall = recall_score(y_test,y_pred)

	return f1,precision,recall

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

	X,y = df.drop("CLASSI_FIN",axis=1).to_numpy(),df["CLASSI_FIN"].to_numpy()
	

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
			
			