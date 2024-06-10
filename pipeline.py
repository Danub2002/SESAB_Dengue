import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix, roc_curve, auc,RocCurveDisplay
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

from tqdm import tqdm

from loguru import logger as log

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

	X,y = df.drop("CLASSI_FIN",axis=1).to_numpy(),df["CLASSI_FIN"].to_numpy()

	# Create the object for SelectKBest and fit and transform the classification data
	# k is the number of features you want to select [here it's 2]
	X = SelectKBest(score_func=chi2,k=5).fit_transform(X,y)

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
			
			