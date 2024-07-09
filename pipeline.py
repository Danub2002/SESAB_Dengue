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
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from loguru import logger as log
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

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
	N_ITERS = 1
	N_FOLDS = 5
	df = pd.read_csv("../dengue_pre_processed.csv")

	param_grid = {
		'KNN': {
			'n_neighbors':np.arange(1, 25, 1), 
			'weights':np.array(['uniform', 'distance']),
			'p':np.arange(1, 4, 1)
		},
		'Decision_Tree': {
			'max_depth':np.arange(1, 25, 1), 
			'criterion':np.array(['gini', 'entropy']),
			'min_samples_leaf':np.arange(1, 25, 1)
		},
		'Logistic_Regression': {
			'penalty': ['l1', 'l2', 'elasticnet', 'none'],
			'C': [0.01, 0.1, 1, 10, 100],
			'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
			'max_iter': [100, 200, 300, 500]
		},
		'Random_Forest': {
			'n_estimators':np.arange(1, 25, 1), 
			'max_depth':np.arange(1, 25, 1)
		}
	}
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

	model_best_score = {
		"KNN": 0,
		"Decision_Tree": 0,
		"Logistic_Regression": 0,
		"Random_Forest": 0
	}
	for i in tqdm(range(N_ITERS)):
		cv = StratifiedKFold(n_splits=N_FOLDS,random_state=i,shuffle=True)
			
		models =[
			("KNN", KNeighborsClassifier()),
			("Decision_Tree", DecisionTreeClassifier(random_state=i)),
			("Logistic_Regression", LogisticRegression(random_state=i)),
			("Random_Forest", RandomForestClassifier(random_state=i)),
			# ("Multilayer Perceptron", MLPClassifier(hidden_layer_sizes=(100, 100),activation='relu',solver='adam',learning_rate_init=0.001,max_iter=200,batch_size=32,random_state=i))				
		]

		for j,(train_index,test_index) in enumerate(cv.split(X,y)):

			log.info(f"iteration: {i} fold: {j}")
			X_train,y_train = X[train_index],y[train_index]
			X_test, y_test = X[test_index],y[test_index]
			
			
			for model_name,model in models:
				RS_cv = StratifiedKFold(n_splits=N_FOLDS,random_state=i,shuffle=True)
				clf= RandomizedSearchCV(estimator=model, param_distributions=param_grid[model_name], cv=RS_cv, scoring='f1',n_jobs=-1,random_state=i,refit=True)
				clf.fit(X_train, y_train)
				best_model = clf.best_estimator_
				y_pred = best_model.predict(X_test)
				# model.fit(X_train,y_train)
				# y_pred = model.predict(X_test)

				f1,precision, recall,fpr,tpr,auc= compute_scores(y_test,y_pred)
				
				if f1 > model_best_score[model_name]:
					model_best_score[model_name] = f1
					save_model(model, f'models/{model_name}_best_model.pkl')

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
			
			