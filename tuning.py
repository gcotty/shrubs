import pandas as pd
from utilities.timer import time_this
from sklearn.model_selection import GridSearchCV

# One hot encoding function for categorical data
def encode(df):
	feat_objs = [col for col in df if df[col].dtype == 'object']
	feat_encs = [pd.get_dummies(df[i], prefix=i) for i in feat_objs]

	for i in feat_encs:
		df = df.join(i)
	
	for i in feat_objs:
		df = df.drop([i], axis=1)

	return df

# Run a gridsearch on a model object
@time_this
def run_gridsearch(X, y, clf, param_grid, cv=5, scorer='roc_auc'):
    
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv,
                               scoring=scorer)

    grid_search.fit(X, y)
    return  grid_search
