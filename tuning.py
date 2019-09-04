import pandas as pd
from utilities.util import isin_row
from utilities.timer import time_this
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.utils import resample

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

@time_this
def run_bootstrap(ratio, n_iter, df, target, clf):
	n = int(len(df) * ratio)
	c_matx = list()
	models = list()

	for i in range(n_iter):
		# Prep test/train splits
		train = resample(df, n_samples=n)
		X_train = train.drop([target], axis=1)
		y_train = train[target]

		test = df[~isin_row(df, train)]
		X_test = test.drop([target], axis=1)
		y_test = test[target]

		# Train the model
		clf.fit(X_train, y_train)

		# Results (defaults to precision_score)
		preds = clf.predict(X_test)
		print("Iteration {}: {}".format(i, precision_score(y_test, preds))
		
		c_matx.append(confusion_matrix(y_test, preds))
		models.append(clf)

		return models, c_matx 

# Save model object
def save_model(clf, name='clf.sav'):
	pickle.dump(clf, open(name, 'wb'))
	print('model object saved as {}!'.format(name))
	
	# To reopen the object, use:
	# clf = pickle.load(open(fname, 'rb'))
