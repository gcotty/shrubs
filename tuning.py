import pandas as pd

# One hot encoding function for categorical data
def encode(df):
	feat_objs = [col for col in df if df[col].dtype == 'object']
	feat_encs = [pd.get_dummies(df[i], prefix=i) for i in feat_objs]

	for i in feat_encs:
		df = df.join(i)
	
	for i in feat_objs:
		df = df.drop([i], axis=1)

	return df

