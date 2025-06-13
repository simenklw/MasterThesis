import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, BayesianRidge
from split import prep_data_before_train, random_split, subset, filter_islands
from scipy.stats import pearsonr
import seaborn as sns


pheno = "wing"

data = pd.read_feather(f"data/processed/{pheno}BV.feather")

X, y, ringnrs, mean_pheno = prep_data_before_train(data, pheno)
del data
X.drop(columns = ["hatchisland"], inplace = True)
X["ringnr"] = ringnrs   

target = pd.DataFrame(y)
target["mean_pheno"] = mean_pheno
target["ringnr"] = ringnrs

folds = random_split(pheno, num_folds=10, seed=42)

X = pd.merge(X,folds, on = "ringnr", how = "inner") 
X = pd.merge(X,target, on = "ringnr", how = "inner")

X = filter_islands(X, 10)

X_train = X[X["fold"]!=0].drop(columns=["ringnr", "ID", "mean_pheno","fold","hatchisland"])
y_train = X[X["fold"]!=0][["ID","mean_pheno"]]

X_test = X[X["fold"]==0].drop(columns=["ringnr", "ID", "mean_pheno","fold", "hatchisland"])
y_test = X[X["fold"]==0][["ID","mean_pheno"]]


model = Ridge(alpha=54_555) #Regularization parameter
model.fit(X_train, y_train["ID"])
y_pred = model.predict(X_test)
pearson = pearsonr(y_test["mean_pheno"], y_pred)
print(pearson[0])