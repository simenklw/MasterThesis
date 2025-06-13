import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import optuna
from scipy.stats import pearsonr
import sys

from split import prep_data_before_train, random_split, subset

phenotype = sys.argv[1]
num_SNPs = int(sys.argv[2])

#Print the job
print(f"Running hyperparameter optimization for xgboost with {phenotype} with {num_SNPs} SNPs...")

data = pd.read_feather(f"data/processed/{phenotype}BV.feather")

df, y, ringnrs, mean_pheno = prep_data_before_train(data, phenotype)
del data
df.drop(columns = ["hatchisland"], inplace = True)
df["ringnr"] = ringnrs   

target = pd.DataFrame(y)
target["mean_pheno"] = mean_pheno
target["ringnr"] = ringnrs

folds = random_split(phenotype, num_folds=10, seed=42)

df = pd.merge(df,folds, on = "ringnr", how = "inner") 
df = pd.merge(df,target, on = "ringnr", how = "inner")

df = subset(df, num_snps=num_SNPs)


# Function to perform hyperparameter optimization using Optuna
def objective(trial, X_train, y_train):
    params = {
        "objective": "reg:pseudohubererror",
        "n_estimators": 600,
        "verbosity": 0, 
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30), 
        "lambda": trial.suggest_float("lambda", 1e-2, 10,log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 1,log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 1,log=True),    
    }

    model = xgb.XGBRegressor(**params, random_state = 42) 
    
    # Perform cross-validation on the 9 inner folds
    mae_list = []
    for val_fold in range(0, 10):
        if val_fold == test_fold:
            continue
        X_train_fold = X_train[X_train['fold'] != val_fold].drop(columns=['fold'])
        y_train_fold = y_train[X_train['fold'] != val_fold]
        X_val_fold = X_train[X_train['fold'] == val_fold].drop(columns=['fold'])
        y_val_fold = y_train[X_train['fold'] == val_fold]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        mae = mean_absolute_error(y_val_fold, y_pred)
        mae_list.append(mae)
        print(f'Validation Fold {val_fold} - MAE: {mae}')
    
    return sum(mae_list) / len(mae_list)

# Outer loop: Iterate through each fold for testing
outer_mae = []
outer_corr = []
output_file = f"xgb_random_split_{phenotype}.csv"

for test_fold in range(0, 10):
    # Split data into training (9 folds) and testing (1 fold)
    X_train = df[df['fold'] != test_fold].drop(columns=['ringnr', 'ID', 'mean_pheno'])
    y_train = df[df['fold'] != test_fold][['ID', 'mean_pheno']]
    
    X_test = df[df['fold'] == test_fold].drop(columns=['ringnr', 'ID', 'mean_pheno'])
    y_test = df[df['fold'] == test_fold][['ID', 'mean_pheno']]
    
    # Perform hyperparameter optimization using Optuna
    sampler = optuna.samplers.TPESampler(seed=42)   
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(lambda trial: objective(trial, X_train, y_train['ID']), n_trials=20, n_jobs=4)
    
    # Train the model with the best parameters on the 9 folds
    best_params = study.best_params
    best_model = xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=600,
        verbosity=0,
        **study.best_params,
        random_state=42
    )
    best_model.fit(X_train.drop(columns=['fold']), y_train['ID'])
    
    # Evaluate on the test fold
    y_pred = best_model.predict(X_test.drop(columns=['fold']))
    test_mae = mean_absolute_error(y_test['ID'], y_pred)
    outer_mae.append(test_mae)
    corr = pearsonr(y_test['mean_pheno'], y_pred)[0]
    outer_corr.append(corr)

    print(f'Test Fold {test_fold} - MAE: {test_mae}')
    print(f'Test Fold {test_fold} - Correlation: {corr}')

    
    with open(output_file, 'a') as f:
        f.write(f'{test_fold},{test_mae},{corr}\n')