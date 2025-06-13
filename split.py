import numpy as np
import pandas as pd

inner_system = {'28':'Indre Kvarøy',
                 '27':'Hestmannøy',
                 '26':'Gjerøy',
                 '38':'Aldra',
                 '20':'Nesøy',
                 '331':'Lurøy',
                 '332':'Onøy'} 

outer_sytem = {'24':'Selvær',
                 '23':'Træna',
                 '22':'Myken',
                 '34':'Lovund',
                 '35':'Sleneset'}
                 

all_islands = {'28':'Indre Kvarøy',
                    '27':'Hestmannøy',
                    '26':'Gjerøy',
                    '38':'Aldra',
                    '20':'Nesøy',
                    '331':'Lurøy',
                    '332':'Onøy',
                    '24':'Selvær',
                    '23':'Træna',
                    '22':'Myken',
                    '34':'Lovund',
                    '35':'Sleneset',
                    '30':'Selsøyvik',
                    '36':'Rødøy',
                    '21':'Sundøy',
                    '29':'Ytre Kvarøy'}


def island_split(pheno:str = "mass") -> pd.DataFrame:

    data = pd.read_feather(f"data/processed/{pheno}BV.feather")  
    data = data[["ringnr", "hatchisland"]]

    #proceed with only the islands where the values count for hatchisland is greater than 10
    island_counts = data.hatchisland.value_counts()
    island_counts = island_counts[island_counts > 10]

    folds = pd.DataFrame(columns = ["ringnr", "fold"])
    
    
    for i, island in enumerate(island_counts.index):
        fold = data[data.hatchisland == island]["ringnr"]
        fold = pd.DataFrame(fold)
        fold["fold"] = i
        folds = pd.concat([folds, fold], axis = 0)

    #    fold_name = f"island_fold_{i}_{pheno}.csv"
    #    fold.to_csv(f"data/interim/folds/{fold_name}", index=False)
    #    print(f"Island {island} has {len(fold)} samples + {i}")

    return folds

def random_split(pheno:str = "mass", num_folds:int = 5, seed: int = 42) -> pd.DataFrame:
    data = pd.read_feather(f"data/processed/{pheno}BV.feather")  
    x = data[["ringnr","hatchisland"]]
    del data
    np.random.seed(seed)
    #Split the data into num_folds folds with random sampling, each fold should have the same number of individuals
    #The folds should be saved in a new column called "fold" in the data frame

    #x = x.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Number of folds
    n_folds = num_folds

    # Create an array with fold numbers
    folds = np.array([i % n_folds for i in range(len(x))])


    # Shuffle the fold assignment array
    np.random.shuffle(folds)

    # Assign the fold numbers to a new column in the dataframe
    x['fold'] = folds
    
    return x

    #for i in range(n_folds):
    #    fold = x[x["fold"] == i]
    #    fold = fold["ringnr"]
    #    fold.to_csv(f"data/interim/folds/random_{n_folds}fold_{i+1}_{pheno}.csv", index=False)



def prep_data_before_train(data: pd.DataFrame, phenotype: str) -> tuple:
    """
    prepare data for training, returns target vector and covariate-matrix and ringnrs for grouping
    :param data: all data from dataloader script
    :param phenotype: the phenotype to be predicted
    :return: X, Y, ringnrs, X contain covariates and Y contains the phenotype
    """
    ringnrs = data.ringnr
    mean_pheno = data.mean_pheno
    X = data.drop(
        columns=[
            "ID",
            "mass",
            "tarsus",
            "ringnr",
            "mean_pheno",
            "IID",
            "FID",
            "MAT",
            "PAT",
            "SEX",
            "PHENOTYPE",
        ],
        errors="ignore",
    )

    try:
        Y = data.loc[:, phenotype]
    except KeyError:
        try:
            Y = data.ID
        except AttributeError:
            Y = data.mass
    del data
    try:
        X.hatchyear = X.hatchyear.astype("int")
        X.island_current = X.island_current.astype("int")
    except AttributeError:
        pass
    Y = (Y - np.mean(Y))/np.std(Y)
    X = X.fillna(0)
    X = X.T
    X = X.astype("int")
    X = X.T 
    return X, Y, ringnrs, mean_pheno

def subset(X:pd.DataFrame, seed:int = 42, num_snps:int = 20000) -> pd.DataFrame:
    """
    Function to subset the data into a smaller dataset for testing purposes, 
    sampling only a subset of the features
    """
    np.random.seed(seed)
    sample_columns = np.random.choice(X.columns[:-5], num_snps, replace = False)
    sample_columns = np.append(sample_columns, ["ringnr","fold", "ID", "mean_pheno"])

    return X[sample_columns]


def sort_snps(X):
    """
    Input: X, numpy ndarray with names of snps
    """
    #Load Helgeland_01_2018.map
    df = pd.read_csv('data/Helgeland_01_2018.map', sep='\t', header=None)
    df.columns = ["chr", "snpID", "value", "bp"]
    #discard snps that should be discarded
    df = df[~df.chr.isin([0,30,32])].reset_index(drop=True)

    snp_df = pd.DataFrame({
	'base': X,
	'snpID': [col.split("_")[0] for col in X]
    })

    map = pd.merge(snp_df, df, on='snpID', how='inner').sort_values(by=['chr','bp']).reset_index(drop=True)

    return map

    

#Remove islands with less than 10 (or some other number) observations
def filter_islands(df, threshold):
    island_count = df.hatchisland.value_counts() 
    islands = island_count[island_count > threshold].index
    return df[df.hatchisland.isin(islands)]


