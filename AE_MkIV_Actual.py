import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras import layers, Model
from split import prep_data_before_train, random_split, subset, filter_islands, sort_snps
import csv
import matplotlib.pyplot as plt 

#For regression
from sklearn.model_selection import KFold 
import catboost as cb
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

############################################################
################     Preprocessing      ####################
############################################################


data = pd.read_feather("data/processed/massBV.feather")

X, y, ringnrs, mean_pheno = prep_data_before_train(data, "mass")
del data
X.drop(columns = ["hatchisland"], inplace = True)
X["ringnr"] = ringnrs   

target = pd.DataFrame(y)
target["mean_pheno"] = mean_pheno
target["ringnr"] = ringnrs

folds = random_split("mass", num_folds=5, seed=42)

X = pd.merge(X,folds, on = "ringnr", how = "inner") 
X = pd.merge(X,target, on = "ringnr", how = "inner")  


df = filter_islands(X, 10)
X_subset = subset(df, num_snps=64000).drop(columns = ["ringnr", "ID", "mean_pheno","fold"])
sorted_snps = sort_snps(X_subset.columns.values)
sorted = sorted_snps.base.to_list()
not_sorted = set(X_subset.columns) - set(sorted_snps.base)

#This is the input that goes to the autoencoder:
X_subset = np.array(X_subset[sorted + list(not_sorted)])
#X_subset = np.array(X_subset)

############################################################
################   AUTOENCODER Mk-IV    ####################
############################################################


##################################################
# 1. (Example) Data Loading & Preprocessing
##################################################
def normalize_genotypes(X):
    """
    Mimics 'genotypewise01' in the GenoCAE readme:
    Map genotype 0->0.0, 1->0.5, 2->1.0.
    If missing (-1), we'll impute as 1 => 0.5 after scaling.
    """
    X_copy = X.copy()
    # Impute missing as 1 (heterozygous)
    X_copy[X_copy < 0] = 1  
    # Scale from {0,1,2} => {0.0,0.5,1.0}
    X_copy = X_copy / 2.0
    return X_copy


##################################################
# 2. Residual Block (as used in M1)
##################################################
class ResidualBlock(layers.Layer):
    """
    A simplified version of 'ResidualBlock2' that
    does: Conv1D -> BN -> ELU -> Conv1D -> BN -> + shortcut -> ELU
    """
    def __init__(self, filters, kernel_size=5):
        super().__init__()
        self.conv1 = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                   padding='same', activation='elu')
        self.bn1   = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                   padding='same', activation=None)
        self.bn2   = layers.BatchNormalization()
        self.elu   = layers.Activation('elu')

    def call(self, x, training=False):
        shortcut = x
        x = self.conv1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.bn2(x, training=training)
        x = x + shortcut
        x = self.elu(x)
        return x


##################################################
# 3. Build Model
##################################################
def build_mk4_autoencoder(input_length, filter:int=8, dropout_rate:float=0.01, dense_layers:int=75, latent_layer:int=20):
    """
    Roughly replicates the M1.json sequence of layers, minus
    marker-specific variables & dynamic shape logic.

    The final 'encoded' layer is 20D. We'll flatten / unflatten around it.
    """
    inputs = layers.Input(shape=(input_length, 1), name="input_genotypes")

    # -------------------- Encoder --------------------
    x = layers.Conv1D(filters=filter, kernel_size=5, padding='same',
                      activation='elu', strides=1)(inputs)
    x = layers.BatchNormalization()(x)
    x = ResidualBlock(filters=filter, kernel_size=5)(x)
    x = layers.MaxPool1D(pool_size=5, strides=2, padding="same")(x)

    x = layers.Conv1D(filters=filter, kernel_size=5, padding='same',
                      activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)

    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_layers)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_layers, activation='elu')(x)

    # Encoded: 20D (like M1.json)
    encoded = layers.Dense(latent_layer, activation=None, name="encoded")(x)

    # -------------------- Decoder --------------------
    # We'll guess a shape for reconstruction
    x = layers.Dense(dense_layers, activation='elu')(encoded)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_layers, activation='elu')(x)
    x = layers.Dropout(dropout_rate)(x)

    reconstruction_size = (input_length // 2) * 8
    x = layers.Dense(reconstruction_size, activation=None)(x)

    x = layers.Reshape((input_length // 2, 8))(x)
    x = layers.Conv1D(filters=filter, kernel_size=5, padding='same', activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D(size=2)(x)

    x = ResidualBlock(filters=filter, kernel_size=5)(x)
    x = layers.Conv1D(filters=filter, kernel_size=5, padding='same',
                      activation='elu', name="nms")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=1, kernel_size=1, padding='same')(x)

    outputs = layers.Flatten(name="logits")(x)
    outputs = layers.Reshape((input_length, 1), name="reconstructed")(outputs)

    autoencoder = Model(inputs, outputs, name="mk4_autoencoder")
    return autoencoder


##################################################
# 4. Helper Functions to Compute Concordance
##################################################
def decode_genotypes(y_pred, threshold_low=0.33, threshold_high=0.66):
    """
    Convert continuous outputs in [0..1] back to {0.0, 0.5, 1.0}.
    By default:
      - x < threshold_low => 0.0
      - threshold_low <= x < threshold_high => 0.5
      - x >= threshold_high => 1.0
    """
    # y_pred is shape (N, M, 1)
    y_pred_flat = y_pred[..., 0]  # shape (N, M)
    out = np.zeros_like(y_pred_flat)
    out[y_pred_flat >= threshold_high] = 1.0
    mid_mask = (y_pred_flat >= threshold_low) & (y_pred_flat < threshold_high)
    out[mid_mask] = 0.5
    return out

def compute_genotype_concordance(y_true, y_pred_decoded):
    """
    y_true, y_pred_decoded each shape (N, M),
    in {0.0, 0.5, 1.0}.
    Returns fraction of positions that match exactly.
    """
    matches = (y_true == y_pred_decoded)
    return np.mean(matches)

def compute_baseline_concordance(y_true):
    """
    Finds the single most common genotype (0.0, 0.5, or 1.0) across
    all samples & SNPs, then measures how often it appears in y_true.
    y_true shape (N, M).
    """
    # Flatten to 1D
    y_flat = y_true.flatten()
    # Count each genotype's frequency
    counts = {}
    for g in [0.0, 0.5, 1.0]:
        counts[g] = np.sum(y_flat == g)

    # Which genotype is most common?
    best_geno = max(counts, key=counts.get)  # 0.0, 0.5, or 1.0
    best_geno_count = counts[best_geno]

    # Baseline is fraction of matches if we guess best_geno always
    baseline_acc = best_geno_count / len(y_flat)
    return baseline_acc, best_geno


##################################################
# 5. Putting it All Together
##################################################
if __name__ == "__main__":
    # Suppose X_subset is your SNP array of shape (N, M)
    # with values in {0,1,2} or -1 for missing.
    # For demonstration, let's pretend we have:
    # X = np.random.choice([0,1,2], size=(2000, 1000))
    # but here we assume you already have X_subset:
    num_samples = X_subset.shape[0]
    num_snps = X_subset.shape[1]
    X = X_subset

    # 1) Normalize
    X_norm = normalize_genotypes(X)           # => in [0,1]
    # shape => (N, M, 1)
    X_norm = np.expand_dims(X_norm, axis=-1)


    #Save results in dictionary
    results = []

    for lr in [5e-4]: 
        for dr in [0.1]: 
            for f in [8]:
                for ll in [10, 20, 30, 40, 50]:    

                    # For Python
                    random.seed(42)
                    # For NumPy
                    np.random.seed(42)
                    # For TensorFlow
                    tf.random.set_seed(42)

                    # 2) Build & Compile
                    model = build_mk4_autoencoder(num_snps, filter=f, dropout_rate=dr, dense_layers=75, latent_layer=ll)
                    model.summary()
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss='mse')


                    # 3) Train/Val Split
                    split_idx = int(num_samples * 0.9)
                    X_train = X_norm[:split_idx]
                    X_val   = X_norm[split_idx:]

                    # 4) Fit
                    history = model.fit(
                        X_train, X_train,
                        validation_data=(X_val, X_val),
                        epochs=50,
                        batch_size=30
                    )


                    # 5) Extract latent space encodings
                    #    This is the "encoded" layer from the autoencoder
                    encoder = Model(inputs=model.input, 
                                    outputs=model.get_layer("encoded").output)
                    embeddings = encoder.predict(X_norm)
                    print("Embeddings shape:", embeddings.shape)


                    # 6) Predict on Entire Dataset
                    recons = model.predict(X_norm)  # shape (N, M, 1)

                    # Convert reconstructions back to {0,0.5,1} calls
                    recons_decoded = decode_genotypes(recons)

                    # Convert X_norm to shape (N,M) for comparison
                    X_norm_2d = X_norm[..., 0]

                    # 7) Compute Genotype Concordance
                    model_gc = compute_genotype_concordance(X_norm_2d, recons_decoded)
                    print(f"Autoencoder Genotype Concordance = {model_gc:.4f}")

                    # 8) Baseline Concordance
                    baseline_acc, best_geno = compute_baseline_concordance(X_norm_2d)
                    print(f"Baseline Concordance = {baseline_acc:.4f}, best genotype = {best_geno}")

                    # Compare
                    improvement = model_gc - baseline_acc
                    print(f"Improvement over baseline = {improvement:.4f}")


                    #Save embeddings for later:
                    np.savetxt(f"Embeddings/embedding_{num_snps}_{ll}.csv", embeddings, delimiter=",")


                ############################################################
                ################   CATBOOST REGRESSION    ##################
                ############################################################

                    do_reg = False

                    if do_reg:
                        #Set the target variable together with the embedding from umap. The target variable is the ID column in df.
                        X2 = embeddings
                        y2 = df[["ID","mean_pheno"]]


                        # Perform 5-fold cross-validation
                        kf = KFold(n_splits=10, shuffle=True, random_state=42)
                        all_pearsonr = []

                        for train_index, test_index in kf.split(X2):
                            X_train2, X_test2 = X2[train_index], X2[test_index]
                            y_train2, y_test2 = y2.iloc[train_index], y2.iloc[test_index]

                            train2 = cb.Pool(X_train2, y_train2["ID"])
                            test2 = cb.Pool(X_test2, y_test2["ID"])

                            model2 = cb.CatBoostRegressor(loss_function="Huber:delta=1", random_state=42)

                            model2.fit(train2, eval_set=test2, verbose=0)

                            y_pred2 = model2.predict(test2)
                            pearson_corr = pearsonr(y_test2["mean_pheno"], y_pred2)[0]
                            all_pearsonr.append(pearson_corr)
                            print(f"Fold Pearson correlation: {pearson_corr}")

                        print(f"Mean Pearson correlation over 10 folds: {np.mean(all_pearsonr)}")

                        results.append([lr,dr,f,np.mean(all_pearsonr)])
                        #Write results to file
                        # 1) Overwrite the CSV with the new, bigger results list
                        with open("my_results.csv", "w", newline="") as f:
                            writer = csv.writer(f)

                            # Write each row in `results`
                            for row in results:
                                writer.writerow(row)

                    else:
                        continue

