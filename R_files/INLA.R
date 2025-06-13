#######################################################################
## Stefanie Muff, September 2023
## 
## Helgeland house sparrow data analysis
## Make predictions using Machine Learning (deep learning) 
## In addition, at the end we give code for the genomics-based animal model fitted with INLA
## !! This requires that you are using R version 4.2 or higher (ideally even 4.3) !!
#######################################################################
# Packages needed for the script to run:
library(nadiv)
library(pedigree)
library(INLA)
library(MASS)
library(MCMCpack)
library(MCMCglmm)
# This is a self-made package that I send you to install locally:
library(SMisc)

library(dplyr)

library(keras)
library(tensorflow)

## Old packages no longer needed (but kept as comments, in case we need them in the future...)
# library(MasterBayes)
# library(pedigreemm)
# library(bdsmatrix)
# library(irlba)
# library(RSpectra)
# library(dplyr)

# Data preparation helper script:
source("h_dataPrep.r")

# Some data wranging to ensure that the IDs in the data correspond to the IDs in the A and G-matrices (nothing to worry about):
# indicates that some IDs are missing:
d.map[3110:3125,] 
# from this we see the number of anmals
Nanimals <- 3116

phenotype <- "mass"

# remove missing values
d.morph <- filter(d.morph, !is.na(eval(as.symbol(phenotype))))

# In the reduced pedigree only Nanimals out of the 3147 IDs are preset. 
d.map$IDC <- 1:nrow(d.map)

d.morph$IDC <- d.map[match(d.morph$ringnr, d.map$ringnr), "IDC"]

### Prepare for use in INLA - 
d.morph$IDC4 <- d.morph$IDC3 <- d.morph$IDC2 <- d.morph$IDC





##################################################################
### Run INLA based on the GBLUP approach
###
### To this end, use the GRM (genetic relatedness matrix) in the animal model
### 
### !!! This is very slow - account for at least 20-30min waiting time before inla terminates !!!
##################################################################

# Relatedness matrix from Henrik (vanRaden method 1) where +0.01 was already added to diagnoal!
d.Gmatrix <- read.table("data/gghatvr3.triangle.g", header = F, sep=" ")

# keep only relatednesses that are relevant for animals in d.morph
#d.Gmatrix <- d.Gmatrix[d.Gmatrix[,1] %in% d.morph$ID & d.Gmatrix[,2] %in% d.morph$ID, ]

# G is a sparse matrix object. We can also verify that it is symmetric (sometimes some numerical problems lead to non-symmetry)
G <- sparseMatrix(i=d.Gmatrix[,1],j=d.Gmatrix[,2],x=d.Gmatrix[,3],symmetric=T) 
G[,] <- as.numeric(G[,])
isSymmetric(G)

# Again extract the rows and columns for the individuals in the data set that we analyse
GG <- G[d.map[1:3116,3],d.map[1:3116,3]]

# To ensure that the matrix is positive definite, we do a computational trick (proposed by vanRaden 2008, see https://doi.org/10.3168/jds.2007-0980 :)
GGG  <- GG*0.99 + 0.01

# Need to derive the inverse to give to INLA
Cmatrix <- solve(GGG)
if (!isSymmetric(Cmatrix)){
  Cmatrix <- forceSymmetric(Cmatrix)
}

## 
## INLA formula
## 
#Here we use body mass as the response, and some fixed and random effects:
formula.mass = eval(as.symbol(phenotype)) ~ sex + FGRM + month + age +   outer + other +
  f(hatchisland, model = "iid", hyper = list(
    prec= list(initial = log(1), prior = "pc.prec", param = c(1,0.05))
  )) + 
  f(hatchyear,model="iid",hyper=list(
    prec=list(initial=log(1), prior="pc.prec",param=c(1,0.05))
  ))+
  f(IDC,model="iid",hyper=list(
    prec=list(initial=log(1), prior="pc.prec",param=c(1,0.05))
  )) +
  f(IDC2,values=1:3116,model="generic0",
    Cmatrix=Cmatrix,
    constr = TRUE,
    hyper=list(
      # The priors are relevant, need to discuss
      prec=list(initial=log(0.5), prior="pc.prec",param=c(sqrt(2),0.05))
    ))



####################################################################
#load data and split into folds
####################################################################

fold_data <- data.frame(ringnr = character(), fold = integer())

for (i in 0:7){
  data_from_file <- read.csv(paste("data/interim/folds/island_fold_",i,"_",phenotype,".csv",sep = ""))
  temp_df <- data.frame(ringnr = data_from_file$ringnr, fold = i)
  
  #append to the fold_data
  fold_data <- rbind(fold_data, temp_df)
}



corr_cv_EG <- c()
corr_cv <- c()


########################################################
#Everthing under here must go in a loop later
########################################################
for (i in 0:7){
  ringnr_test <- fold_data[fold_data$fold == i, "ringnr"]
  ringnr_train <- fold_data[fold_data$fold !=i, "ringnr"]
  
  
  #make train and test split
  d.morph_train <- filter(d.morph, !ringnr %in% ringnr_test)
  d.morph_test <- filter(d.morph, ringnr %in% ringnr_test)
  
  
  n_train <- dim(d.morph_train)[1]
  n_test <- dim(d.morph_test)[1]
  N <- n_train + n_test
  
  
  #Saving the phenotypic value in the test set, both for phenotype and breeding value 
  pheno_test_EG <- d.morph_test[,phenotype]
  pheno_test <- as.data.frame(d.morph_test %>% 
                                group_by(ringnr) %>% 
                                summarize(
                                  mean_pheno = mean(eval(as.symbol(phenotype)))
                                ))[,"mean_pheno"]
  
  #INLA does not have a predict function, so have to fill the test-values with NAs and merge it back into the training set.
  d.morph_test[, phenotype] <- NA
  d.morph_train <- union_all(d.morph_train, d.morph_test)
  
  
  names(d.morph_train)
  #All individuals
  idxs <- 1:Nanimals
  #get the indices corresponding to the individuals in the test set.
  idxs_test <- which(d.map$ringnr %in% unique(ringnr_test))
  
  
  
  
  cat("STARTING INLA\n")
  
  model1.mass = inla(
      formula=formula.mass, 
      family="gaussian",
      data=d.morph_train,
      control.family=list(hyper = list(theta = list(initial=log(0.5),  prior="pc.prec",param=c(sqrt(2),0.05)))),
      control.compute=list(dic=F, return.marginals=FALSE), verbose = TRUE
  )
  
  cat("INLA DONE")
  
  
  #get predicted values
  preds_EG <- model1.mass$summary.fitted.values$mean[(n_train + 1):N]
  preds_BV_mode <- model1.mass$summary.random$IDC2$mode[idxs_test]
  preds_BV_mean <- model1.mass$summary.random$IDC2$mean[idxs_test]
  
  
  #calculate matrics:
  corr_EG <- cor(preds_EG, pheno_test_EG, method = "pearson")
  corr_BV_mode <- cor(preds_BV_mode, pheno_test, method = "pearson")
  corr_BV_mean <- cor(preds_BV_mean, pheno_test, method = "pearson")
  
  
  
  file_path <- paste("INLA_",phenotype,"_island_split.csv", sep = "")
  
  if (!file.exists(file_path)){
    write.csv(data.frame(corr_EG, corr_BV_mode, corr_BV_mean), file_path, row.names = FALSE)
  } else {
    write.table(data.frame(corr_EG, corr_BV_mode, corr_BV_mean), file_path, col.names = FALSE, row.names = FALSE, append = TRUE)
  }
  
}


























