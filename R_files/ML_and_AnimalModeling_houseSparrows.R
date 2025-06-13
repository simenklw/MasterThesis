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


# In the reduced pedigree only Nanimals out of the 3147 IDs are preset. 
d.map$IDC <- 1:nrow(d.map)

d.morph$IDC <- d.map[match(d.morph$ringnr, d.map$ringnr), "IDC"]

### Prepare for use in INLA - 
d.morph$IDC4 <- d.morph$IDC3 <- d.morph$IDC2 <- d.morph$IDC




##############################################
### Preparations: Start with body mass and extract the id-value for each individual  
##############################################

# keep all mass records that are not NA
d.mass <- d.morph[!is.na(d.morph$mass),c("ringnr","mass")]
names(d.mass) <- c("ringnr","mass")

d.mean.mass <- as.data.frame(d.mass %>%
                               group_by(ringnr) %>%
                               summarize(mean_mass = mean(mass)))

formula.mass.lmm = mass ~   sex + FGRM + month + age +  outer + other +  
  (1 | island_current) +
  (1 | hatchyear) +
  (1 | ringnr)


library(lme4)
dd <- d.morph[!is.na(d.morph$mass),]
r.mass.lmer <- lmer(formula.mass.lmm, data = dd)

# Residuals
d.mass.res <- data.frame(ringnr = dd$ringnr, mass_res =residuals(r.mass.lmer))

# Mean over repeats for each individual
d.mean.mass.res <- as.data.frame(d.mass.res %>%
                                   group_by(ringnr) %>%
                                   summarize(mean_mass_res = mean(mass_res)))

# ID effect
d.ID.mass <- data.frame(ringnr = d.mean.mass[,1],ID.mass= ranef(r.mass.lmer)$ringnr)


# We take as the new phenotype the estimated ID effect:
d.ID.mass <- data.frame(ringnr=d.mean.mass[,1],ID = d.ID.mass[,2], mean_pheno = d.mean.mass$mean_mass)

tmp_hatchisland_ringnr_df <- d.morph[, c("ringnr", "hatchisland")]
tmp_hatchisland_ringnr_df <- tmp_hatchisland_ringnr_df %>%
  distinct(ringnr, .keep_all = T)
d.ID.mass <- d.ID.mass %>%
  merge(tmp_hatchisland_ringnr_df, by = "ringnr")

# ## This was the OLD way - I don't think it should be done:
# # We take as the new phenotype the sum of the ID effect and the mean of the residual for each individual: 
# d.ID.res.mass <- data.frame(ringnr=d.mean.mass[,1],sum.ID.res = d.ID.mass[,2]+d.mean.mass.res[,2])

#############################################################
### Now we also load the raw SNP data matrix
#############################################################
library(data.table)
no_snps <- 20000

# Using the quality-controlled SNP matrix from Kenneth:
SNP.matrix <- data.frame(fread("data/Helgeland_01_2018_QC.raw"))
#SNP.matrix <- data.frame(fread("data/full_imputed_dosage.raw"))

names(SNP.matrix)[2] <- "ringnr"

set.seed(323422)
SNP.matrix.reduced <- cbind(SNP.matrix[,1:6],
                            (SNP.matrix[,sort(sample(7:181369,no_snps,replace=FALSE))])
)


# Generate a data frame where individuals with ring numbers from d.ID.res.mass are contained, as well as the phenotype (here the residuals from the lmer analysis with mass as response)
d.dat <- merge(d.ID.mass[,c("ringnr","ID","mean_pheno","hatchisland")],SNP.matrix,by="ringnr")
#FINISH!!!!!!!!!!!!

#############################################################
### Run keras with the SNPs as input and the phenotypes (ID+residuals) as response
#############################################################
nn <- nrow(d.dat)

# Scale the part of d.dat that contains the snp information
SNP_SNP <- scale(d.dat[,8:(no_snps+7)])

SNP_SNP[is.na(SNP_SNP)] = 0


# Select training subset
set.seed(1234)
nn_train <- sample(1:nn,ceiling(nn*0.9),replace=FALSE)



x_train <- SNP_SNP[nn_train,]
y_train <- d.dat$ID[nn_train]

x_test <- SNP_SNP[-nn_train,]
y_test <- d.dat$ID[-nn_train]


xtrain = array(x_train, dim = c(nrow(x_train), no_snps, 1))
xtest = array(x_test, dim = c(nrow(x_test), no_snps, 1))

ytrain = array(y_train, dim = c(length(y_train),  1))
ytest = array(y_test, dim = c(length(y_test),  1))

# Define the model

model = keras_model_sequential() %>%
  layer_conv_1d(filters = 32,
                kernel_size = 1,
                activation = 'relu',
                input_shape = c(no_snps,1)) %>% 
  layer_dropout(rate = 0.2) %>%
  layer_conv_1d(filters = 32,
                kernel_size = 8,
                activation = 'relu',
                input_shape = c(no_snps,1)) %>% 
  layer_dropout(rate = 0.2) %>%
  # # layer_conv_1d(filters = 32,
  # #               kernel_size = 3,
  # #               activation = 'relu',
  # #               input_shape = c(no_snps,1)) %>% 
  # layer_dropout(rate = 0.20) %>%
  # layer_max_pooling_1d() %>% 
  layer_flatten() %>% 
  #layer_dense(units = 256, activation = "relu") %>%
  #layer_dropout(rate = 0.1) %>% 
  #layer_dense(units = 128, activation = "relu",kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  #layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'linear')

model %>% compile(
  optimizer = "RMSprop",
  loss = "mse",
  metrics = c("mae")
)

# Train the model 

history = model %>% fit(xtrain, ytrain,
                        epochs = 20,
                        batch_size = 256,
                        validation_split = 0.2)

#plot(history)

# Evaluate the model

test_prediction <- model %>% evaluate(xtest,ytest)

data.frame(y = predict(model, xtest))

cor(y_test,data.frame(y = predict(model, (xtest)))[,1])


################################################################
## Cross-validation
################################################################

nn <- nrow(d.dat)
set.seed(123)
t.perm <- sample(1:nn)


cc = 10 
d.acc.mass.ml <- rep(NA,cc)

for (ii in 1:cc){
  nn_out <- t.perm[round(((ii-1)*nn/cc +1),0) : round((ii*nn/cc),0)]
  
  x_train <-  SNP_SNP[-nn_out,]
  y_train <-  d.dat$sum.ID.res[-nn_out]
  x_test <-  SNP_SNP[nn_out,]
  y_test <-  d.dat$sum.ID.res[nn_out]
  
  xtrain = array(x_train, dim = c(nrow(x_train), no_snps, 1))
  xtest = array(x_test, dim = c(nrow(x_test), no_snps, 1))
  
  ytrain = array(y_train, dim = c(length(y_train),  1))
  ytest = array(y_test, dim = c(length(y_test),  1))
  
  model = keras_model_sequential() %>%
    layer_conv_1d(filters = 50,
                  kernel_size = 4,
                  activation = 'relu',
                  input_shape = c(no_snps,1)) %>% 
    layer_dropout(rate = 0.1) %>%
    # layer_conv_1d(filters = 16,
    #               kernel_size = 2,
    #               activation = 'relu',
    #               input_shape = c(no_snps,1)) %>%
    # # layer_conv_1d(filters = 32,
    # #               kernel_size = 3,
    # #               activation = 'relu',
    # #               input_shape = c(no_snps,1)) %>% 
    # layer_dropout(rate = 0.20) %>%
    # layer_max_pooling_1d() %>% 
    layer_flatten() %>% 
    #layer_dense(units = 256, activation = "relu") %>%
    #layer_dropout(rate = 0.1) %>% 
    #layer_dense(units = 128, activation = "relu",kernel_regularizer = regularizer_l2(l = 0.001)) %>%
    #layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1, activation = 'linear')
  
  
  model %>% compile(
    optimizer = "RMSprop",
    loss = "mse",
    metrics = c("mae")
  )
  
  # Train the model 
  
  history = model %>% fit(xtrain, ytrain,
                          epochs = 25,
                          batch_size = 128,
                          validation_split = 0.2)
  
  
  test_prediction <- model %>% evaluate(xtest,ytest)
  
  #data.frame(y = predict(model, as.matrix(xtest)))
  data.frame(y = predict(model, xtest))
  
  d.acc.mass.ml[ii] <- cor(y_test,data.frame(y = predict(model,(xtest)))[,1])
}

write.table(d.acc.mass.ml , file= "~/AnimalModels/GGvsA/code/deepLearning/acc.mass.ml.txt", row.names=FALSE,sep=" ",quote=FALSE,col.names=FALSE)




##################################################################
### Run INLA based on the GBLUP approach
###
### To this end, use the GRM (genetic relatedness matrix) in the animal model
### 
### !!! This is very slow - account for at least 20-30min waiting time before inla terminates !!!
##################################################################

# Relatedness matrix from Henrik (vanRaden method 1) where +0.01 was already added to diagnoal!
d.Gmatrix <- read.table("../data/gghatvr3.triangle.g", header = F, sep=" ")

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
formula.mass = mass ~ sex + FGRM + month + age +   outer + other +
  f(hatchyear,model="iid",hyper=list(
    prec=list(initial=log(50), prior="pc.prec",param=c(1,0.05))
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

model1.mass = inla(formula=formula.mass, family="gaussian",
                   data=d.morph,
                   control.family=list(hyper = list(theta = list(initial=log(0.5),  prior="pc.prec",param=c(sqrt(2),0.05)))),
                   control.compute=list(dic=F, return.marginals=FALSE)
                   # control.compute=list(config = TRUE)
)


model1.mass$summary.fixed
model1.mass$summary.hyperpar
inla_emarginal(model1.mass)
inla_mmarginal(model1.mass)
