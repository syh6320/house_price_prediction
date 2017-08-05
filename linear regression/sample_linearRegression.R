
train <- read.csv("G:/kaggle/houseprice/data/train.csv", stringsAsFactors = FALSE)
with(train, plot(X1stFlrSF, SalePrice))
with(train, plot(X1stFlrSF, log(SalePrice))) # to stablize the variance of saleprice

train$LogPrice <- log(train$SalePrice)
mod1 <- lm(LogPrice ~ X1stFlrSF, data = train)
summary(mod1)

train.sub <- train[,c("TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2",
                      "X1stFlrSF","X2ndFlrSF", "LowQualFinSF", "LogPrice")]
summary(train.sub)
mod.simp <- lm(LogPrice ~ TotalBsmtSF+BsmtFinSF1+BsmtFinSF2+
                 X1stFlrSF+X2ndFlrSF+LowQualFinSF, data= train.sub)
train.sub$TotalBsmtSF <- scale(train.sub$TotalBsmtSF)
train.sub$BsmtFinSF1 <- scale(train.sub$BsmtFinSF1)
train.sub$BsmtFinSF2 <- scale(train.sub$BsmtFinSF2)
train.sub$X1stFlrSF <- scale(train.sub$X1stFlrSF)
train.sub$X2ndFlrSF <- scale(train.sub$X2ndFlrSF)
train.sub$LowQualFinSF <- scale(train.sub$LowQualFinSF)

mod.simp <- lm(LogPrice ~., data = train.sub)
# standardizing won't change the significant of features, but the estimate will change.

x = train.sub[, -7]
y <- train.sub[, 7]
t(x) %*% x
# to calculate the XT*X
# error due to only takig matrix
x <- as.matrix(x)
t(x) %*% x
# note that X dim is n * (p+1), XT*X dim is (p+1) * (p+1)
# inverse
xtxi <- solve(t(x) %*% x)
# beta estimator
xtxi %*% t(x) %*% y

# sigma estimator
head(mod.simp$res)
sqrt(sum(mod.simp$res^2)/(dim(train.sub)[1] - 7)) 

# R square
1 - sum(mod.simp$res^2)/sum((y-mean(y))^2)
# adjusted R square
1 - (sum(mod.simp$res^2)/sum((y-mean(y))^2)) * 
  (dim(train.sub)[1] - 1) /(dim(train.sub)[1] - 7)

# F test score
sst = sum((y - mean(y))^2) # sum of square total, df = 1459
ssr = sum(mod.simp$res^2) #  sum of square residual, df = 1459 - 6
ssm = sum((y - mean(y))^2) - sum(mod.simp$res^2) # sum of square model, df = 6
Fstats = (ssm)/(6) / (ssr / (dim(train)[1] - 6 -1)) # df is 6 and 1453
1 - pf(Fstats, 6, 1460-7) # def = p and n-1-p

# residual
sort(mod.simp$res)[c(1, 1460)]
plot(mod.simp$fit, mod.simp$res, xlab = 'Fitted', ylab = 'residual')
which.min(mod.simp$res)

plot(mod.simp)

# first plot we can check unbiased/biased and homo/hetero of the residual
# second plot to check the normality of the residual. 
# qqplot: for ith percentile data point, find ith percentile in normal distribution.

# collinearity
# variance inflation example
n <- 100
nosim <- 1000
set.seed(1)
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)
betas <- sapply(1:nosim, function(i){
  y <- x1+rnorm(n, sd=.3) 
  c(coef(lm(y ~ x1))[2], coef(lm(y ~ x1+x2))[2], coef(lm(y ~ x1+x2+x3))[2]) 
})
round(apply(betas,1,sd),5)

n <- 100
nosim <- 1000
set.seed(1)
x1 <- rnorm(n)
x2 <- x1/sqrt(2) + rnorm(n)/sqrt(2)
x3 <- x1*0.95 + rnorm(n)*sqrt(1-0.95^2)
betas <- sapply(1:nosim, function(i){
  y<- x1 + rnorm(n, sd=.3) 
  c(coef(lm(y ~ x1))[2], coef(lm(y ~ x1+x2))[2], coef(lm(y ~ x1+x2+x3))[2]) 
})
round(apply(betas,1,sd),5)
# observed due to collineariy, variance increased

# check VIF
summary(lm(x3 ~ x1 + x2)) # R squared is 0.874 
# vif should be
1/(1 - 0.874)

library(glmnet)
# glmnet only takes matrix, can use is.data.frame() or is.matrix() to test
# glmnet standardizes every feature, even categorical feature
# http://stackoverflow.com/questions/17887747/how-does-glmnets-standardize-argument-handle-dummy-variables

# has to take care of the missing value first
sort(sapply(train, function(x) {length(which(is.na(x)))}))
miss.col <- which(sapply(train, function(x) {length(which(is.na(x)))}) > 0)
ind <- model.matrix( ~., train[, -c(1, 81, miss.col)])
dep <- log(train$SalePrice)
fit <- glmnet(x=ind, y=dep) # default is alpha = 1, lasso
plot(fit)
# Understand the plot
# The top row indicates the number of nonzero coefficients at the current λ,
# which is the effective degrees of freedom (df) for the lasso.
# y axis is the value of coefficient
# x axis is the sum of absolute value of coefficients
plot(fit, label = T)
plot(fit, xvar = "lambda", label = T)

print(fit)
# Df is the non zero beta, 
# saturated model is a model with a parameter for every observation so that the data are fitted exactly.
# Deviance_model = 2*(loglikelihood_saturate_model - loglikelihood_current_model)
# Deviance_null = 2*(loglikelihood_saturate_model - loglikelihood_intercept_only_model)
# Deviance percentage = 1 -  Deviance_model / Deviance_null
# lambda value

coef(fit, s = 1/exp(2)) # s stands for lambda
coef(fit, s = 1/exp(8))

# We can choose lambda by checking the picture, Still kinda subjective
# use cross validation to get optimal value of lambda, 
cvfit <- cv.glmnet(ind, dep)
plot(cvfit)
# Two selected lambdas are shown, 
cvfit$lambda.min # value of lambda gives minimal mean cross validated error
cvfit$lambda.1se # most regularized model such that error is within one std err of the minimum
x = coef(cvfit, s = "lambda.min")
coef(cvfit, s = "lambda.1se")

# need to deal with NA, o.w., feature trained on different sets.
sum.na <- sort(sapply(train, function(x) { sum(is.na(x)) }), decreasing=TRUE)
# There are different reasons causing NA. 
# 1. For features with too many missingness, we will delete them
keep.col <- names(which(sum.na < dim(train)[1] * 0.05))
length(keep.col)
train <- train[, keep.col]

# 2. Missingness is caused by that it doesn't exsit, for example
# When the Basement is not finished, there is no related features
colnames(train)[which(grepl("Bsmt", colnames(train)))]
with(subset(train, is.na(BsmtExposure)), summary(TotalBsmtSF))
with(subset(train, is.na(BsmtExposure)), summary(BsmtFinSF1))
with(subset(train, is.na(BsmtExposure)), summary(BsmtFinSF2))
with(subset(train, is.na(BsmtExposure)), summary(BsmtUnfSF)) # so they are all unfinished.

train$BsmtExposure[which(is.na(train$BsmtExposure))] <- 'Unf'
train$BsmtFinType1[which(is.na(train$BsmtFinType1))] <- 'Unf'
train$BsmtFinType2[which(is.na(train$BsmtFinType2))] <- 'Unf'
train$BsmtQual[which(is.na(train$BsmtQual))] <- 'Unf'
train$BsmtCond[which(is.na(train$BsmtCond))] <- 'Unf'

# 3. Simple missing might due to operation or data transfer
# we could use different ways to impute, such as mean, or median
summary(train$MasVnrArea)
train$MasVnrArea[which(is.na(train$MasVnrArea))] <- median(train$MasVnrArea, na.rm = T)
train$MasVnrArea[which(is.na(train$MasVnrType))] <- NA

# 4. More advanced way is to use model based: using other features to predict the missing value
library(mice)

imp.train <- mice(train, m=1, printFlag=FALSE)
sort(sapply(complete(imp.train), function(x) { sum(is.na(x)) }), decreasing=TRUE)
train$MasVnrType <- as.factor(train$MasVnrType)
train$Electrical <- as.factor(train$Electrical)
imp.train <- mice(train, m=1, method='cart', printFlag=FALSE)
sort(sapply(complete(imp.train), function(x) { sum(is.na(x)) }), decreasing=TRUE)

# imputation article: http://www.stat.columbia.edu/~gelman/arm/missing.pdf
# tutorial on how to use mice: https://datascienceplus.com/imputing-missing-data-with-r-mice-package/
# note the categorical (character) variable needs to be factor.
train <- complete(imp.train)
mod.all <- lm(LogPrice ~., data = train)
# categorical variables, each level is a new variable.
# For example BsmtExposure
table(train$BsmtExposure)
# BsmtExposure feature has: Av  Gd  Mn  No Unf levels
# BsmtExposure:   No, Gd, Mn, NO, Av, No
# now becomes:
# BsmtExposureGd: 0   1
# BsmtExposureMn: 0   0
# BsmtExposureNO: 0   0
# BsmtExposureAv: 0   0