train <- read.csv("train.csv", stringsAsFactors = FALSE)
train$LogPrice <- log(train$SalePrice)

# need to deal with NA, o.w., feature trained on different sets.
sum.na <- sort(sapply(train, function(x) { sum(is.na(x)) }), decreasing=TRUE)
sum.na

#Deal with missing value: there are different reasons causing NA.

# 1. For features with too many(>5%) missingness, we will delete them
dim(train)
keep.col <- names(which(sum.na < dim(train)[1] * 0.05))
length(keep.col)
train <- train[, keep.col]

sum.na <- sort(sapply(train, function(x) { sum(is.na(x)) }), decreasing=TRUE)
sum.na

# 2. Impute missing value as a new level: Missingness is caused by that it doesn't exsit, for example: When the Basement is not finished, there is no related features

# BsmtExposure  BsmtFinType2    BsmtQual      BsmtCond  BsmtFinType1    
#    38            38            37            37            37 

train[which(is.na(train$BsmtExposure)),c('BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','BsmtQual','BsmtCond')]

# create a new level "unfinished" for NA in "Bsmt"s features 

train$BsmtExposure[which(is.na(train$BsmtExposure))] <- 'Unf'
train$BsmtFinType1[which(is.na(train$BsmtFinType1))] <- 'Unf'
train$BsmtFinType2[which(is.na(train$BsmtFinType2))] <- 'Unf'
train$BsmtQual[which(is.na(train$BsmtQual))] <- 'Unf'
train$BsmtCond[which(is.na(train$BsmtCond))] <- 'Unf'

sum.na <- sort(sapply(train, function(x) { sum(is.na(x)) }), decreasing=TRUE)
sum.na

# 3. Simple missing might due to operation or data transfer
# we could use different ways to impute, such as mean, or median
summary(train$MasVnrArea)
train$MasVnrArea[which(is.na(train$MasVnrArea))] <- median(train$MasVnrArea, na.rm = T)

# 4. More advanced way is to use model based: using other features to predict the missing value
# imputation article: http://www.stat.columbia.edu/~gelman/arm/missing.pdf
# tutorial on how to use mice: https://datascienceplus.com/imputing-missing-data-with-r-mice-package/
library(mice)
imp.train <- mice(train, m=1, printFlag=FALSE)
# note the categorical (character) variable needs to be factor.
train$MasVnrType <- as.factor(train$MasVnrType)
train$Electrical <- as.factor(train$Electrical)
imp.train <- mice(train, m=1, method='cart', printFlag=FALSE)
train_complete <- complete(imp.train)

#confirm no NAs
sort(sapply(train_complete, function(x) { sum(is.na(x)) }), decreasing=TRUE)

#see imputing results
table(train$MasVnrType)
table(train_complete$MasVnrType)

table(train$Electrical)
table(train_complete$Electrical)

#store the imputed data 
write.csv(train_complete, file = "train_complete.csv",row.names = FALSE)
#==============================================================================
train<-read.csv("train_complete.csv",header=T)
head(train)


catg_feature <- c("MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour",
                  "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
                  "BldgType", "HouseStyle", "OverallQual", "OverallCond", "RoofStyle", "RoofMatl",
                  "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation",
                  "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating",
                  "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu",
                  "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC",
                  "Fence", "MiscFeature", "MoSold", "SaleType", "SaleCondition")

all_name<-names(train)
setdiff(catg_feature, all_name)

catg_feature <- c("MSSubClass", "MSZoning", "Street", "LotShape", "LandContour",
                  "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
                  "BldgType", "HouseStyle", "OverallQual", "OverallCond", "RoofStyle", "RoofMatl",
                  "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation",
                  "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating",
                  "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", 
                  "PavedDrive", 
                  "MoSold", "SaleType", "SaleCondition")

train[catg_feature] <- lapply(train[catg_feature], as.factor)
# model.matrix: create X matrix: add intercept, get dummy for categorical features. So need to set categorical feature as factors before fit data into "model.matrix"
ind <- model.matrix( ~., subset(train, select = -c(Id,SalePrice,LogPrice)))
dep <- train$LogPrice
table(train$MSZoning)
head(ind)
table(train$Street)

set.seed(123)
train.ind <- sample(1:dim(ind)[1], dim(ind)[1] * 0.7)
x.train <- ind[train.ind, ]
x.test <- ind[-train.ind, ]
y.train <- dep[train.ind]
y.test <- dep[-train.ind]

library(glmnet)
fit.lasso <- glmnet(x.train, y.train,  alpha=1)
fit.ridge <- glmnet(x.train, y.train, alpha=0)
fit.elnet <- glmnet(x.train, y.train, alpha=.5)


# 10-fold Cross validation for each alpha = 0, 0.1, ... , 0.9, 1.0
fit.lasso.cv <- cv.glmnet(x.train, y.train, type.measure="mse", alpha=1, 
                          family="gaussian")
fit.ridge.cv <- cv.glmnet(x.train, y.train, type.measure="mse", alpha=0,
                          family="gaussian")
fit.elnet.cv <- cv.glmnet(x.train, y.train, type.measure="mse", alpha=.5,
                          family="gaussian")


# Plot solution paths:
par(mfrow=c(3,2))
# For plotting options, type '?plot.glmnet' in R console
plot(fit.lasso, xvar="lambda")
plot(fit10, main="LASSO")

plot(fit.ridge, xvar="lambda")
plot(fit0, main="Ridge")

plot(fit.elnet, xvar="lambda")
plot(fit5, main="Elastic Net")

for (i in 0:10) {
  assign(paste("fit", i, sep=""), cv.glmnet(x.train, y.train, type.measure="mse", 
                                            alpha=i/10,family="gaussian"))
}


y_pred0 <- predict(fit0, s=fit0$lambda.1se, newx=x.test)
y_pred0
#yhat_ridge <- predict(fit.ridge.cv, s=fit.ridge.cv$lambda.1se, newx=x.test)
y_pred1 <- predict(fit1, s=fit1$lambda.1se, newx=x.test)
y_pred2 <- predict(fit2, s=fit2$lambda.1se, newx=x.test)
y_pred3 <- predict(fit3, s=fit3$lambda.1se, newx=x.test)
y_pred4 <- predict(fit4, s=fit4$lambda.1se, newx=x.test)
y_pred5 <- predict(fit5, s=fit5$lambda.1se, newx=x.test)
y_pred6 <- predict(fit6, s=fit6$lambda.1se, newx=x.test)
y_pred7 <- predict(fit7, s=fit7$lambda.1se, newx=x.test)
y_pred8 <- predict(fit8, s=fit8$lambda.1se, newx=x.test)
y_pred9 <- predict(fit9, s=fit9$lambda.1se, newx=x.test)
y_pred10 <- predict(fit10, s=fit10$lambda.1se, newx=x.test)

mse0 <- mean((y.test - y_pred0)^2)
mse1 <- mean((y.test - y_pred1)^2)
mse2 <- mean((y.test - y_pred2)^2)
mse3 <- mean((y.test - y_pred3)^2)
mse4 <- mean((y.test - y_pred4)^2)
mse5 <- mean((y.test - y_pred5)^2)
mse6 <- mean((y.test - y_pred6)^2)
mse7 <- mean((y.test - y_pred7)^2)
mse8 <- mean((y.test - y_pred8)^2)
mse9 <- mean((y.test - y_pred9)^2)
mse10 <- mean((y.test - y_pred10)^2)

all_mse<-c(mse0,mse1,mse2,mse3,mse4,mse5,mse6,mse7,mse8,mse9,mse10)
plot(seq( 0,  1, 0.1),all_mse,type='l',xlab="alpha",ylab="MSE")# choose alpha=0.7
mse7#MSE
sqrt(mse7)#RMSE
coef(fit7, s = "lambda.1se")



# potential improve:

#feature engineering: generte new features 

#different missing value impute techniques

# delete low variance features
low_var<-names(train[which(apply(train, 2,function(x) {sort(table(x),decreasing = TRUE)[1]>dim(train)[1]*0.95}))])
low_var
sapply(train[low_var],table)
table(train$Street)
new_train<-train[which(apply(train, 2,function(x) {sort(table(x),decreasing = TRUE)[1]<dim(train)[1]*0.95}))]
dim(new_train)
