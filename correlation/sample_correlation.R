train <- read.csv("train.csv", stringsAsFactors = FALSE)

head(train)

train

#density function
plot(density(train$SalePrice)) # Right skewed
plot(density(log(train$SalePrice)))

#boxplot for categorical features
library(lattice)
bwplot(OverallQual ~ SalePrice, data = train)
bwplot(Neighborhood ~ SalePrice, data = train)

#which
c(T,F,T)
which(c(T,F,T))
x <- c(1,1,1,0,0,1,1)
x!=1
which(x!=1)


#Homework Example

# 1. correlation matrix of continous features

str(train)

# Be Cautious: Not all "numeric features" are REAL numerical features. need to check one by one.

# for example: 
table(train$MoSold) #MoSold: categorical features

#new homework: fill the type of features, excel file

library(corrplot)
trainCont <- train[sapply(train,is.numeric)] 
new.trainCont<-subset(trainCont, select = -c(Id,MoSold) )#exclude MoSold
correlations <- cor(new.trainCont, use = "pairwise.complete.obs")
head(correlations)
rowInd <- apply(correlations, 1, function(x) return(sum(x > 0.5 | x < -0.5) > 1))
corrplot(correlations[rowInd, rowInd], method = "square")

# 2. predictive continous variables
correlations[,"SalePrice"]>0.5|correlations[,"SalePrice"]< (-0.5)
which(correlations[,"SalePrice"]>0.5|correlations[,"SalePrice"]< (-0.5))

# 3. predictive categorical features
trainChar <- train[sapply(train,is.character)] 
trainChar <- names(train)[which(sapply(train, is.character))]# not complete, please double check!
trainChar

bwplot(PavedDrive ~ SalePrice, data = train)
bwplot(ExterQual ~ SalePrice, data = train)

# 4. feature engineering examples

#TotalBath
train$TotalBath <- with(train, BsmtFullBath + 0.5 * BsmtHalfBath + FullBath + 0.5 * HalfBath)
with(train,cor(TotalBath,SalePrice))
with(train,cor(BsmtFullBath,SalePrice))

#YearRemodAdd
dim(subset(train, YearBuilt != YearRemodAdd))
train$remod <- with(train, ifelse(YearBuilt != YearRemodAdd, 1, 0))
boxplot(subset(train, remod == 1)$SalePrice,
        subset(train, remod == 0)$SalePrice)
boxplot(log(subset(train, remod == 1)$SalePrice),
        log(subset(train, remod == 0)$SalePrice))
