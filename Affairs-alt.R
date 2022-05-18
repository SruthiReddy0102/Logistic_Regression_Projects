install.packages('AER')
data(Affairs,package="AER")
View(Affairs)
summary(Affairs)
Affairs$affairs <- as.factor(Affairs$affairs)
Affairs$affairs <- as.numeric(Affairs$affairs)
attach(Affairs)
nrow(Affairs)
affairs

for (i in 1:nrow(Affairs))
 
  if(Affairs$affairs[i] > 1){
  Affairs$affairs[i] <- 1
  } else{
    Affairs$affairs[i] <- 0
  }

View(Affairs$affairs)

View(Affairs)

# No imputation as there are no NA values in the dataset

sum(is.na(Affairs))

str(Affairs)

#preparing a linear regression

model <- lm(affairs ~ ., data = Affairs) #609 is the residual deviance
summary(model) #education, occupation, childreneyes, gendermale

pred <- predict(model, Affairs)
pred

# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1

model1 <- glm(affairs ~ .-education-occupation, data = Affairs, family = "binomial")
summary(model1) # 610 is the residual deviance

exp(coef(model1))

# Predicition to check model validation
prob <- predict(model1, Affairs, type = "response")
prob


#Optimal Cutoff
install.packages("InformationValue")
library(InformationValue)
Opt_Cutoff <- optimalCutoff(Affairs$affairs, prob)
Opt_Cutoff # 0.517 is the optimal cutoff

# Multi Collinearity

library(car)
vif(model1) # No value is greater than 10 so no collinearity problem exist

# misclassification error

misClassError(Affairs$affairs, prob, threshold = Opt_Cutoff)

#ROC curve

plotROC(Affairs$affairs, prob) # 0.7125

# We are going to use and Residual Deviance to compare the between different models

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > Opt_Cutoff, Affairs$affairs)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion)) # TP+TN/TP+TN+FP+FN
Acc #0.768


# Data Partitioning
n <-  nrow(Affairs)
n1 <-  n * 0.85
n2 <-  n - n1
train_index <-  sample(1:n, n1)
train <- Affairs[train_index, ]
test <-  Affairs[-train_index, ]

# Train the model using Training data
finalmodel <- glm(affairs ~ .-education-occupation, data = train, family = "binomial")
summary(finalmodel) # residual deviance is 529

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > Opt_Cutoff, test$affairs)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy # 0.8131 for test data

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL

pred_values <- ifelse(prob_test > Opt_Cutoff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values
View(test)

table(test$affairs, test$pred_values)


# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix 
confusion_train <- table(prob_train > Opt_Cutoff, train$affairs)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train #0.749

attach(Affairs)

# Additional metrics
# Calculate the below metrics
# precision | recall | True Positive Rate | False Positive Rate | Specificity | Sensitivity

# ROC Curve => used to evaluate the betterness of the logistic model
# more area under ROC curve better is the model 
# We will use ROC curve for any classification technique not only for logistic
install.packages("ROCR")
library(ROCR)
rocrpred <- prediction(prob, Affairs$affairs)
rocrperf <- performance(rocrpred, 'tpr', 'fpr')

str(rocrperf)

plot(rocrperf, colorize=T, text.adj=c(-0.2,1.7))
# More area under the ROC Curve better is the logistic regression model obtained

## Getting cutt off or threshold value along with true positive and false positive rates in a data frame 
rocr_cutoff <- data.frame(cut_off = rocrperf@alpha.values[[1]], fpr=rocrperf@x.values, tpr=rocrperf@y.values)

colnames(rocr_cutoff) <- c("cut_off","FPR","TPR")
View(rocr_cutoff)

library(dplyr)
rocr_cutoff$cut_off <- round(rocr_cutoff$cut_off, 6)
# Sorting data frame with respect to tpr in decreasing order 
rocr_cutoff <- arrange(rocr_cutoff, desc(TPR))
View(rocr_cutoff)



