# Practical Machine Learning - Assignment Project
Milana Smuk  
15 June 2017  

## Summary

###Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

Data set can be found [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv).
More information about data set can be found [here](http://groupware.les.inf.puc-rio.br/har ) 



##Data pre-processing

For this analysis are used following packages.


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(rpart)
library(rpart.plot)
```

Downloading the data: 


```r
url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
file1 <- "pml-training.csv"
file2 <- "pml-testing.csv"
download.file(url=url1, dest=file1, mode="wb")
download.file(url=url2, dest=file2, mode="wb")
training <- read.csv("pml-training.csv",row.names=1,na.strings=c("NA","#DIV/0!",""))
testing <- read.csv("pml-testing.csv",row.names=1,na.strings=c("NA","#DIV/0!",""))
```

Inspecting of the data basic, it is shown that there are variables having a lot of missing values, variables containing NA values as well as variables not directly related to the target variable "classe". All those variables are removed; names of the sets are kept.



Now the data set has 19622 rows and 52 columns.

Target variable "Classe" is a factor variable which looks like:


```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

Splitting the data set based on the target "classe" variable into training and testing part in 70:30 ratio. For the reproducibility of the analysis, seed is set as well.


```r
set.seed(888)
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
trainSub <- training[inTrain, ]
testSub<- training[-inTrain, ]
```

New dimension of the training set is 13737 .

## Modelling and Testing

I will try to fit a Random forest and Decision tree models and check their performances on the test subset.

###1. Model - Decision Tree


```r
modelFitDT <- rpart(classe ~., data = trainSub, method = "class")
predDT <- predict(modelFitDT, testSub, type = "class")
CMDT <- confusionMatrix(predDT, testSub$classe)
print(CMDT)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1457  171   20   93   73
##          B   29  667   46   58   79
##          C   37  118  773  262   77
##          D  111   79   33  453  121
##          E   40  104  154   98  732
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6936          
##                  95% CI : (0.6817, 0.7054)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6114          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8704   0.5856   0.7534  0.46992   0.6765
## Specificity            0.9152   0.9553   0.8983  0.93010   0.9176
## Pos Pred Value         0.8032   0.7588   0.6101  0.56838   0.6489
## Neg Pred Value         0.9467   0.9057   0.9452  0.89957   0.9264
## Prevalence             0.2845   0.1935   0.1743  0.16381   0.1839
## Detection Rate         0.2476   0.1133   0.1314  0.07698   0.1244
## Detection Prevalence   0.3082   0.1494   0.2153  0.13543   0.1917
## Balanced Accuracy      0.8928   0.7705   0.8259  0.70001   0.7970
```


Accuracy of this model is about 71%, which means that out-of-sample error is about 29%. Additionally, Confusion matrix isn't diagonal enough, so this model isn't good enough as well. We will try with generalized boosted model next.
 
###2. Model - Random Forest


```r
modelFitRF <- randomForest(classe ~., data = trainSub)
predRF <- predict(modelFitRF, testSub)
CMRF <- confusionMatrix(predRF, testSub$classe)
print(CMRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    9    0    0    0
##          B    0 1128    4    0    0
##          C    0    2 1022   11    0
##          D    0    0    0  953    4
##          E    0    0    0    0 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9927, 0.9966)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9903   0.9961   0.9886   0.9963
## Specificity            0.9979   0.9992   0.9973   0.9992   1.0000
## Pos Pred Value         0.9947   0.9965   0.9874   0.9958   1.0000
## Neg Pred Value         1.0000   0.9977   0.9992   0.9978   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1917   0.1737   0.1619   0.1832
## Detection Prevalence   0.2860   0.1924   0.1759   0.1626   0.1832
## Balanced Accuracy      0.9989   0.9947   0.9967   0.9939   0.9982
```

Accuracy of this model is much better, about 99.5%, out-of-sample error is now about 0.5%. Confusion matrix is better than in first model.


## Conclusion
Based on modelling and testing part, we can conclude using of random forest model in testing of given test set.


```r
predTest <- predict(modelFitRF, testing, type = "class")
print(predTest)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
