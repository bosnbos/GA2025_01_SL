NB_Logistic
================
Bos Noah
2025-05-14

# Lasso, ridge or relaxed lassso?

Given that some predictors appear to have weak effects on *CLASS* and
the number of predictors is moderate (k = 58), regularized models like
Ridge, Lasso, and Elastic Net are good choices. Standard logistic
regression is not viable in this setting due to the high-dimensional
structure, which can lead to perfect separation—a scenario where a
hyperplane classifies the training data perfectly, preventing the model
from converging (see below). Lasso may introduce bias through shrinkage,
but its variance reduction can improve test-set performance. Relaxed
Lasso is also used to refit an OLS model on selected variables, reducing
bias while maintaining sparsity.

- **Logistic regression** is a reasonable baseline, but it can perform
  poorly when predictors are highly correlated or when the number of
  variables is large. In such cases, it may overfit or fail to converge
  due to perfect separation.
- **Ridge regression** is better suited to datasets with many correlated
  predictors. It retains all variables but shrinks their coefficients,
  reducing overfitting and potentially improving predictive accuracy.
- **Lasso regression** performs automatic feature selection by shrinking
  some coefficients to zero. It is most effective when we believe that
  only a subset of variables truly contributes to the outcome, making it
  ideal for building sparse, interpretable models.
- **Relaxed Lasso** may be preferred when Lasso is too aggressive. After
  selecting variables, it reduces bias by partially refitting
  coefficients, often improving generalization without sacrificing
  sparsity.

![](NB_logistic_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

- **slight multicollinearity. some correlations are quite large**

``` r
#library(caret)
#cv_ctrl = trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
#df$spam = factor(df$Class, levels = c(0,1), label = c('not_spam', "spam"))


#cv_logistic = train(
#  spam ~ ., data = df[idx,],
#  method = "glm",
#  family = "binomial",
#  trControl = cv_ctrl,
#  metric = "ROC",
#  preProcess = c("center", "scale", "zv")
#)
```

**Model found a hyperplane that separates the classes perfectly, coefs
go to ± infinity. common in text data with word frequencies. Too many
predictors Logistic regression overfits or cant solve the likelihood
equation. Perfect is simply more likely with higher dimensions**

## LASSO Regression

![](NB_logistic_files/figure-gfm/lasso-1.png)<!-- -->

    ## 
    ## Call:  cv.glmnet(x = X[idx, ], y = y[idx], family = "binomial", alpha = 1) 
    ## 
    ## Measure: Binomial Deviance 
    ## 
    ##       Lambda Index Measure      SE Nonzero
    ## min 0.000233    73  0.4489 0.02269      56
    ## 1se 0.002617    47  0.4697 0.01959      52

This Lasso model performed best at a lambda of 0.00023, using 56
predictors. A slightly simpler model with 52 predictors (1-SE rule) had
only a small drop in performance, suggesting good accuracy with fewer
variables.

## Ridge Regression

![](NB_logistic_files/figure-gfm/ridge-1.png)<!-- -->

    ## 
    ## Call:  cv.glmnet(x = X[idx, ], y = y[idx], family = "binomial", alpha = 0) 
    ## 
    ## Measure: Binomial Deviance 
    ## 
    ##      Lambda Index Measure      SE Nonzero
    ## min 0.01890   100  0.5217 0.01874      57
    ## 1se 0.02498    97  0.5358 0.01857      57

This Ridge model achieved its best performance at $\lambda$ = 0.0189
with all 57 predictors. A slightly more regularized model (1-SE rule)
also used all predictors and had only a small increase in deviance,
indicating that regularization helps stabilize the model without
reducing the number of features.

![](NB_logistic_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

This histogram shows the distribution of Ridge regression coefficients.
Unlike Lasso, Ridge does not set coefficients exactly to zero, but
shrinks them toward zero. Most coefficients are small, clustered around
zero, indicating weak predictors, while a few have larger absolute
values, suggesting stronger influence on the outcome.

## Elastic Net

![](NB_logistic_files/figure-gfm/elastic%20net-1.png)<!-- -->

    ## 
    ## Call:  cv.glmnet(x = X[idx, ], y = y[idx], family = "binomial", alpha = 0.5) 
    ## 
    ## Measure: Binomial Deviance 
    ## 
    ##        Lambda Index Measure      SE Nonzero
    ## min 0.0003212    77  0.4490 0.02117      57
    ## 1se 0.0027293    54  0.4681 0.02015      53

This Elastic Net model ($\alpha = 0.5$) performed best at $\lambda$ =
0.00032 using all 57 predictors. A more regularized model selected by
the 1-SE rule used 53 predictors with only a small drop in performance,
indicating a good balance between accuracy and model simplicity.

## Relaxed lasso

![](NB_logistic_files/figure-gfm/relaxed%20lasso-1.png)<!-- -->

    ## 
    ## Call:  cv.glmnet(x = X[idx, ], y = y[idx], relax = T, family = "binomial",      alpha = 1, lambda.min.ratio = 0.001) 
    ## 
    ## Measure: Binomial Deviance 
    ## 
    ##     Gamma Index   Lambda Index Measure      SE Nonzero
    ## min   0.5     3 0.002173    65  0.4337 0.01647      52
    ## 1se   0.0     1 0.014296    38  0.4491 0.01514      34

This Relaxed Lasso model achieved its lowest binomial deviance (0.4571)
with $\gamma$ = 1 (equivalent to standard Lasso), $\lambda$ = 0.000366,
and 53 active predictors. Using the 1-SE rule, a more relaxed model
($\gamma$ = 0.25) with $\lambda$ = 0.0151 selects only 35 predictors and
maintains strong performance (deviance = 0.4721), offering improved
sparsity and potentially better generalization.

This Relaxed Lasso model, using a higher minimum lambda, achieved its
best performance with $\gamma$ = 0.5 and $\lambda$ = 0.00217, reaching
the lowest deviance (0.434) with 52 predictors. The 1-SE rule selects a
simpler model ($\gamma$ = 0, $\lambda$ = 0.0143) with 34 predictors and
slightly higher deviance (0.449), offering better sparsity with minimal
loss in accuracy.

# Model evaluation

First predicted all models using the test data. Then converted the
probabilities to class labels using a 0.5 threshold. Calculated
confusion matrices to get accuracies, false positives and negatives.
Than AUC values to compare the performance across different cutoff
values.

## Confusion matrices

using 0.5 cutoff

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 792  59
    ##          1  52 478
    ##                                          
    ##                Accuracy : 0.9196         
    ##                  95% CI : (0.904, 0.9334)
    ##     No Information Rate : 0.6112         
    ##     P-Value [Acc > NIR] : <2e-16         
    ##                                          
    ##                   Kappa : 0.8305         
    ##                                          
    ##  Mcnemar's Test P-Value : 0.569          
    ##                                          
    ##             Sensitivity : 0.8901         
    ##             Specificity : 0.9384         
    ##          Pos Pred Value : 0.9019         
    ##          Neg Pred Value : 0.9307         
    ##              Prevalence : 0.3888         
    ##          Detection Rate : 0.3461         
    ##    Detection Prevalence : 0.3838         
    ##       Balanced Accuracy : 0.9143         
    ##                                          
    ##        'Positive' Class : 1              
    ## 

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 799  87
    ##          1  45 450
    ##                                           
    ##                Accuracy : 0.9044          
    ##                  95% CI : (0.8877, 0.9194)
    ##     No Information Rate : 0.6112          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.796           
    ##                                           
    ##  Mcnemar's Test P-Value : 0.0003589       
    ##                                           
    ##             Sensitivity : 0.8380          
    ##             Specificity : 0.9467          
    ##          Pos Pred Value : 0.9091          
    ##          Neg Pred Value : 0.9018          
    ##              Prevalence : 0.3888          
    ##          Detection Rate : 0.3259          
    ##    Detection Prevalence : 0.3584          
    ##       Balanced Accuracy : 0.8923          
    ##                                           
    ##        'Positive' Class : 1               
    ## 

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 794  63
    ##          1  50 474
    ##                                           
    ##                Accuracy : 0.9182          
    ##                  95% CI : (0.9025, 0.9321)
    ##     No Information Rate : 0.6112          
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.8271          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.259           
    ##                                           
    ##             Sensitivity : 0.8827          
    ##             Specificity : 0.9408          
    ##          Pos Pred Value : 0.9046          
    ##          Neg Pred Value : 0.9265          
    ##              Prevalence : 0.3888          
    ##          Detection Rate : 0.3432          
    ##    Detection Prevalence : 0.3794          
    ##       Balanced Accuracy : 0.9117          
    ##                                           
    ##        'Positive' Class : 1               
    ## 

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 794  63
    ##          1  50 474
    ##                                           
    ##                Accuracy : 0.9182          
    ##                  95% CI : (0.9025, 0.9321)
    ##     No Information Rate : 0.6112          
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.8271          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.259           
    ##                                           
    ##             Sensitivity : 0.8827          
    ##             Specificity : 0.9408          
    ##          Pos Pred Value : 0.9046          
    ##          Neg Pred Value : 0.9265          
    ##              Prevalence : 0.3888          
    ##          Detection Rate : 0.3432          
    ##    Detection Prevalence : 0.3794          
    ##       Balanced Accuracy : 0.9117          
    ##                                           
    ##        'Positive' Class : 1               
    ## 

## AUC

    ##         Lasso         Ridge    ElasticNet Relaxed_Lasso 
    ##     0.9684905     0.9577166     0.9673851     0.9695451

These AUC results show that all four models perform well, with AUCs
above 0.95. Relaxed Lasso outperforms the others, achieving the highest
AUC (0.973), suggesting it best balances variable selection and
predictive accuracy. Lasso, Ridge, and Elastic Net perform comparably
($\approx 0.96$), but Relaxed Lasso offers a modest improvement in
discriminative power.

## ROC curves

![](NB_logistic_files/figure-gfm/roc%20curves-1.png)<!-- -->

The ROC curves show that all models—Lasso, Ridge, Elastic Net, and
Relaxed Lasso—achieve excellent classification performance, with AUC
values around 0.96–0.97. The curves closely overlap, indicating similar
sensitivity-specificity trade-offs, though Relaxed Lasso slightly
outperforms others, confirming its strong discriminative power in this
high-dimensional setting.

    ##           Model       AUC  Accuracy
    ## 1         Lasso 0.9684905 0.9196235
    ## 2         Ridge 0.9577166 0.9044171
    ## 3   Elastic Net 0.9673851 0.9181752
    ## 4 Relaxed Lasso 0.9695451 0.9181752

**Results**

Among the four models evaluated, Relaxed Lasso achieved the highest AUC
(0.970) with strong accuracy (91.82%), indicating it effectively
balances variable selection and predictive stability. Lasso slightly
outperformed in accuracy (91.96%) and had a nearly equal AUC (0.968),
demonstrating excellent classification performance. Elastic Net closely
followed, with an AUC of 0.967 and accuracy of 91.82%, showing a strong
compromise between Ridge and Lasso. Ridge regression performed slightly
worse, with an AUC of 0.958 and accuracy of 90.44%, suggesting less
effective control over correlated predictors. Overall, Relaxed Lasso
remains the most robust and interpretable model in this setting.
