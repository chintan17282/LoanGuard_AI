# LoanGuard_AI
Capstone Project for Berkley HAAS (ML &amp; AI)



##### What question are we trying to answer?
- Employ machine learning to predict which individuals are at the highest risk of defaulting on their loans?

##### What kind of problem is it? 
- Binary Classification



# DataSet

- [Dataset](https://www.coursera.org/projects/data-science-coding-challenge-loan-default-prediction)

- The dataset contains 255,347 rows and 18 columns in total.

**Features**

|      | Column Name    | Data Type | Description                                                  |
| ---- | -------------- | --------- | ------------------------------------------------------------ |
| 1    | LoanID         | string    | A unique identifier for each loan.                           |
| 2    | Age            | integer   | The age of the borrower.                                     |
| 3    | Income         | integer   | The annual income of the borrower.                           |
| 4    | LoanAmount     | integer   | The amount of money being borrowed.                          |
| 5    | CreditScore    | integer   | The credit score of the borrower indicating their creditworthiness. |
| 6    | MonthsEmployed | integer   | The number of months the borrower has been employed.         |
| 7    | NumCreditLines | integer   | The number of credit lines the borrower has open.            |
| 8    | InterestRate   | float     | The interest rate for the loan.                              |
| 9    | LoanTerm       | integer   | The term length of the loan in months.                       |
| 10   | DTIRatio       | float     | The Debt-to-Income ratio indicating the borrower's debt compared to their income. |
| 11   | Education      | string    | The highest level of education attained by the borrower (PhD Master's Bachelor's High School). |
| 12   | EmploymentType | string    | The type of employment status of the borrower (Full-time Part-time Self-employed Unemployed). |
| 13   | MaritalStatus  | string    | The marital status of the borrower (Single Married Divorced). |
| 14   | HasMortgage    | string    | Whether the borrower has a mortgage (Yes or No).             |
| 15   | HasDependents  | string    | Whether the borrower has dependents (Yes or No).             |
| 16   | LoanPurpose    | string    | The purpose of the loan (Home Auto Education Business Other). |
| 17   | HasCoSigner    | string    | Whether the loan has a co-signer (Yes or No).                |
| 18   | Default        | integer   | The binary target variable indicating whether the loan defaulted -1 or not (0). |



# Exploratory Data Analysis

File: [01_EDA.ipynb](01_EDA.ipynb)

## 1. Cleaning

- `LoanID`  has all distinct values. This will thus not be useful in our model. Thus dropping `LoanID`



## 2. Univariate Analysis

- There are 6 Catagorial Data

  1. Education, 2. EmploymentType 3. MaritalStatus, 4. HasMortgage, 5.HasDependents, 6. LoanPurpose,7. HasCoSigner, with possible values as

     ```json
     {
         "Education":     [ "Bachelor's", "Master's", "High School","PhD"],
         "EmploymentType":[ "Full-time", "Unemployed", "Self-employed", "Part-time" ],
         "MaritalStatus": [ "Divorced", "Married", "Single" ],
         "HasMortgage":   [ "Yes", "No" ],
         "HasDependents": [ "Yes", "No" ],
         "LoanPurpose":   [ "Other", "Auto", "Business", "Home", "Education" ],
         "HasCoSigner":   [ "Yes", "No" ]
     }
     ```

- Null Check

  - There were no null values in either Numerical or Categorial Data.

- Imbalance Check

  - DataSet is Imbalanced with 11.6% target as `1` and rest 88.4 as `0`

- Plotting the univariate features

  - <u>Income</u>

    <img src="images/univariate_plot_Income.png" alt="Income" style="zoom:80%;" />

  - Age

    <img src="images/univariate_plot_Age.png" alt="Age" style="zoom:80%;" />
    
  - LoanAmount

    <img src="images/univariate_plot_LoanAmount.png" alt="LoanAmount" style="zoom:80%;" />

  - InterestRate

    <img src="images/univariate_plot_InterestRate.png" alt="InterestRate" style="zoom:80%;" />

  - CreditScore

    <img src="images/univariate_plot_CreditScore.png" alt="CreditScore" style="zoom:80%;" />

  - **Conclusion**
     1. The Loan defaulters are high where `InterestRate` or `LoanAmount` is High.
     2. Younger people or low income category too had more defaulters.
     3. Credit Score in general did not show any obvious trend.
  
  

## 3. Bivariate Analysis

For Bivariate Analysis, a random sample was picked form "Loan Defaulters" to see if we can see any trend. For This 2 continuous numerical feature were plotted against a category,
1. Scatter Plots also shows the vaiation of CreditScore
2. ScatterPlot and KdePlot filters only Loan defaulters 


__1. Impact of Income on LoanAmount, with Purpose of Loan__
  <img src="images/bivariate_plots_Income_LoanAmount_LoanPurpose_IncomeGroup.png" style="zoom:60%;" />

    1. Lower Income Group are the once who have taken Highest Loan.
    2. Highest Loan Defaulters have defaulted in Education, Auto and Business Loan.
    3. Number of loan Taken in each Income group for Each Loan purpose is almost.

  




__2. Impact of InterestRate on LoanAmount, with Purpose of Loan__
  <img src="images/bivariate_plots_LoanAmount_InterestRate_LoanPurpose_IncomeGroup.png" style="zoom:60%;" />

    1. As expected, the High Interest, Higher Amount Loan defaulters are higher.
    2. Highest Loan Defaulters have defaulted in Education, Auto and Business Loan.


__3. Impact of InterestRate on LoanAmount, with AgeGroup__
  For The Purpose of bucketing age, groups age groups between multiples of 5 were grouped togerher as 5-10, 11-15 and so on.

  <img src="images/bivariate_plots_InterestRate_LoanAmount_LoanPurpose_AgeGroup.png" style="zoom:60%;" />

    1. Variation of above, it was seen Younger people have defaulted in Loan most, in all category



## 4. Multivariate Analysis

For Multivariate Analysis, a random sample was picked to see if we can see any trend. For This 1 continuous numerical feature were plotted against 2 categororial features,
1. ViolinPlot plotted, the distribution against Loan defaulters.
2. CountPlot Plotted count in each category
3. HistPlot on right only considered the defaulters and plotted them for each category

__1. Analysis of LoanAmount for each Education category__
  <img src="images/bivariate_gridplot_alpha_LoanAmount_Education_Default.png" style="zoom:60%;" />

__2. Analysis of LoanAmount for each Employment category__
  <img src="images/bivariate_gridplot_alpha_LoanAmount_EmploymentType_Default.png" style="zoom:60%;" />

__3. Analysis of LoanAmount for each LoanTerm category__
  <img src="images/bivariate_gridplot_alpha_LoanAmount_LoanTerm_Default.png" style="zoom:60%;" />

__4. Analysis of LoanAmount for each LoanPurpose category__
  <img src="images/bivariate_gridplot_alpha_LoanAmount_LoanPurpose_Default.png" style="zoom:60%;" />

__5. Analysis of LoanAmount for each MaritalStatus category__
  <img src="images/bivariate_gridplot_alpha_LoanAmount_MaritalStatus_Default.png" style="zoom:60%;" />

__6. Analysis of LoanAmount for each HasMortgage category__
  <img src="images/bivariate_gridplot_alpha_LoanAmount_HasMortgage_Default.png" style="zoom:60%;" />

  - **Conclusion**
     1. It was generally seen Loan Defaulters are generally The once who had no previous Mortgage and this trent is seen almost all LoanAmount  range.
     2. Higher Amount of of Loan Defaulters are Singles. Where as `Divorced` are more likely to default on High Amount Loan.
     3. Highest Loan was taken for Business, Defaulters were even spread across all LoanPurpose.
     4. Highest Loan was taken and defaulted on 24 months term. In addition to that. Highest Defaulters wuere is  higher tange across all term.
     
     

## 5. Outliers

- Outliers check was done using Z-Score on 2 fields (1. Income and 2, LoanAmount), both had no Outliers. 
  - Income Range is from 15000.00 to 149999.00
  - Loan Amount if in range 5000.00 to 249999.00
  - Creditscore, MonthsEmployed and LoanTerm all Seem to be in valid Range 

|       |        Age |     Income | LoanAmount | CreditScore | MonthsEmployed | NumCreditLines | InterestRate |   LoanTerm |   DTIRatio | Default    |
| ----: | ---------: | ---------: | ---------: | ----------: | -------------: | -------------: | -----------: | ---------: | ---------: | ---------- |
| count | 255347.000 | 255347.000 | 255347.000 |  255347.000 |     255347.000 |     255347.000 |   255347.000 | 255347.000 | 255347.000 | 255347.000 |
|  mean |     43.498 |  82499.305 | 127578.866 |     574.264 |         59.542 |          2.501 |       13.493 |     36.026 |      0.500 | 0.116      |
|   std |     14.990 |  38963.014 |  70840.706 |     158.904 |         34.643 |          1.117 |        6.636 |     16.969 |      0.231 | 0.320      |
|   min |     18.000 |  15000.000 |   5000.000 |     300.000 |          0.000 |          1.000 |        2.000 |     12.000 |      0.100 | 0.000      |
|   25% |     31.000 |  48825.500 |  66156.000 |     437.000 |         30.000 |          2.000 |        7.770 |     24.000 |      0.300 | 0.000      |
|   50% |     43.000 |  82466.000 | 127556.000 |     574.000 |         60.000 |          2.000 |       13.460 |     36.000 |      0.500 | 0.000      |
|   75% |     56.000 | 116219.000 | 188985.000 |     712.000 |         90.000 |          3.000 |       19.250 |     48.000 |      0.700 | 0.000      |
|   max |     69.000 | 149999.000 | 249999.000 |     849.000 |        119.000 |          4.000 |       25.000 |     60.000 |      0.900 | 1.000      |

---

# Feature Selection

**Notebook**: [02_FeatureSelection.ipynb](02_FeatureSelection.ipynb)

### 1. Categorial 
<img src="images/feature_selection_category.png" alt="Categorial Feature Selection" style="zoom:80%;" />

#### Below variables have least impact on target variable. 
These are bottom 30% in both filters
- LoanPurpose (Auto, Education, Others)

#### Highest impact features on target variable, these are top 10 by both filters
- Education (High School, Master's)
- EmploymentType (Full-time)
- HasDependents (Yes, No)
- HasCoSigner (Yes, No)
- HasMortgage (Yes, No)
- MaritalStatus_(Married, Divorced)


### 2. Numerical 

#### Age
<img src="images/feature_selection_numerical_Age.png" alt="Age" style="zoom:80%;" />

#### Income
<img src="images/feature_selection_numerical_Income.png" alt="Income" style="zoom:80%;" />

#### InterestRate
<img src="images/feature_selection_numerical_InterestRate.png" alt="InterestRate" style="zoom:80%;" />

#### LoanAmount
<img src="images/feature_selection_numerical_LoanAmount.png" alt="LoanAmount" style="zoom:80%;" />

#### MonthsEmployed
<img src="images/feature_selection_numerical_MonthsEmployed.png" alt="MonthsEmployed" style="zoom:80%;" />

#### DTIRatio
<img src="images/feature_selection_numerical_DTIRatio.png" alt="DTIRatio" style="zoom:80%;" />

#### Highest impact numerical features on target variable
Using Mutual Information with mutual_info_classif()
- Age, Income, NumCreditLines, InterestRate, LoanTerm are top 5 picks

#### Least impact numerical features on target variable
- DTIRatio and CreditScore are least contributing features

---
# Classification
<img src="images/Classification.png" style="zoom:100%;" />

This criterion is linked to a Learning OutcomeModeling:

## 1. Metric 

The choice of metrics depends on what exactly we are trying to answer. As per the problem statement,

> One of the primary objectives of companies with financial loan services is to decrease payment defaults and ensure that individuals are paying back their loans as expected.

The question we, want to answer is 

> How do we predict which individuals are at the highest risk of defaulting on their loans, so that proper interventions can be effectively deployed to the right audience.?

In technical terms we would like to identity majority of our `True Positives` and reduce `False Negative` . 
1. __Recall (Sensitivity)__ is a metric, that measures proportion of correctly predicted positive observations. It answers the question: “**Out of all actual positives, how many did the model capture?**”. 

   $\large Recall\; (Sensitivity)= \Large  \frac{TPs}{(TPs + FNs)}$

   Thus to achieve high `Precision Score` we would to increase True Positives (TP) and recduce False Positive (FP) 

Additionally, we also would like to reduce False Positive, this will make the model more pessimistic and loss of opportunity of more applications are rejected or more resources are wasted if more application are scrutinised.

2. __Precision__ score's focus is **out of the predictions made by the model, what percent is correct>?**

    $\large Precision\; (Sensitivity)= \Large  \frac{TPs}{(TPs + FPs)}$

    Thus, Model should be able to capture majority of `True Positives` and also reduce `False Positives`

Unbalanced dataset particularly are need additional Consideration.  

3. __F1 score__ is essential because it balances precision and recall, providing a single metric that considers both FPs and FNs. 

    $\large F1\; = 2* \Large \frac{Recall\; *\; Precision }{(Recall\; +\; Precision)}$

__Thus to conclude, the 3 Metrics for evaluation will be__
1. Recall (Sensitivity) Score
2. Precision Score
3. F1 Score

---

## Transformers

|Column | Transformation |Notes|
|:--|:--|:--|
|Education|OneHotEncoding||
|EmploymentType|OneHotEncoding||
|MaritalStatus|OneHotEncoding||
|LoanPurpose|OneHotEncoding||
|HasMortgage|OneHotEncoding |Option: `drop=if_binary`|
|HasDependents|OneHotEncoding |Option: `drop=if_binary`|
|HasCoSigner|OneHotEncoding |Option: `drop=if_binary`|
|LoanTerm|OrdinalEncoder|


## Classification Algorithm

Two Algorithms  which will be suitable for to evaluate the model for is
Based on the observation, we had seen, the data is non-linear. Thus first we would like to  
1.  Non-Linear Algorithm
    -  K-Nearest Neighbours
    -  Decision Tree (with/without class weight)

We would also like to give Linear Algirithm a shot, thus
2.  Linear Algorithm 
    -  LogistisRegression with Polynomial Features (with/without class weight)    

If the models are determined to be weak, we will use following Ensemble algorithm
3.  Ensemble Algorithm
    -  Boosting (CatBoostClassifier, XGBClassifier)
    -  Bagging (e.g RandomForestClassifier, BalancedBaggingClassifier)

4. Prior to Modeling use __Data Sampling Algorithms__ to balance Dataset
    1. Random under sampling
    2. Random over sampling
    3. Smote,Tomek, SMOTETomek
    4. PolynomialFeatures + PCA

--

## Algorithm 1

#### Steps

1. Column Transformation
2. Undersampling the majority class using **`RandomUnderSampler`** 
3. **GridSearchCV + K-Nearest Neighbor**

#### Notebook

- [03_KNN-UnderSampling.ipynb](03_KNN-UnderSampling.ipynb)

#### Scores

```bash
Recall Score:       0.623
Precision Score:    0.169
F1 Score:           0.266
ROC AUC Score:      0.609
Accuracy Score:     0.599
```

#### Custom Threshold

<img src="images/KNN_undersampling.png" style="zoom:100%;" />

| threshold | recall | precision | f1-score | accuracy |
| :-------- | -----: | --------: | -------: | -------- |
| 0.300     |  0.950 |     0.127 |    0.224 | 0.233    |
| 0.400     |  0.825 |     0.142 |    0.242 | 0.399    |
| 0.500     |  0.623 |     0.169 |    0.266 | 0.599    |
| 0.600     |  0.365 |     0.204 |    0.262 | 0.760    |
| 0.700     |  0.153 |     0.250 |    0.189 | 0.848    |

---

## Algorithm 2

#### Steps

1. Column Transformation
2. Using  **`TomekLinks`** to Unbalanced dataset
3. **GridSearchCV + K-Nearest Neighbor**

#### Notebook

- [04_KNN-TomekLinks.ipynb](04_KNN-TomekLinks.ipynb)

#### Scores

```bash
Recall Score:       0.031
Precision Score:    0.316
F1 Score:           0.057
ROC AUC Score:      0.511
Accuracy Score:     0.879
```

#### Custom Threshold

<img src="images/KNN_Tomek.png" style="zoom:100%;" />

| threshold | recall | precision | f1-score | accuracy |
| :-------- | -----: | --------: | -------: | -------- |
| 0.300     |  0.194 |     0.241 |    0.215 | 0.835    |
| 0.400     |  0.089 |     0.292 |    0.137 | 0.869    |
| 0.500     |  0.032 |     0.316 |    0.058 | 0.879    |
| 0.600     |  0.011 |     0.388 |    0.021 | 0.883    |
| 0.700     |  0.003 |     0.464 |    0.006 | 0.883    |

---

## Algorithm 3

#### Steps

1. Column Transformation
2. Using  **`SMOTE`** to Unbalanced dataset
3. **GridSearchCV + K-Nearest Neighbor**

#### Notebook

- [05_KNN-SMOTE.ipynb](05_KNN-SMOTE.ipynb)

#### Scores

```bash
Recall Score:       0.472
Precision Score:    0.167
F1 Score:           0.247
ROC AUC Score:      0.581
Accuracy Score:     0.664
```

#### Custom Threshold

<img src="images/KNN_SMOTE.png" style="zoom:100%;" />

| threshold | recall | precision | f1-score | accuracy |
| :-------- | -----: | --------: | -------: | -------- |
| 0.300     |  0.604 |     0.157 |    0.249 | 0.575    |
| 0.400     |  0.495 |     0.165 |    0.248 | 0.649    |
| 0.500     |  0.472 |     0.167 |    0.247 | 0.664    |
| 0.600     |  0.364 |     0.176 |    0.238 | 0.728    |
| 0.700     |  0.330 |     0.182 |    0.234 | 0.749    |

---

## Algorithm 4

#### Steps

1. Column Transformation
2. Determine approx range if all parameters for Decision Tree
3. **GridSearchCV + Decision Tree**

#### Notebook

- [06_DecisionTree.ipynb](06_DecisionTree.ipynb)

#### Scores

```bash
Recall Score:       0.777
Precision Score:    0.172
F1 Score:           0.281
ROC AUC Score:      0.644
Accuracy Score:     0.541
```

#### Custom Threshold

<img src="images/Decision-I.png" style="zoom:100%;" />

| threshold | recall | precision | f1-score | accuracy |
| --------- | :----: | :-------: | :------: | :------: |
| 0.300     | 0.962  |   0.129   |  0.227   |  0.243   |
| 0.400     | 0.838  |   0.155   |  0.261   |  0.452   |
| 0.500     | 0.777  |   0.172   |  0.281   |  0.541   |
| 0.600     | 0.511  |   0.236   |  0.323   |  0.753   |
| 0.700     | 0.415  |   0.253   |  0.315   |  0.791   |



---

## Algorithm 5

#### Steps

1. Column Transformation
2. Determine approx range if all parameters for Decision Tree
3. **GridSearchCV + BalancedRandomForestClassifier**

#### Notebook

- [07_DecisionTree-BalancedRandomForest.ipynb](07_DecisionTree-BalancedRandomForest.ipynb)

#### Scores

```bash
Recall Score:       0.767
Precision Score:    0.175
F1 Score:           0.285
ROC AUC Score:      0.647
Accuracy Score:     0.555
```

#### Custom Threshold

<img src="images/Decision-II.png" style="zoom:100%;" />

| threshold | recall | precision | f1-score | accuracy |
| --------- | :----: | :-------: | :------: | :------: |
| 0.300     | 0.939  |   0.137   |  0.239   |  0.311   |
| 0.400     | 0.851  |   0.155   |  0.262   |  0.446   |
| 0.500     | 0.767  |   0.175   |  0.285   |  0.555   |
| 0.600     | 0.605  |   0.206   |  0.308   |  0.685   |
| 0.700     | 0.449  |   0.240   |  0.313   |  0.772   |



---

## Algorithm 6

#### Steps

1. Column Transformation
2. PCA
3. **CatBoostClassifier**

#### Notebook

- [08_Ensemble-Bagging.ipynb](08_Ensemble-Bagging.ipynb)

#### Scores

```bash
Recall Score:       0.675
Precision Score:    0.217
F1 Score:           0.328
ROC AUC Score:      0.678
Accuracy Score:     0.680
```

#### Custom Threshold

<img src="images/CatBoostClassifier.png" style="zoom:100%;" />

| threshold | recall | precision | f1-score | accuracy |
| --------- | :----: | :-------: | :------: | :------: |
| 0.300     | 0.884  |   0.158   |  0.268   |  0.441   |
| 0.400     | 0.790  |   0.183   |  0.297   |  0.568   |
| 0.500     | 0.675  |   0.217   |  0.328   |  0.680   |
| 0.600     | 0.539  |   0.261   |  0.352   |  0.771   |
| 0.700     | 0.380  |   0.327   |  0.352   |  0.838   |

---

## Algorithm 7

#### Steps

1. Column Transformation
2. SMOTETomek to balance the dataset
3. **XGBClassifier**

#### Notebook

- [09_Ensemble-Boosting.ipynb](09_Ensemble-Boosting.ipynb)

#### Scores

```bash
Recall Score:       0.114
Precision Score:    0.497
F1 Score:           0.185
ROC AUC Score:      0.549
Accuracy Score:     0.884
```

#### Custom Threshold

<img src="images/XGBClassifier.png" style="zoom:100%;" />

| threshold | recall | precision | f1-score | accuracy |
| --------- | :----: | :-------: | :------: | :------: |
| 0.300     | 0.362  |   0.326   |  0.343   |  0.840   |
| 0.400     | 0.216  |   0.416   |  0.285   |  0.874   |
| 0.500     | 0.114  |   0.497   |  0.185   |  0.884   |
| 0.600     | 0.048  |   0.560   |  0.089   |  0.886   |
| 0.700     | 0.021  |   0.626   |  0.041   |  0.885   |

---

## Algorithm 8

#### Steps

1. Column Transformation
2. PCA
3. **XGBClassifier**

#### Notebook

- [10_Ensemble-PCA-Boosting.ipynb](10_Ensemble-PCA-Boosting.ipynb) Section 4

#### Scores

```bash
Recall Score:       0.522
Precision Score:    0.241
F1 Score:           0.33
ROC AUC Score:      0.654
Accuracy Score:     0.755
```

#### Custom Threshold

<img src="images/XGBClassifier.png" style="zoom:100%;" />

| threshold | recall | precision | f1-score | accuracy |
| --------- | :----: | :-------: | :------: | :------: |
| 0.300     | 0.752  |   0.182   |  0.293   |  0.581   |
| 0.400     | 0.641  |   0.209   |  0.315   |  0.679   |
| 0.500     | 0.522  |   0.241   |  0.330   |  0.755   |
| 0.600     | 0.388  |   0.280   |  0.326   |  0.814   |
| 0.700     | 0.244  |   0.334   |  0.282   |  0.856   |

---

## Algorithm 9

#### Steps

1. Column Transformation
2. **XGBClassifier**

#### Notebook

- [10_Ensemble-PCA-Boosting.ipynb](10_Ensemble-PCA-Boosting.ipynb) Section 5

#### Scores

```bash
Recall Score:       0.092
Precision Score:    0.509
F1 Score:           0.157
ROC AUC Score:      0.541
Accuracy Score:     0.885
```

#### Custom Threshold

<img src="images/XGBClassifier-II.png" style="zoom:100%;" />

| threshold |  recall   | precision | f1-score | accuracy |
| --------- | :-------: | :-------: | :------: | :------: |
| recall    | precision | f1-score  | accuracy |          |
| 0.300     |   0.338   |   0.339   |  0.339   |  0.847   |
| 0.400     |   0.185   |   0.424   |  0.258   |  0.877   |
| 0.500     |   0.093   |   0.509   |  0.157   |  0.885   |
| 0.600     |   0.044   |   0.624   |  0.082   |  0.886   |

---



## Algorithm 10

#### Steps

1. Column Transformation
2. **RandomForestClassifier**

#### Notebook

- [11_RandomForest.ipynb](11_RandomForest.ipynb) Section 3

#### Scores

```bash
Recall Score:       0.031
Precision Score:    0.629
F1 Score:           0.059
ROC AUC Score:      0.514
Accuracy Score:     0.886
```

#### Custom Threshold

<img src="images/RandomForestClassifier.png" style="zoom:100%;" />

| threshold | recall | precision | f1-score | accuracy |
| --------- | :----: | :-------: | :------: | :------: |
| 0.300     | 0.238  |   0.372   |  0.291   |  0.866   |
| 0.400     | 0.104  |   0.519   |  0.174   |  0.885   |
| 0.500     | 0.033  |   0.623   |  0.062   |  0.886   |
| 0.600     | 0.008  |   0.833   |  0.015   |  0.885   |
| 0.700     | 0.001  |   1.000   |  0.002   |  0.885   |

---

## Algorithm 11

#### Steps

1. Column Transformation
2. **CatBoostClassifier**

#### Notebook

- [11_RandomForest.ipynb](11_RandomForest.ipynb) Section 4

#### Scores

```bash
Recall Score:       0.691
Precision Score:    0.219
F1 Score:           0.333
ROC AUC Score:      0.685
Accuracy Score:     0.680
```

#### Custom Threshold

<img src="images/CatBoostClassifier-II.png" style="zoom:100%;" />

| threshold | recall | precision | f1-score | accuracy |
| --------- | :----: | :-------: | :------: | :------: |
| 0.300     | 0.893  |   0.157   |  0.267   |  0.432   |
| 0.400     | 0.803  |   0.184   |  0.300   |  0.566   |
| 0.500     | 0.691  |   0.219   |  0.333   |  0.680   |
| 0.600     | 0.556  |   0.266   |  0.360   |  0.771   |
| 0.700     | 0.389  |   0.327   |  0.355   |  0.837   |

## Algorithm 12

#### Steps

1. Column Transformation
2. PCA
3. **LogisticRegression**

#### Notebook

- [12_LogisticRegression.ipynb](12_LogisticRegression.ipynb)

#### Scores

```bash
Recall Score:       0.025
Precision Score:    0.583
F1 Score:           0.048
ROC AUC Score:      0.511
Accuracy Score:     0.885
```

#### Custom Threshold

<img src="images/LogisticRegression-I.png" style="zoom:100%;" />

| threshold | recall | precision | f1-score | accuracy |
| --------- | :----: | :-------: | :------: | :------: |
| 0.300     | 0.209  |   0.404   |  0.275   |  0.873   |
| 0.400     | 0.087  |   0.507   |  0.148   |  0.885   |
| 0.500     | 0.025  |   0.583   |  0.048   |  0.885   |
| 0.600     | 0.004  |   0.667   |  0.009   |  0.885   |
| 0.700     | 0.000  |   1.000   |  0.001   |  0.885   |

---

# Business Recommendation

## For Numerical Features
#### Highest impact numerical features on target variable
Using Mutual Information with mutual_info_classif()
- Age, Income, NumCreditLines, InterestRate, LoanTerm are top 5 picks

#### Least impact numerical features on target variable
- DTIRatio and CreditScore are least contributing features

## For Categorial Features
#### Below variables have least impact on target variable. 
These are bottom 30% in both filters
- LoanPurpose (Auto, Education, Others)

#### Highest impact features on target variable, these are top 10 by both filters
- Education (High School, Master's)
- EmploymentType (Full-time)
- HasDependents (Yes, No)
- HasCoSigner (Yes, No)
- HasMortgage (Yes, No)
- MaritalStatus_(Married, Divorced)

## Proposed Model 
- Majorty of the algorithms are weak, to balance both Recall and Precision. Algorithm 10, using "CatBoostClassifier" was able to predict 69% of recalls, and Precision too was at 58%. ROC AUC Score was also at 68.5%. 












