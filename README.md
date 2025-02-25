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

# Phase II (Proposal)

---

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

## Classification Algorithm


Two Algorithms  which will be suitable for to evaluate the model for is
1.  Linear Algorithm
    -  LogistisRegression with Polynomial Features (with/without class weight)    
      
2.  Non-Linear Algorithm
    -  Decision Tree (with/without class weight) 

3.  Ensemble Algorithm
    -  Boosting 
    -  Bagging (e.g BalancedBaggingClassifier)

Alternatively

3. Prior to Modeling use __Data Sampling Algorithms__
    1. Random under sampling
    2. Random over sampling
    3. Combonition of both, _over_ and _under_ Sampling (E.g. Smote + Tomek)

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



