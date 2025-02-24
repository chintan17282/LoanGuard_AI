# LoanGuard_AI
Capstone Project for Berkley HAAS (ML &amp; AI)



##### What question are we trying to answer?
- Employ machine learning to predict which individuals are at the highest risk of defaulting on their loans?

##### What kind of problem is it? 
- Binary Classification



# Exploratory Data Analysis

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

