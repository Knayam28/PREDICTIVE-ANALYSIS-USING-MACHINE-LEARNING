# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: MAYANK SINGH

*INTERN ID*: CT04DH1908

*DOMAIN*: DATA ANALYTICS

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

*DESCRIPTION*

This task involves performing a predictive analysis of house prices in Boston using machine learning techniques. Specifically, the workflow encapsulates the process of using a classic regression algorithm—Linear Regression—on the Boston Housing dataset. The primary objective is to build a model capable of predicting median house values in various Boston neighborhoods using features representing local characteristics, environment, and demographics. The task also includes essential steps like data preprocessing, model training/testing, and performance evaluation using mean squared error.

Stepwise Task Breakdown:

Dataset Introduction and Loading:
The Boston Housing dataset, often cited in machine learning literature, contains information collected by the U.S Census Service concerning housing in the area of Boston, Massachusetts. The data for each home or neighborhood includes 13 features, both numerical and categorical:

CRIM: Per capita crime rate by town.

ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.

INDUS: Proportion of non-retail business acres per town.

CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise).

NOX: Nitric oxides concentration (parts per 10 million).

RM: Average number of rooms per dwelling.

AGE: Proportion of owner-occupied units built prior to 1940.

DIS: Weighted distances to five Boston employment centers.

RAD: Index of accessibility to radial highways.

TAX: Full-value property-tax rate per $10,000.

PTRATIO: Pupil-teacher ratio by town.

B: Calculated as 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents.

LSTAT: % lower status of the population.

MEDV: Median value of owner-occupied homes in $1000's (the target variable).

The dataset is imported using Pandas’ read_csv function and displayed for preliminary inspection. This initial step also helps ascertain the presence of missing values (evidenced by NaN entries in some feature columns).

Data Exploration and Cleaning:
The next step is to visually and programmatically inspect the dataset for missing values and possible outliers. While the code preview does not show explicit missing value handling, in a full analysis, this would typically include imputation or removal of incomplete records, normalization of feature scales if needed, and transformation of categorical variables. For a robust pipeline, exploratory data analysis (EDA)—such as plotting distributions, understanding correlations, and checking for multicollinearity—would also be carried out at this stage.

Feature/Label Definition and Data Splitting:
Once the dataset is cleaned and explored, features (inputs) and the target variable (output) are separated. Here, the features correspond to all columns except ‘MEDV,’ which serves as the label. The data is then partitioned into a training set and a testing set using scikit-learn’s train_test_split function. This split allows for unbiased assessment of the model's predictive power on unseen data.

Model Selection, Training, and Prediction:
Linear Regression is chosen as the predictive model. It’s a widely understood, interpretable regression method that seeks the line (or hyperplane) that best fits the training data. Using scikit-learn's LinearRegression estimator, the model is trained (fit) using the training data. Predictions (predict) are then made for the test set, representing the model’s estimated median home values for previously unseen neighborhoods.

Performance Evaluation:
To measure how well the model generalizes, mean squared error (MSE) between the predicted and actual MEDV values from the test set is computed using scikit-learn’s mean_squared_error function. The MSE gives a sense of the average squared difference between predicted and actual values—lower values are better, indicating a model that more accurately fits the data.

Discussion and Interpretation:
After evaluation, analysis focuses on interpreting the findings: understanding which features most strongly impact home values, potential shortcomings of linear regression (such as its inability to capture nonlinear relationships), and how the model’s performance might be improved (e.g., by using regularization, feature engineering, or more complex models).

Key Learning Outcomes:

Data Science Workflow: Fundamental understanding of the end-to-end process: data loading, preparation, model building, evaluation, and reporting.

Regression Modelling: Hands-on implementation of linear regression and practical appreciation for its strengths and limitations on real datasets.

Feature Engineering: The importance of handling missing values, multicollinearity, and outlier detection. While this specific notebook doesn't show all preprocessing, it's integral to real-world predictive modeling.

Model Evaluation: Use of appropriate regression metrics (MSE) for numeric prediction and understanding how to interpret these results.

Real-world Applicability: Insights into how environmental factors, socio-economic indicators, and urban layouts can statistically influence housing prices, a common domain in urban economics and real estate analytics.

Applications and Extensions:
While the immediate product is a linear regression model for predicting Boston house prices, the pipeline is generic and serves as a template for any regression task involving structured tabular data. With further sophistication, the analysis can be extended by evaluating alternative algorithms (e.g., Lasso, Ridge, Random Forest Regression), plotting residuals, tuning hyperparameters, and explaining predictions using model interpretability tools.

Conclusion:
This task exemplifies practical machine learning for tabular, regression-driven prediction in a real-world context, demonstrating not only model implementation but also highlighting the importance of careful data handling and rigorous evaluation.

#OUTPUT
