# Telecom Churn Prediction

## Project Description
This project aims to predict whether a customer will switch to another telecom provider (churn) using a dataset with 21 variables related to customer behavior, such as monthly bill, internet usage, etc. We employed logistic regression, decision trees, and random forest algorithms to achieve this goal. The accuracy and AUC (Area Under the ROC Curve) of the models improved with each subsequent model.

## Steps Performed

### Logistic Regression
1. Data Cleaning and Preprocessing
   - Handled missing values.
   - Encoded categorical variables.
   - Normalized numerical features.

2. Train-Test Split
3. Feature Scaling
4. Feature Selection using RFE (Recursive Feature Elimination)
5. Handling Multicollinearity by calculating VIFs
6. Model Building
7. Model Evaluation (Accuracy: 78.3%, AUC: 0.85)
8. Prediction on Test Data

### Decision Trees
1. Train-Test Split
2. Model Building
3. Hyperparameter Tuning using Grid Search Cross-Validation
4. Model Evaluation (Accuracy: 79.8%, AUC: 0.87)

### Random Forest
1. Train-Test Split
2. Model Building
3. Hyperparameter Tuning using Grid Search Cross-Validation
4. Model Evaluation (Accuracy: 80.5%, AUC: 0.94)

## Logistic Regression vs. Trees
Random Forests provided a significant improvement in results compared to both logistic regression and decision trees, with much less effort. They leveraged the predictive power of decision trees and learned much more than a single decision tree could do alone. However, Random Forests lack visibility regarding key features and the direction of their effects, which logistic regression handles well. If interpretation is not of key significance, Random Forests are highly effective.

                  Logistic Regression 	   Decision Tree	   Random Forest
**Accuracy**		        78.3%			        79.8%		        80.5%

**AUC**			           0.85				     0.87			     0.94


**Key Components**

Code: Implementation of Logistic Regression, Decision Tree and Random Forest algorithms in Python.

Data: The Telecom Churn Dataset used for this project.

Documentation: Detailed explanation of the steps performed and comparison of the algorithms.


