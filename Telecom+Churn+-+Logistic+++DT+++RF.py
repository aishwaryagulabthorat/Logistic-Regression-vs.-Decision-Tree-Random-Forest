#!/usr/bin/env python
# coding: utf-8

# ## Telecom Churn Case Study
# With 21 predictor variables we need to predict whether a particular customer will switch to another telecom provider or not. In telecom terminology, this is referred to as churning and not churning, respectively.

# ### Step 1: Importing and Merging Data

# In[168]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[169]:


# Importing Pandas and NumPy
import pandas as pd, numpy as np


# In[170]:


# Importing all datasets
churn_data = pd.read_csv("/Users/aishwaryathorat/Movies/MS Courses/Upg/DT/Tree models instead of Logistic Regression/churn_data.csv")
churn_data.head()


# In[171]:


customer_data = pd.read_csv("/Users/aishwaryathorat/Movies/MS Courses/Upg/DT/Tree models instead of Logistic Regression/customer_data.csv")
customer_data.head()


# In[172]:


internet_data = pd.read_csv("/Users/aishwaryathorat/Movies/MS Courses/Upg/DT/Tree models instead of Logistic Regression/internet_data.csv")
internet_data.head()


# #### Combining all data files into one consolidated dataframe

# In[173]:


# Merging on 'customerID'
df_1 = pd.merge(churn_data, customer_data, how='inner', on='customerID')


# In[174]:


# Final dataframe with all predictor variables
telecom = pd.merge(df_1, internet_data, how='inner', on='customerID')


# ### Step 2: Inspecting the Dataframe

# In[175]:


# Let's see the head of our master dataset
telecom.head()


# In[176]:


# Let's check the dimensions of the dataframe
telecom.shape


# In[177]:


# statistical aspects of the dataframe
telecom.describe()


# In[178]:


telecom.info()


# ### Step 3: Data Preparation

# #### Converting some binary variables (Yes/No) to 0/1

# In[179]:


# List of variables to map

varlist =  ['PhoneService', 'PaperlessBilling', 'Churn', 'Partner', 'Dependents']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
telecom[varlist] = telecom[varlist].apply(binary_map)


# In[180]:


telecom.head()


# #### For categorical variables with multiple levels, create dummy features (one-hot encoded)

# In[181]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(telecom[['Contract', 'PaymentMethod', 'gender', 'InternetService']], drop_first=True)

# Adding the results to the master dataframe
telecom = pd.concat([telecom, dummy1], axis=1)


# In[182]:


telecom.head()


# In[183]:


# Creating dummy variables for the remaining categorical variables and dropping the level with big names.

# Creating dummy variables for the variable 'MultipleLines'
ml = pd.get_dummies(telecom['MultipleLines'], prefix='MultipleLines')
# Dropping MultipleLines_No phone service column
ml1 = ml.drop(['MultipleLines_No phone service'], axis=1)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,ml1], axis=1)

# Creating dummy variables for the variable 'OnlineSecurity'.
os = pd.get_dummies(telecom['OnlineSecurity'], prefix='OnlineSecurity')
os1 = os.drop(['OnlineSecurity_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,os1], axis=1)

# Creating dummy variables for the variable 'OnlineBackup'.
ob = pd.get_dummies(telecom['OnlineBackup'], prefix='OnlineBackup')
ob1 = ob.drop(['OnlineBackup_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,ob1], axis=1)

# Creating dummy variables for the variable 'DeviceProtection'. 
dp = pd.get_dummies(telecom['DeviceProtection'], prefix='DeviceProtection')
dp1 = dp.drop(['DeviceProtection_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,dp1], axis=1)

# Creating dummy variables for the variable 'TechSupport'. 
ts = pd.get_dummies(telecom['TechSupport'], prefix='TechSupport')
ts1 = ts.drop(['TechSupport_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,ts1], axis=1)

# Creating dummy variables for the variable 'StreamingTV'.
st =pd.get_dummies(telecom['StreamingTV'], prefix='StreamingTV')
st1 = st.drop(['StreamingTV_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,st1], axis=1)

# Creating dummy variables for the variable 'StreamingMovies'. 
sm = pd.get_dummies(telecom['StreamingMovies'], prefix='StreamingMovies')
sm1 = sm.drop(['StreamingMovies_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,sm1], axis=1)


# In[184]:


telecom.head()


# #### Dropping the repeated variables

# In[185]:


# We have created dummies for the below variables, so we can drop them
telecom = telecom.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies'], axis=1)


# In[186]:


telecom = telecom[~(telecom.TotalCharges == " ")]


# In[187]:


telecom.TotalCharges = telecom.TotalCharges.astype(float, errors="ignore")


# In[188]:


# #The varaible was imported as a string we need to convert it to float
# telecom['TotalCharges'] = telecom['TotalCharges'].convert_objects(convert_numeric=True)


# In[189]:


telecom.info()


# Now you can see that you have all variables as numeric.

# #### Checking for Outliers

# In[190]:


# Checking for outliers in the continuous variables
num_telecom = telecom[['tenure','MonthlyCharges','SeniorCitizen','TotalCharges']]


# In[191]:


# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%
num_telecom.describe(percentiles=[.25, .5, .75, .90, .95, .99])


# From the distribution shown above, you can see that there no outliers in your data. The numbers are gradually increasing.

# #### Checking for Missing Values and Inputing Them

# In[192]:


# Adding up the missing values (column-wise)
telecom.isnull().sum()


# It means that 11/7043 = 0.001561834 i.e 0.1%, best is to remove these observations from the analysis

# In[193]:


# Checking the percentage of missing values
round(100*(telecom.isnull().sum()/len(telecom.index)), 2)


# In[194]:


# Removing NaN TotalCharges rows
telecom = telecom[~np.isnan(telecom['TotalCharges'])]


# In[195]:


# Checking percentage of missing values after removing the missing values
round(100*(telecom.isnull().sum()/len(telecom.index)), 2)


# Now we don't have any missing values

# ### Step 4: Test-Train Split

# In[196]:


from sklearn.model_selection import train_test_split


# In[197]:


# Putting feature variable to X
X = telecom.drop(['Churn','customerID'], axis=1)

X.head()


# In[198]:


# Putting response variable to y
y = telecom['Churn']

y.head()


# In[199]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ### Step 5: Feature Scaling

# In[200]:


from sklearn.preprocessing import StandardScaler


# In[201]:


scaler = StandardScaler()

X_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])

X_train.head()


# In[202]:


### Checking the Churn Rate
churn = (sum(telecom['Churn'])/len(telecom['Churn'].index))*100
churn


# We have almost 27% churn rate

# ### Step 6: Looking at Correlations

# In[203]:


# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[204]:


telecom.head()


# In[205]:


telecom.columns


# In[206]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(telecom[['tenure', 'PhoneService', 'PaperlessBilling',
       'MonthlyCharges', 'TotalCharges', 'Churn', 'SeniorCitizen', 'Partner',
       'Dependents', 'Contract_One year', 'Contract_Two year',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
       'gender_Male', 'InternetService_Fiber optic', 'InternetService_No',
       'MultipleLines_No', 'MultipleLines_Yes', 'OnlineSecurity_No',
       'OnlineSecurity_Yes', 'OnlineBackup_No', 'OnlineBackup_Yes',
       'DeviceProtection_No', 'DeviceProtection_Yes', 'TechSupport_No',
       'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_Yes']].corr(),annot = True)
plt.show()


# #### Dropping highly correlated dummy variables

# In[207]:


X_test = X_test.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',
                       'StreamingTV_No','StreamingMovies_No'], axis=1)
X_train = X_train.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',
                         'StreamingTV_No','StreamingMovies_No'], axis=1)


# #### Checking the Correlation Matrix

# After dropping highly correlated variables now let's check the correlation matrix again.

# In[208]:


plt.figure(figsize = (20,10))
sns.heatmap(X_train.corr(),annot = True)
plt.show()


# ### Step 7: Model Building
# Let's start by splitting our data into a training set and a test set.

# #### Running Your First Training Model

# In[209]:


import statsmodels.api as sm


# In[210]:


X_train.head()


# In[211]:


X_train.info()


# In[212]:


y_train.info()


# In[213]:


# X_train=np.asarray(X_train)
# y_train=np.asarray(y_train)


# In[229]:


# Logistic regression model
logm1 = sm.GLM(y_train.astype(float),(sm.add_constant(X_train.astype(float))), family = sm.families.Binomial())
logm1.fit().summary()


# In[230]:


import statsmodels.api as sm

# # Ensure the input data is correctly formatted and numeric
X_train = sm.add_constant(X_train.astype(float))  # Add a constant term for the intercept
logit_model = sm.Logit(y_train.astype(float), X_train.astype(float))
result = logit_model.fit()
print(result.summary())


# ### Step 8: Feature Selection Using RFE

# In[216]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[217]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg, n_features_to_select=15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[218]:


rfe.support_


# In[219]:


X_train


# In[221]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[222]:


col = X_train.columns[rfe.support_]


# In[223]:


X_train.columns[~rfe.support_]


# ##### Assessing the model with StatsModels

# In[225]:


print(X_train.dtypes)


# In[226]:


X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_train = X_train.fillna(0)


# In[228]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train.astype(float),X_train_sm.astype(float), family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[232]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm.astype(float))
y_train_pred[:10]


# In[233]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# ##### Creating a dataframe with the actual churn flag and the predicted probabilities

# In[234]:


y_train_pred_final = pd.DataFrame({'Churn':y_train.values, 'Churn_Prob':y_train_pred})
y_train_pred_final['CustID'] = y_train.index
y_train_pred_final.head()


# ##### Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

# In[235]:


y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[236]:


from sklearn import metrics


# In[237]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
print(confusion)


# In[55]:


# Predicted     not_churn    churn
# Actual
# not_churn        3270      365
# churn            579       708  


# In[238]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))


# #### Checking VIFs

# In[239]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[240]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# There are a few variables with high VIF. It's best to drop these variables as they aren't helping much with prediction and unnecessarily making the model complex. The variable 'MonthlyCharges' has the highest VIF. So let's start by dropping that.

# In[241]:


col = col.drop('MonthlyCharges', 1)
col


# In[242]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train.astype(float),X_train_sm.astype(float), family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[243]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[244]:


y_train_pred[:10]


# In[245]:


y_train_pred_final['Churn_Prob'] = y_train_pred


# In[246]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[247]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))


# So overall the accuracy hasn't dropped much.

# ##### Let's check the VIFs again

# In[248]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[249]:


# Let's drop TotalCharges since it has a high VIF
col = col.drop('TotalCharges')
col


# In[250]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[251]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[252]:


y_train_pred[:10]


# In[253]:


y_train_pred_final['Churn_Prob'] = y_train_pred


# In[254]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[255]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))


# The accuracy is still practically the same.

# ##### Let's now check the VIFs again

# In[256]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# All variables have a good value of VIF. So we need not drop any more variables and we can proceed with making predictions using this model only

# In[257]:


# Let's take a look at the confusion matrix again 
confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
confusion


# In[76]:


# Actual/Predicted     not_churn    churn
        # not_churn        3269      366
        # churn            595       692  


# In[258]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted)


# ## Metrics beyond simply accuracy

# In[259]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[260]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[261]:


# Let us calculate specificity
TN / float(TN+FP)


# In[262]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[263]:


# positive predictive value 
print (TP / float(TP+FP))


# In[264]:


# Negative predictive value
print (TN / float(TN+ FN))


# ### Step 9: Plotting the ROC Curve

# An ROC curve demonstrates several things:
# 
# - It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
# - The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
# - The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

# In[265]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False)
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[266]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Churn, y_train_pred_final.Churn_Prob, drop_intermediate = False )


# In[267]:


draw_roc(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)


# ### Step 10: Finding Optimal Cutoff Point

# Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# In[268]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[269]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[270]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# #### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

# In[271]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Churn_Prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[272]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.final_predicted)


# In[273]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.final_predicted )
confusion2


# In[274]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[275]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[276]:


# Let us calculate specificity
TN / float(TN+FP)


# In[277]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[278]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[279]:


# Negative predictive value
print (TN / float(TN+ FN))


#  

#  

#  

#  

# ## Precision and Recall

# In[99]:


#Looking at the confusion matrix again


# In[280]:


confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
confusion


# ##### Precision
# TP / TP + FP

# In[281]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# ##### Recall
# TP / TP + FN

# In[282]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# Using sklearn utilities for the same

# In[283]:


from sklearn.metrics import precision_score, recall_score


# In[284]:


get_ipython().run_line_magic('pinfo', 'precision_score')


# In[285]:


precision_score(y_train_pred_final.Churn, y_train_pred_final.predicted)


# In[286]:


recall_score(y_train_pred_final.Churn, y_train_pred_final.predicted)


# ### Precision and recall tradeoff

# In[287]:


from sklearn.metrics import precision_recall_curve


# In[288]:


y_train_pred_final.Churn, y_train_pred_final.predicted


# In[289]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)


# In[290]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ### Step 11: Making predictions on the test set

# In[291]:


X_test[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(X_test[['tenure','MonthlyCharges','TotalCharges']])


# In[292]:


X_test = X_test[col]
X_test.head()


# In[293]:


X_test_sm = sm.add_constant(X_test)


# Making predictions on the test set

# In[295]:


y_test_pred = res.predict(X_test_sm.astype(float))


# In[296]:


y_test_pred[:10]


# In[297]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[298]:


# Let's see the head
y_pred_1.head()


# In[299]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[300]:


# Putting CustID to index
y_test_df['CustID'] = y_test_df.index


# In[301]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[302]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[303]:


y_pred_final.head()


# In[304]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Churn_Prob'})


# In[305]:


# Rearranging the columns
y_pred_final = y_pred_final[['CustID','Churn','Churn_Prob']]


# In[306]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[307]:


y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.42 else 0)


# In[308]:


y_pred_final.head()


# In[309]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Churn, y_pred_final.final_predicted)


# In[310]:


confusion2 = metrics.confusion_matrix(y_pred_final.Churn, y_pred_final.final_predicted )
confusion2


# In[311]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[312]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[313]:


# Let us calculate specificity
TN / float(TN+FP)


# ## Using Decision Trees

# For Decision trees we do not need to scale features.

# In[314]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)


# In[315]:


X_train.shape, X_test.shape


# In[316]:


from sklearn.tree import DecisionTreeClassifier


# In[317]:


dt_base = DecisionTreeClassifier(random_state=42, max_depth=4)


# In[318]:


dt_base.fit(X_train, y_train)


# In[319]:


y_train_pred = dt_base.predict(X_train)
y_test_pred = dt_base.predict(X_test)


# In[320]:


from sklearn.metrics import classification_report


# In[321]:


print(classification_report(y_test, y_test_pred))


# #### Plot the ROC curve

# In[324]:


from sklearn.metrics import RocCurveDisplay


# In[326]:


# plot_roc_curve(dt_base, X_train, y_train, drop_intermediate=False)
RocCurveDisplay.from_estimator(dt_base, X_train, y_train)
plt.show()


# #### Hyper-parameter tuning for the Decision Tree

# In[327]:


from sklearn.model_selection import GridSearchCV


# In[328]:


dt = DecisionTreeClassifier(random_state=42)


# In[329]:


params = {
    "max_depth": [2,3,5,10,20],
    "min_samples_leaf": [5,10,20,50,100,500]
}


# In[330]:


grid_search = GridSearchCV(estimator=dt,
                           param_grid=params,
                           cv=4,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[331]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)\n')


# In[332]:


grid_search.best_score_


# In[333]:


dt_best = grid_search.best_estimator_
dt_best


# In[335]:


# plot_roc_curve(dt_best, X_train, y_train)
RocCurveDisplay.from_estimator(dt_best, X_train, y_train)
plt.show()


# ## Using Random Forest

# In[336]:


from sklearn.ensemble import RandomForestClassifier


# In[337]:


rf = RandomForestClassifier(n_estimators=10, max_depth=4, max_features=5, random_state=100, oob_score=True)


# In[338]:


get_ipython().run_cell_magic('time', '', 'rf.fit(X_train, y_train)\n')


# In[340]:


rf.oob_score_


# In[341]:


# plot_roc_curve(rf, X_train, y_train)
RocCurveDisplay.from_estimator(rf, X_train, y_train)
plt.show()


# ### Hyper-parameter tuning for the Random Forest

# In[342]:


rf = RandomForestClassifier(random_state=42, n_jobs=-1)


# In[343]:


params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10, 25, 50, 100]
}


# In[344]:


grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[345]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)\n')


# In[346]:


grid_search.best_score_


# In[347]:


rf_best = grid_search.best_estimator_
rf_best


# In[349]:


# plot_roc_curve(rf_best, X_train, y_train)
RocCurveDisplay.from_estimator(rf_best, X_train, y_train)
plt.show()


# In[350]:


rf_best.feature_importances_


# In[351]:


imp_df = pd.DataFrame({
    "Varname": X_train.columns,
    "Imp": rf_best.feature_importances_
})


# In[352]:


imp_df.sort_values(by="Imp", ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:




