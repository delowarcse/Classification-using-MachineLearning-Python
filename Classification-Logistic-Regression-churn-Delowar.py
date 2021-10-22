#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression with Python
# 
# ## Objectives
# 
# *   Use scikit Logistic Regression to classify
# *   Understand confusion matrix
# 

# In this notebook, you will learn Logistic Regression, and then, you'll create a model for a telecommunication company, to predict when its customers will leave for a competitor, so that they can take some action to retain the customers.
# 

# <h1>Table of contents</h1>
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ol>
#         <li><a href="https://#about_dataset">About the dataset</a></li>
#         <li><a href="https://#preprocessing">Data pre-processing and selection</a></li>
#         <li><a href="https://#modeling">Modeling (Logistic Regression with Scikit-learn)</a></li>
#         <li><a href="https://#evaluation">Evaluation</a></li>
#         <li><a href="https://#practice">Practice</a></li>
#     </ol>
# </div>
# <br>
# <hr>
# 

# <a id="ref1"></a>
# 
# ## What is the difference between Linear and Logistic Regression?
# 
# While Linear Regression is suited for estimating continuous values (e.g. estimating house price), it is not the best tool for predicting the class of an observed data point. In order to estimate the class of a data point, we need some sort of guidance on what would be the <b>most probable class</b> for that data point. For this, we use <b>Logistic Regression</b>.
# 
# <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# <font size = 3><strong>Recall linear regression:</strong></font>
# <br>
# <br>
#     As you know, <b>Linear regression</b> finds a function that relates a continuous dependent variable, <b>y</b>, to some predictors (independent variables $x_1$, $x_2$, etc.). For example, simple linear regression assumes a function of the form:
# <br><br>
# $$
# y = \theta_0 + \theta_1  x_1 + \theta_2  x_2 + \cdots
# $$
# <br>
# and finds the values of parameters $\theta_0, \theta_1, \theta_2$, etc, where the term $\theta_0$ is the "intercept". It can be generally shown as:
# <br><br>
# $$
# ℎ_\theta(𝑥) = \theta^TX
# $$
# <p></p>
# 
# </div>
# 
# Logistic Regression is a variation of Linear Regression, useful when the observed dependent variable, <b>y</b>, is categorical. It produces a formula that predicts the probability of the class label as a function of the independent variables.
# 
# Logistic regression fits a special s-shaped curve by taking the linear regression function and transforming the numeric estimate into a probability with the following function, which is called the sigmoid function 𝜎:
# 
# $$
# ℎ\_\theta(𝑥) = \sigma({\theta^TX}) =  \frac {e^{(\theta\_0 + \theta\_1  x\_1 + \theta\_2  x\_2 +...)}}{1 + e^{(\theta\_0 + \theta\_1  x\_1 + \theta\_2  x\_2 +\cdots)}}
# $$
# Or:
# $$
# ProbabilityOfaClass\_1 =  P(Y=1|X) = \sigma({\theta^TX}) = \frac{e^{\theta^TX}}{1+e^{\theta^TX}}
# $$
# 
# In this equation, ${\theta^TX}$ is the regression result (the sum of the variables weighted by the coefficients), `exp` is the exponential function and $\sigma(\theta^TX)$ is the sigmoid or [logistic function](http://en.wikipedia.org/wiki/Logistic_function?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01), also called logistic curve. It is a common "S" shape (sigmoid curve).
# 
# So, briefly, Logistic Regression passes the input through the logistic/sigmoid but then treats the result as a probability:
# 
# <img
# src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/images/mod_ID_24_final.png" width="400" align="center">
# 
# The objective of the **Logistic Regression** algorithm, is to find the best parameters θ, for $ℎ\_\theta(𝑥)$ = $\sigma({\theta^TX})$, in such a way that the model best predicts the class of each case.
# 

# ### Customer churn with Logistic Regression
# 
# A telecommunications company is concerned about the number of customers leaving their land-line business for cable competitors. They need to understand who is leaving. Imagine that you are an analyst at this company and you have to find out who is leaving and why.
# 

# In[ ]:


get_ipython().system('pip install scikit-learn==0.23.1')


# Let's first import required libraries:
# 

# In[2]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# <h2 id="about_dataset">About the dataset</h2>
# We will use a telecommunications dataset for predicting customer churn. This is a historical customer dataset where each row represents one customer. The data is relatively easy to understand, and you may uncover insights you can use immediately. Typically it is less expensive to keep customers than acquire new ones, so the focus of this analysis is to predict the customers who will stay with the company. 
# 
# This data set provides information to help you predict what behavior will help you to retain customers. You can analyze all relevant customer data and develop focused customer retention programs.
# 
# The dataset includes information about:
# 
# *   Customers who left within the last month – the column is called Churn
# *   Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# *   Customer account information – how long they had been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# *   Demographic info about customers – gender, age range, and if they have partners and dependents
# 

# ### Load the Telco Churn data
# 
# Telco Churn is a hypothetical data file that concerns a telecommunications company's efforts to reduce turnover in its customer base. Each case corresponds to a separate customer and it records various demographic and service usage information. Before you can work with the data, you must use the URL to get the ChurnData.csv.

# ## Load Data From CSV File
# 

# In[3]:


churn_df = pd.read_csv("ChurnData.csv")
churn_df.head()


# <h2 id="preprocessing">Data pre-processing and selection</h2>
# 

# Let's select some features for the modeling. Also, we change the target data type to be an integer, as it is a requirement by the skitlearn algorithm:
# 

# In[4]:


churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()


# ## Practice
# 
# How many rows and columns are in this dataset in total? What are the names of columns?
# 

# In[5]:


# write your code here
churn_df.shape


# <details><summary>Click here for the solution</summary>
# 
# ```python
# churn_df.shape
# 
# ```
# 
# </details>
# 

# Let's define X, and y for our dataset:
# 

# In[6]:


X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]


# In[7]:


y = np.asarray(churn_df['churn'])
y [0:5]


# Also, we normalize the dataset:
# 

# In[8]:


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ## Train/Test dataset
# 

# We split our dataset into train and test set:
# 

# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# <h2 id="modeling">Modeling (Logistic Regression with Scikit-learn)</h2>
# 

# Let's build our model using **LogisticRegression** from the Scikit-learn package. This function implements logistic regression and can use different numerical optimizers to find parameters, including ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers. You can find extensive information about the pros and cons of these optimizers if you search it in the internet.
# 
# The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to solve the overfitting problem of machine learning models.
# **C** parameter indicates **inverse of regularization strength** which must be a positive float. Smaller values specify stronger regularization.
# Now let's fit our model with train set:
# 

# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# Now we can predict using our test set:
# 

# In[11]:


yhat = LR.predict(X_test)
yhat


# **predict_proba**  returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of class 0, P(Y=0|X), and second column is probability of class 1, P(Y=1|X):
# 

# In[12]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# <h2 id="evaluation">Evaluation</h2>
# 

# ### jaccard index
# 
# Let's try the jaccard index for accuracy evaluation. we can define jaccard as the size of the intersection divided by the size of the union of the two label sets. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
# 

# In[13]:


from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,pos_label=0)


# ### confusion matrix
# 
# Another way of looking at the accuracy of the classifier is to look at **confusion matrix**.
# 

# In[14]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))


# In[15]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')


# Look at first row. The first row is for customers whose actual churn value in the test set is 1.
# As you can calculate, out of 40 customers, the churn value of 15 of them is 1.
# Out of these 15 cases, the classifier correctly predicted 6 of them as 1, and 9 of them as 0.
# 
# This means, for 6 customers, the actual churn value was 1 in test set and classifier also correctly predicted those as 1. However, while the actual label of 9 customers was 1, the classifier predicted those as 0, which is not very good. We can consider it as the error of the model for first row.
# 
# What about the customers with churn value 0? Lets look at the second row.
# It looks like  there were 25 customers whom their churn value were 0.
# 
# The classifier correctly predicted 24 of them as 0, and one of them wrongly as 1. So, it has done a good job in predicting the customers with churn value 0. A good thing about the confusion matrix is that it shows the model’s ability to correctly predict or separate the classes.  In a specific case of the binary classifier, such as this example,  we can interpret these numbers as the count of true positives, false positives, true negatives, and false negatives.
# 

# In[16]:


print (classification_report(y_test, yhat))


# Based on the count of each section, we can calculate precision and recall of each label:
# 
# *   **Precision** is a measure of the accuracy provided that a class label has been predicted. It is defined by: precision = TP / (TP + FP)
# 
# *   **Recall** is the true positive rate. It is defined as: Recall =  TP / (TP + FN)
# 
# So, we can calculate the precision and recall of each class.
# 
# **F1 score:**
# Now we are in the position to calculate the F1 scores for each label based on the precision and recall of that label.
# 
# The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. It is a good way to show that a classifer has a good value for both recall and precision.
# 
# Finally, we can tell the average accuracy for this classifier is the average of the F1-score for both labels, which is 0.72 in our case.
# 

# ### log loss
# 
# Now, let's try **log loss** for evaluation. In logistic regression, the output can be the probability of customer churn is yes (or equals to 1). This probability is a value between 0 and 1.
# Log loss( Logarithmic loss) measures the performance of a classifier where the predicted output is a probability value between 0 and 1.
# 

# In[17]:


from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


# <h2 id="practice">Practice</h2>
# Try to build Logistic Regression model again for the same dataset, but this time, use different __solver__ and __regularization__ values? What is new __logLoss__ value?
# 

# In[18]:


# write your code here
LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))


# <details><summary>Click here for the solution</summary>
# 
# ```python
# LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
# yhat_prob2 = LR2.predict_proba(X_test)
# print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))
# 
# ```
# 
# </details>
# 
