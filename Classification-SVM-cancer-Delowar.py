#!/usr/bin/env python
# coding: utf-8

# # SVM (Support Vector Machines)
# 
# ## Objectives
# 
# *   Use scikit-learn to Support Vector Machine to classify
# 

# In this notebook, you will use SVM (Support Vector Machines) to build and train a model using human cell records, and classify cells to whether the samples are benign or malignant.
# 
# SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the data are not otherwise linearly separable. A separator between the categories is found, then the data is transformed in such a way that the separator could be drawn as a hyperplane. Following this, characteristics of new data can be used to predict the group to which a new record should belong.
# 

# <h1>Table of contents</h1>
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ol>
#         <li><a href="https://#load_dataset">Load the Cancer data</a></li>
#         <li><a href="https://#modeling">Modeling</a></li>
#         <li><a href="https://#evaluation">Evaluation</a></li>
#         <li><a href="https://#practice">Practice</a></li>
#     </ol>
# </div>
# <br>
# <hr>
# 

# In[1]:


get_ipython().system('pip install scikit-learn==0.23.1')


# In[2]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# <h2 id="load_dataset">Load the Cancer data</h2>
# The example is based on a dataset that is publicly available from the UCI Machine Learning Repository (Asuncion and Newman, 2007)[http://mlearn.ics.uci.edu/MLRepository.html]. The dataset consists of several hundred human cell sample records, each of which contains the values of a set of cell characteristics. The fields in each record are:
# 
# | Field name  | Description                 |
# | ----------- | --------------------------- |
# | ID          | Clump thickness             |
# | Clump       | Clump thickness             |
# | UnifSize    | Uniformity of cell size     |
# | UnifShape   | Uniformity of cell shape    |
# | MargAdh     | Marginal adhesion           |
# | SingEpiSize | Single epithelial cell size |
# | BareNuc     | Bare nuclei                 |
# | BlandChrom  | Bland chromatin             |
# | NormNucl    | Normal nucleoli             |
# | Mit         | Mitoses                     |
# | Class       | Benign or malignant         |
# 
# <br>
# <br>

# ## Load Data From CSV File
# 

# In[4]:


cell_df = pd.read_csv("cell_samples.csv")
cell_df.head()


# The ID field contains the patient identifiers. The characteristics of the cell samples from each patient are contained in fields Clump to Mit. The values are graded from 1 to 10, with 1 being the closest to benign.
# 
# The Class field contains the diagnosis, as confirmed by separate medical procedures, as to whether the samples are benign (value = 2) or malignant (value = 4).
# 
# Let's look at the distribution of the classes based on Clump thickness and Uniformity of cell size:
# 

# In[5]:


ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()


# ## Data pre-processing and selection
# 

# Let's first look at columns data types:
# 

# In[6]:


cell_df.dtypes


# It looks like the **BareNuc** column includes some values that are not numerical. We can drop those rows:
# 

# In[7]:


cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes


# In[8]:


feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]


# We want the model to predict the value of Class (that is, benign (=2) or malignant (=4)). As this field can have one of only two possible values, we need to change its measurement level to reflect this.
# 

# In[9]:


cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
y [0:5]


# ## Train/Test dataset
# 

# We split our dataset into train and test set:
# 

# In[10]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# <h2 id="modeling">Modeling (SVM with Scikit-learn)</h2>
# 

# The SVM algorithm offers a choice of kernel functions for performing its processing. Basically, mapping data into a higher dimensional space is called kernelling. The mathematical function used for the transformation is known as the kernel function, and can be of different types, such as:
# 
# ```
# 1.Linear
# 2.Polynomial
# 3.Radial basis function (RBF)
# 4.Sigmoid
# ```
# 
# Each of these functions has its characteristics, its pros and cons, and its equation, but as there's no easy way of knowing which function performs best with any given dataset. We usually choose different functions in turn and compare the results. Let's just use the default, RBF (Radial Basis Function) for this lab.
# 

# In[11]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 


# After being fitted, the model can then be used to predict new values:
# 

# In[12]:


yhat = clf.predict(X_test)
yhat [0:5]


# <h2 id="evaluation">Evaluation</h2>
# 

# In[13]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[14]:


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


# In[15]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')


# You can also easily use the **f1\_score** from sklearn library:
# 

# In[16]:


from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 


# Let's try the jaccard index for accuracy:
# 

# In[17]:


from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,pos_label=2)


# <h2 id="practice">Practice</h2>
# Can you rebuild the model, but this time with a __linear__ kernel? You can use __kernel='linear'__ option, when you define the svm. How the accuracy changes with the new kernel function?
# 

# In[18]:


# write your code here
clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train) 
yhat2 = clf2.predict(X_test)
print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_score(y_test, yhat2,pos_label=2))


# <details><summary>Click here for the solution</summary>
# 
# ```python
# clf2 = svm.SVC(kernel='linear')
# clf2.fit(X_train, y_train) 
# yhat2 = clf2.predict(X_test)
# print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
# print("Jaccard score: %.4f" % jaccard_score(y_test, yhat2,pos_label=2))
# 
# ```
# 
# </details>
# 
