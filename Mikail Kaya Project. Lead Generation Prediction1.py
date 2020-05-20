#!/usr/bin/env python
# coding: utf-8

# ## Mikail Kaya Final Project
# ## Project Guide  
# ------------  
# - [Project Overview](#project-overview)  
# - [Part 1: Data Preparation](#I)
# - [Part 2: Data Visualization](#II)
# - [Part 3: Machine Learning](#III)

# <details>
# <summary>
# Roadmap for Building Machine Learning Models
# </summary>
# <p>
# 
# 
#     1. Prepare Problem  
#     a) Define The Business Objective  
#     b) Select the datasets  
#     c) Load dataset  
#     d) Load libraries  
# 
# 
# **Data Pre-processing**  
# This is the first step in building a machine learning model. Data pre-processing refers to the transformation of data
# before feeding it into the model. It deals with the techniques that are used to convert unusable raw data into clean 
# reliable data.  
#   
# Since data collection is often not performed in a controlled manner, raw data often contains outliers 
# (for example, age = 120), nonsensical data combinations (for example, model: bicycle, type: 4-wheeler), missing values, 
# scale problems, and so on. Because of this, raw data cannot be fed into a machine learning model because it might 
# compromise the quality of the results. As such, this is the most important step in the process of data science.  
#   
# 
#     2. Summarize Data  
#     a) Descriptive statistics  
#     b) Data visualizations  
# 
#     3. Prepare Data  
#     a) Data Cleaning  
#     b) Feature Selection  
#     c) Data Transformation  
# 
# **Model Learning**  
# After pre-processing the data and splitting it into train/test sets, I move on to modeling. Models are nothing but sets of well-defined methods called algorithms that use pre-processed data to learn patterns, which can later be used to make predictions. There are different types of learning algorithms, including supervised, semi-supervised, unsupervised, and reinforcement learning.
#   
#     4. Modeling Strategy  
#     a) Select Suitable Algorithms  
#     b) Select Training/Testing Approaches  
#     c) Train   
#   
#   
# **Model Evaluation**  
# In this stage, the models are evaluated with the help of specific performance metrics. With these metrics, I can go on to 
# tune the hyperparameters of a model in order to improve it. This process is called hyperparameter optimization. I will 
# repeat this step until satisfying with the performance.  
#   
#     4. Evaluate Algorithms  
#     a) Split-out validation dataset  
#     b) Test options and evaluation metric  
#     c) Spot Check Algorithms  
#     d) Compare Algorithms  
#   
# **Prediction**  
# Once I am happy with the results from the evaluation step, I will then move on to predictions. Predictions are made 
# by the trained model when it is exposed to a new dataset. In a business setting, these predictions can be shared with 
# decision makers to make effective business choices.  
#   
#     5. Improve Accuracy  
#     a) Algorithm Tuning  
#     b) Ensembles  
# 
# **Model Deployment**  
# The whole process of machine learning does not just stop with model building and prediction. It also involves making use 
# of the model to build an application with the new data. Depending on the business requirements, the deployment may be a 
# report, or it may be some repetitive data science steps that are to be executed. After deployment, a model needs proper 
# management and maintenance at regular intervals to keep it up and running.  
# 
#     6. Finalize Model  
#     a) Predictions on validation dataset  
#     b) Create standalone model on entire training dataset  
#     c) Save model for later use  
# 
# 
# </p>
# </details>

# ## Project Overview
# <details>
# <summary>About Lead Generation</summary>
# <p>
# 
# 
# Background:
# Lead generation is a challenge that businesses have faced since the dawn of capitalism. To understand lead generation, let's first introduce the idea of a "sales funnel."
# 
# Example Sales Funnel:
# A sales funnel is simply a series of expected steps that a prospect undergoes before buying. Most businesses require sales funnels because visitors are not ready to become customers right away.
# 
# Here's an example for a task-management SaaS business:
# •	Awareness: Karen, a project manager at an IT company, stumbles upon a blog post detailing 10 strategies for boosting the efficiency of meetings.
# •	Lead: She then enters her email into the opt-in box at the bottom of the blog post, which promises a PDF with 5 more strategies.
# •	Prospect: 2 weeks later, Karen receives an invitation to a webinar on how a better task management software can make her job easier. She attends.
# •	Sale: The webinar is informative and effective, so she agrees to a 3-month pilot trial for the software.
# 
# As you might guess, this process looks different for every business. Some require longer funnels that can span months (or even years) while others ask for the sale immediately.
# Lead generation, or capturing the contact info of people who could be interested in your company's product, is often regarded as the gateway to the rest of the sales funnel.
# That's why lead generation optimization will be likely be one of your most requested and valuable jobs as a data scientist.
# 
# Data:
# I have one table called lead_gen.csv.
# It contains 280,000 observations from a multi-channel marketing campaign. Each observation represents one session from one visitor. The table excludes data from visitors who are already leads.
#  
# ![image1.png](attachment:image.png)
# 
# Data Dictionary:
# •	Source - Marketing channel that visitor came from.
# •	returning - Has the visitor been to the website before?
# •	mobile - Device (mobile / desktop). Tablets count as mobile.
# •	country - Visitor country based on IP address.
# •	pages_viewed - Number of pages viewed in the session.
# •	lead - Did the visitor opt into email list during the session?
# 
# Objectives:
# For this project, a "conversion" is defined as a visitor who became a lead.
# •	First, determine which sources/countries/devices had the highest conversion rates.
# •	Next, build a model that can predict conversion rate based on visitor information.
# •	What insights can you draw from your model? Which features were the most impactful?
# •	Finally, provide actionable insights to the business. What have we learned from this campaign?
# 
# <p>
# 

# <a id="I"></a>
# 
# # I.  Data Preparation

# <details>
# <summary>About Pandas and Numpy</summary>
# <p>
# 
# **[Pandas](http://pandas.pydata.org)** is a Python library that provides extensive means for data analysis. Data scientists often work with data stored in table formats like `.csv`, `.tsv`, or `.xlsx`. Pandas makes it very convenient to load, process, and analyze such tabular data using SQL-like queries. In conjunction with `Matplotlib` and `Seaborn`, `Pandas` provides a wide range of opportunities for visual analysis of tabular data.
# 
# The main data structures in `Pandas` are implemented with **Series** and **DataFrame** classes. Series is a one-dimensional indexed array of some fixed data type. Data Frames is a two-dimensional data structure - a table - where each column contains data of the same type. You can see it as a dictionary of `Series` instances. `DataFrames` are great for representing real data: rows correspond to instances (examples, observations, etc.), and columns correspond to features of these instances.
# 
# **[Numpy](http://numpy.org)** is a free Python library equipped with a collection of complex mathematical operations suitable for processing statistical data.It is time to look at a few functions which will be critical to succes in understanding and completeing this assignment. All of the functions come from the "`numpy`" package.
# 
# The functions are:  
# 
# - `np.mean` - For calculating a mean
# - `np.std` -  For calculating standard deviation
# - `np.var` -  For calculating variance
# - `np.ptp` -  For calculating range (ptp stands for "point to point")
# - `np.sqrt` - For taking the square root of a number (instead of x \*\* .5)
# - `np.min` -  For finding the minimum value of a collection
# - `np.max` -  For finding the maximum value of a collection 
# 
# These functions work with most any type of collection: list, tuple, array, etc; through not with dictionaries.
# </p>
# </details>
# <details>
# <summary>About printing DataFrames in Jupyter notebooks</summary>
# <p>
# In Jupyter notebooks, Pandas DataFrames are printed as these pretty tables seen above while `print(df.head())` looks worse.
# By default, Pandas displays 20 columns and 60 rows, so, if your DataFrame is bigger, use the `set_option` function as shown in the example below:
# 
# ```python
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.max_rows', 100)
# ```
# Recall that each row corresponds to one client, an **instance**, and columns are **features** of this instance.
# 
# </p>
# </details>
# 
# 

# In[1]:


# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns # Seaborn is a Python data visualization library based on matplotlib
sns.set()  #  I will import Seaborn functionalities
import warnings # I don't like warnings
warnings.filterwarnings('ignore')


# 
# I’ll demonstrate the main methods in action by analyzing a dataset. I received 'Lead Generation' dataset from professor. Let’s read the data (using `read_csv`), and take a look at the first 5 lines using the `head` method:
# 

# In[2]:


# Disply all Columns

pd.options.display.max_columns=100


# In[5]:


lead_gen2 = pd.read_csv('lead_gen2.csv')
lead_gen2.head()


# Let’s have a look at data dimensionality, features names, and feature types.

# In[6]:


print(lead_gen2.shape)


# From the output, we can see that the table contains 287742 rows and 10 columns.
# 
# Now let’s try printing out column names using `columns`:

# In[7]:


print(lead_gen2.columns)


# We can use the `info()` method to output some general information about the dataframe: 

# In[8]:


lead_gen2.info()


# 
# The `describe` method shows basic statistical characteristics of each numerical feature (`int64` and `float64` types): number of non-missing values, mean, standard deviation, range, median, 0.25 and 0.75 quartiles.

# In[9]:


lead_gen2.describe() # Similar to summary() in R


# In[11]:


lead_gen2.describe().transpose()  # change the rows and columns


# In[13]:


# group by the 'label' and show descriptive stats
lead_gen2.groupby('lead').agg(['count', 'mean','std','min','max','median']).T.head(100)


# In order to see statistics on non-numerical features, one has to explicitly indicate data types of interest in the `include` parameter.

# In[11]:


lead_gen2['lead'].value_counts()


# 9248 visitors out of 287742 are *lead*; their `lead` value is `1`. 

# In[12]:


lead_gen2['lead'].value_counts(normalize=True)


# In[13]:


(lead_gen2['lead'].value_counts().plot(
        kind='bar',
        figsize=(8, 6),
        title='Distribution of Target Variable',
    )
);
plt.show()


# The target variable, `lead`, is **unbalanced** - meaning the target variable ('not lead') has more observations in one specific class ('0') than the other ('1'). This can present a problem since the **positive class** we want to predict is the "bad" loan class (`1.0`). You can find more info for [Unbalanced Datasets & What To Do About Them](https://medium.com/strands-tech-corner/unbalanced-datasets-what-to-do-144e0552d9cd)
# 
# Because of this unbalanced data, we will make sure that both our training set and testing set **maintain this ratio** of lead:not lead. This is acheived by using the `stratify` argument in the `train_test_split()` function, which was imported from the `sklearn.model_selection` module.

# In[14]:


lead_gen2['lead'].mean()


# 0.3% is actually quite bad rate for converting a visitor to lead.
# 

# 
# ##### Applying Functions to Cells, Columns and Rows
# 

# In[113]:


lead_gen2.apply(np.max) # np.max finds max value in the column.


# <a id="II"></a>
# # II. Data Visualization
# 

# ### 1. Demonstration of main Pandas and NumPy methods 
# <details>
# <summary>About Data Vizualization in Machine Learning</summary>
# <p>
# In the field of Machine Learning, *data visualization* is not just making fancy graphics for reports; it is used extensively in day-to-day work for all phases of a project.
# 
# To start with, visual exploration of data is the first thing one tends to do when dealing with a new task. We do preliminary checks and analysis using graphics and tables to summarize the data and leave out the less important details. It is much more convenient for us, humans, to grasp the main points this way than by reading many lines of raw data. It is amazing how much insight can be gained from seemingly simple charts created with available visualization tools.
# 
# Next, when we analyze the performance of a model or report results, we also often use charts and images. Sometimes, for interpreting a complex model, we need to project high-dimensional spaces onto more visually intelligible 2D or 3D figures.
# 
# All in all, visualization is a relatively fast way to learn something new about your data. Thus, it is vital to learn its most useful techniques and make them part of your everyday ML toolbox.
# 
# We are going to get hands-on experience with visual exploration of data using popular libraries such as `matplotlib` and `seaborn`.
# <p>

# ### 1. Univariate visualization
# 
# *Univariate* analysis looks at one feature at a time. When we analyze a feature independently, we are usually mostly interested in the *distribution of its values* and ignore other features in the dataset.
# 
# Below, I will consider different statistical types of features and the corresponding tools for their individual visual analysis.
# 
# #### 1.1 Quantitative features
# 
# *Quantitative features* take on ordered numerical values. Those values can be *discrete*, like integers, or *continuous*, like real numbers, and usually express a count or a measurement.
# 
# ##### 1.1.1 Histograms and density plots
# 
# The easiest way to take a look at the distribution of a numerical variable is to plot its *histogram* using the `DataFrame`'s method [`hist()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.hist.html).

# 
# <details>
# <summary>About Histograms</summary>
# <p>
# A histogram groups values into *bins* of equal value range. The shape of the histogram may contain clues about the underlying distribution type: Gaussian, exponential etc. You can also spot any skewness in its shape when the distribution is nearly regular but has some anomalies. Knowing the distribution of the feature values becomes important when you use Machine Learning methods that assume a particular type of it, most often Gaussian.
# 
# In the above plot, we see that the variable *Total day minutes* is normally distributed, while *Total intl calls* is prominently skewed right (its tail is longer on the right).
# 
# There is also another, often clearer, way to grasp the distribution: *density plots* or, more formally, *Kernel Density Plots*. They can be considered a [smoothed](https://en.wikipedia.org/wiki/Kernel_smoother) version of the histogram. Their main advantage over the latter is that they do not depend on the size of the bins. Let's create density plots for the same two variables:

# In[20]:


features = ['pages_viewed'] # Lead Generation dataset only quantitative feature is 'page_viewed'

lead_gen2[features].hist(figsize=(7, 7))


# In[21]:


# separate the data from the target attributes
X = lead_gen2.pages_viewed.head(10)
X


# ##### Log Normalization

# In[22]:


from sklearn import preprocessing
import numpy as np


# In[16]:


x_log=np.log(lead_gen2.pages_viewed)


# In[17]:


x_log


# In[68]:


plt.hist(x_log)


# ###### Add this normalize column to the dataset

# In[77]:


#lead_gen2 = pd.concat([lead_gen2, x_log], axis=1)


# In[78]:


lead_gen2.columns


# In[79]:


lead_gen2.drop(['pages_viewed'], axis=1, inplace=True)


# In[80]:


lead_gen2.columns


# In[81]:


lead_gen2 = pd.concat([lead_gen2, x_log], axis=1)


# In[82]:


lead_gen2.columns


# ##### Re-index the columns names

# In[96]:


lead_gen2=lead_gen2[['source_Taboola', 'source_Facebook', 'source_influencer',
       'source_Adwords', 'returning', 'mobile', 'country_USA',
       'country_Canada', 'country_UK', 'pages_viewed', 'lead']]


# In[97]:


lead_gen2.head()


# In[99]:


lead_gen2['pages_viewed'].plot(kind='density', subplots=True, layout=(5, 5), 
                  sharex=False, figsize=(30, 30));


# In[86]:


features = ['pages_viewed']
lead_gen2[features].hist(figsize=(7, 7))


# It is also possible to plot a distribution of observations with `seaborn`'s [`distplot()`](https://seaborn.pydata.org/generated/seaborn.distplot.html). For example, let's look at the distribution of *pages_viewed*. By default, the plot displays both the histogram with the [kernel density estimate](https://en.wikipedia.org/wiki/Kernel_density_estimation) (KDE) on top.

# In[87]:


# increasing the width of the Chart
import seaborn as sns
plt.rcParams['figure.figsize'] = 5,5 # similar to par(mfrow = c(2,1), mar = c(4,4,2,1)) # 2 columns and 1 row
sns.distplot(lead_gen2["pages_viewed"]) # pass it one variable

# if you are getting warnings related to the package you should use ignore function
import warnings
warnings.filterwarnings ('ignore')


# ### 2. Multivariate visualization
# 
# *Multivariate* plots allow us to see relationships between two and more different variables, all in one figure. Just as in the case of univariate plots, the specific type of visualization will depend on the types of the variables being analyzed.
# 
# #### 2.1 Quantitative–Quantitative
# 
# ##### 2.1.1 Correlation matrix
# 
# Let's look at the correlations among the numerical variables in the dataset. This information is important to know as there are Machine Learning algorithms (for example, linear and logistic regression) that do not handle highly correlated input variables well.
# 
# First, I will use the method [`corr()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) on a `DataFrame` that calculates the correlation between each pair of features. Then, I pass the resulting *correlation matrix* to [`heatmap()`](https://seaborn.pydata.org/generated/seaborn.heatmap.html) from `seaborn`, which renders a color-coded matrix for the provided values:

# In[101]:


#Note: If there are non-numerical features, 
# we should drop them. In this dataset, there is no non-numerical variables.

# Calculate Correlation. 
corr_matrix = lead_gen2.corr()
corr_matrix


# <h5>Highly correlated items = not good!</h5>
# <h5>Low correlated items = good </h5>
# <h5>Correlations with target (dv) = good (high predictive power)</h5>

# In[102]:


import matplotlib.pyplot as plot
plot.pcolor(lead_gen2.corr())
plot.show()


# In[103]:


# Correlation heatmap of the numberic variables
plt.rcParams['figure.figsize'] = 7,5  # control plot sizeimport seaborn as sns
sns.heatmap(lead_gen2.corr())


# In[104]:


# seaborn
## first_twenty = har_train.iloc[:, :20] # pull out first 20 feats
corr = lead_gen2.corr()  # compute correlation matrix
mask = np.zeros_like(corr, dtype=np.bool)  # make mask
mask[np.triu_indices_from(mask)] = True  # mask the upper triangle

fig, ax = plt.subplots(figsize=(9, 7))  # create a figure and a subplot
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # custom color map
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    center=0,
    linewidth=0.5,
    cbar_kws={'shrink': 0.5}
);


# #### Horizontal Boxplots

# In[106]:


# Boxplots
# Horizontal boxplot with observations
plt.rcParams['figure.figsize'] = 8,4
sns.boxplot(lead_gen2['pages_viewed'])


# The height of the histogram bars here is normed and shows the density rather than the number of examples in each bin.
# 
# ##### Vertical Boxplot
# 
# Another useful type of visualization is a *box plot*. `seaborn` does a great job here:

# In[107]:


plt.rcParams['figure.figsize'] = 4,8
sns.boxplot(x='pages_viewed', data=lead_gen2, orient = 'v');


# # III. Machine Learning

# ##### Data splitting

# In[18]:


X = lead_gen2.drop('lead', axis = 1)
y = lead_gen2.lead


# In[19]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

training_features, test_features, training_target, test_target, = train_test_split(lead_gen2.drop('lead', axis = 1),
                                               lead_gen2.lead,
                                               test_size = .2,
                                               random_state=12)


# In[20]:


# Import Classifiers packages

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier

from sklearn import tree


# In[21]:


# Confusion Matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[22]:


# Using a Dummy Classifier

from sklearn.dummy import DummyClassifier
dummy_baseline = DummyClassifier(strategy="most_frequent")
dummy_baseline.fit(test_features, test_target)

DummyClf = dummy_baseline.score(test_features, test_target)


# In[31]:


# Create classifiers

DTModel = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
NBModel = GaussianNB()
LRModel = LogisticRegression(random_state = 0, solver='newton-cg', max_iter=1000)
SVMModel = SVC(kernel = 'linear', random_state = 0)
KSVMModel = SVC(kernel = 'rbf', random_state = 0)
KNNModel = KNeighborsClassifier(n_neighbors = 57)#  , metric = 'minkowski', p = 2
RFModel = RandomForestClassifier( n_estimators=87, random_state=0)
#BModel = BaggingClassifier(tree.DecisionTreeClassifier(random_state=0))
#ADBModel = AdaBoostClassifier(random_state=0)
#GBModel = GradientBoostingClassifier(random_state=0)
#ExTrModel = ExtraTreesClassifier(random_state=0)
#RCModel = RidgeClassifier(random_state=0)


# ## Model Fitting
# ##### Decision Tree

# In[100]:


DTModel.fit(training_features, training_target)
DTPred = DTModel.predict(test_features)
ADT = DTModel.score(test_features, test_target)


# ##### Naive Bayes

# In[101]:


NBModel.fit(training_features, training_target)
NB_pred = NBModel.predict(test_features)
ACNB= accuracy_score(test_target, NB_pred)


# ##### Logistic Regression

# In[102]:


LRModel.fit(training_features, training_target)
LR_pred =LRModel.predict(test_features)
ACLR= accuracy_score(test_target, LR_pred )


# ##### Support Vector Machine

# In[103]:


SVMModel.fit(training_features, training_target)
SVM_pred = SVMModel.predict(test_features)
ACSVM= accuracy_score(test_target, SVM_pred )


# ##### K-Nearest Neighbors (KNN)

# In[111]:


KNNModel.fit(training_features, training_target)
KNN_pred = KNNModel.predict(test_features)
ACKNN= accuracy_score(test_target, KNN_pred )


# ##### Random Forest

# In[113]:


RFModel.fit(training_features, training_target)
RF_pred = RFModel.predict(test_features)
ACRF= accuracy_score(test_target, RF_pred )


# ##### Ensemble - Ada-Boost Prediction Accuracy 

# In[115]:


ADBModel.fit(training_features, training_target)
ADB_pred = ADBModel.predict(test_features)
ACADB = accuracy_score(test_target, ADB_pred )


# ##### Ensemble - Gradient Boosting Prediction Accuracy

# In[116]:


GBModel.fit(training_features, training_target)
GB_pred = GBModel.predict(test_features)
ACGB= accuracy_score(test_target, GB_pred )


# In[119]:


print(' The fraction of correct classifications is            : {:.2f}%'.format(DummyClf * 100))
print(" Logistic Regression Prediction Accuracy               : {:.2f}%".format(ACLR * 100))
print(" Support Vector Machine Prediction Accuracy            : {:.2f}%".format(ACSVM * 100))
print(" Ensemble - Ada-Boost Prediction Accuracy              : {:.2f}%".format(ACADB * 100))
print(" Random Forest Prediction Accuracy                     : {:.2f}%".format(ACRF * 100))
print(" Ensemble - Gradient Boosting Prediction Accuracy      : {:.2f}%".format(ACGB * 100))
print(" Decision Tree Prediction Accuracy                     : {:.2f}%".format(ADT * 100))
print(" Naive Bayes Prediction Accuracy                       : {:.2f}%".format(ACNB * 100))
print(" k Nearest Neighbors Prediction Accuracy               : {:.2f}%".format(ACKNN * 100))


# <h2 id="t3" style="margin-bottom: 18px">Confusion matrix</h2>
# 
# An interesting way to evaluate the results is by means of a confusion matrix, which shows the correct and incorrect predictions for each class. In the first row, the first column indicates how many classes 0 were predicted correctly, and the second column, how many classes 0 were predicted as 1. In the second row, we note that all class 1 entries were erroneously predicted as class 0.
# 
# Therefore, the higher the diagonal values of the confusion matrix the better, indicating many correct predictions.

# In[120]:


from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

conf_mat = confusion_matrix(y_true = test_target, y_pred = LR_pred)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
#cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
# fig.colorbar(cax)
# ax.set_xticklabels([''] + labels)
# ax.set_yticklabels([''] + labels)
sns.heatmap(conf_mat/np.sum(conf_mat), annot=True, fmt='.2%', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()


# <h2 id="t4" style="margin-bottom: 18px">Resampling</h2>
# 
# A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and / or adding more examples from the minority class (over-sampling).

# ![](https://raw.githubusercontent.com/rafjaa/machine_learning_fecib/master/src/static/img/resampling.png)

# Despite the advantage of balancing classes, these techniques also have their weaknesses (there is no free lunch). The simplest implementation of over-sampling is to duplicate random records from the minority class, which can cause overfitting. In under-sampling, the simplest technique involves removing random records from the majority class, which can cause loss of information.
# 
# Let's implement a basic example, which uses the <code>DataFrame.sample</code> method to get random samples each class:

# In[121]:


# Class count
#df_train = lead_gen2
count_class_0, count_class_1 = lead_gen2.lead.value_counts()

# Divide by class
df_class_0 = lead_gen2[lead_gen2['lead'] == 0]
df_class_1 = lead_gen2[lead_gen2['lead'] == 1]


# In[122]:


print(count_class_0)
print(count_class_1)


# ## Random under-sampling

# In[123]:


df_class_0_under = df_class_0.sample(count_class_1)
df_class_0_under.shape


# In[124]:


df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
df_test_under.shape


# In[127]:


print('Random under-sampling:')
print(df_test_under.lead.value_counts())

df_test_under.lead.value_counts().plot(kind='bar', title='Count (lead)');


# In[128]:


df_test_under.head()


# In[131]:


X = df_test_under.drop('lead', axis = 1)
y = df_test_under.lead


# In[133]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

training_features, test_features, training_target, test_target, = train_test_split(df_test_under.drop('lead', axis = 1),
                                               df_test_under.lead,
                                               test_size = .2,
                                               random_state=12)


# In[134]:


LRModelUnder = LogisticRegression(random_state = 0, solver='newton-cg', max_iter=1000)

LRModelUnder.fit(training_features, training_target)
LR_predUnder =LRModelUnder.predict(test_features)


ACLRUnder = accuracy_score(test_target, LR_predUnder)
print(" Logistic Regression Prediction Accuracy Under              : {:.2f}%".format(ACLRUnder * 100))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




