










# Home Challenge Modelling

#### Task: Build propensity to pay model

###### Author: Jacky Barraza




## Technical Set-Up

The first step I carried out was to create an environment on pycharm for running notebooks on it, and set up the libraries, I planned to use to work on a simple machine learning task. 



## Methodology

To address the challenge, I first need to define a high-level plan of tasks. For this, I will first look at the data.



## Initial Exploration of the Data

I have loaded the database files as a pandas dataframe and carried out general data exploration on the content of the dataset. This gave me a big picture of the condition of the data and how I might approach it. Below I will describe the plan to carry out the tasks.



## Plan

The following is a high-level plan of tasks. I will use this plan as a general but flexible guide when developing. I am going to use as inspiration the "Machine Learning Project Checklist" in [1], to select tasks that are relevant to this project.

*[1] Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn & TensorFlow. 2017.*



1. Frame the problem
   	Define a performance metric.

2. Explore and Prepare the Data

   ​	a) Data exploration tasks:
   ​			Get some basic information about the data
   ​			Check for missing values, check datatypes
   ​			Study correlations between variables
   ​			Visualize the data.
   ​	b) Data transformation tasks:
   ​			If there are missing values, define a strategy to deal with them.
   ​			Feature Engineering: New variable by relationship on other variables (ratios), decompose categorical features.
   ​			Defining the target variable (it is already defined), check it against other variables

   ​	c) Data selection and final preparation:

   ​			Feature selection: exploring variables that correlated with the target and variance threshold.
   ​			Feature scaling, choose Validation method and metrics

3. Build a model with the approach of exploring models for selection following the next steps:

​			a) First step: decide on a set of models that can apply to the hypothesis and evaluate it

​			b) Second step: as it is a unbalanced dataset, it will apply a set of techniques to deal with it a check models performance

​			c) Third step: decide which model to use as the final model and plan for training and tuning it.

4. Evaluate the model

​			Use the final model to classify the test sampling  and measure its performance.

5. Discussion

​			Analyze results and define which could be the next steps.



## Plan execution



### 1. Frame the Problem

The general purpose it is interpreted as a task to build a "propensity to pay model", so applying machine learning machine learning it could be possible to reduce uncertanity/check the effiencie on the goal of debt collection on the company's strategies. Also, it could be a tool that can be used to define which specific techniques works when contacting "entities" and checking company's performance on reaching goals on debt collection during a specific time, in this case 90 days.



The problem is a classification problem. Some of the algorithms that will test are Logistic Regression, Decision Tree and Random Forest. It is going to be decided depending on time dedicated to the data preparation if other algorithms will be tested. 



### 2. Data transformation and exploration 

###### 1_Notebook_EDA_and_EngVariable

[.../PropensityToPayModelByJackyB/html_files/1_Notebook_EDA_and_EngVar.html]



In this first notebook, it will be found all related to data cleaning, analysis, and preparation for modeling. 



The first overview provided information on the size 30,000 rows and 18 columns, nulls and outlayers were found and it was not found duplicates rows. The actions to handle outlayers and NaN were applied for each variable. The premise to input NaN was thought on reducing the impact on data distribution, meaning creating a minimal impact on mean and std. 



The target variable is defined already in the dataset.



Different analysis was performed for understanding the data and the correlation was based on analysis of univariant and bivariant through  building histograms, Xplots, boxplots, barplots and time  plots. A section of  “Observations” was described to highlight relevant information that drives decisions.



#### Output

As final output, a dataframe was exported to move to the next steps.

../PropensityToPayModel/data/processed/df_clean.csv



### 3.  Data selection and final preparation  

###### 2v1_Notebook_FeatureSel_Validation

[.../PropensityToPayModel/html_files/2_Notebook_FeatureSel_Validation.html]



On this section, it was explored the distribution of the target variable, where it was confirmed the unbalanced dataset with target=0 83.1% - target=1 - 16.9%. 



This information will be considered on the modeling phase. It is important to mention that unbalanced dataset can cause noise in the model, it will affect the generalization proccess of the samples, and it will affect the minirotary class which is the pay (1), when no default/not pay(0) is around four times more than default.



##### Feature Selection

The strategy applied for feature selection was based on variance threhold, variables with a high variance may contain more useful information for the model. 



##### Data Normalization

I used the MinMax to normalize the data.



##### Metrics

The main metric to consider for comparing and validating the models will be ROC-AUC score. The reason is that on this model, I care about the "pay" samples, and the class is unbalanced. The reason for choosing the ROC-AUC score is because it is independent of the threshold set for classification and because it only considers the rank of each prediction and not its absolute value.



Other metrics I am checking in the models are:

​	F1 Score: It is a harmonic mean of precision and recall given by- 

​								F1 = 2*Precision*Recall / (Precision + Recall) 

​	This is because will provide the model performance.

​	Matrix of Confusion, where I am looking to get a good performance on the false negative, as it will have a higher impact on predicting if an "entity" is paying and result in default.



##### Validation

It was decided to select the technique of train/test split as the main technic. Divided by a percentage 70-30% or 80-20%. I am going to try it and choose the one that provides  better metrics. I am adding the applying "stratify" parameter because it will help to make a split so that the proportion of values in the sample produced will be the same as the proportion of values provided. For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones.



Another technique applied is the Kfold during the model calculation.



#### Output

The output for this section is the X and y arrays, besides the train, val, and test datasets. These files are located on the following path of the data structure:



> .../PropensityToPayModel/data/output/X.csv
>
> .../PropensityToPayModel/data/output/y.csv
>
> .../PropensityToPayModel/data/output/X_train.csv
>
> .../PropensityToPayModel/data/output/y_train.csv
>
> .
>
> .
>
> .
>
> .../PropensityToPayModel/data/output/y_test.csv



### 4.  Building models



In this phase, different models were built. There are three notebooks related to this phase. Each notebook has the model construction and evaluation. 



###### 3v1_Notebook_Modeling

[.../PropensityToPayModel/html_files/3v1_Notebook_Modeling.html]



The first approach is decided to build models selected with the input as it is. The models were created with simple parameters. The best model performance following the AUC, are the Random Forest, Logistic Regression, and Decision Tree. However, if I look at the F1 score and the confusion matrix, the XGBClassifier gives a descent number in comparison with the other models. Therefore, the models mentioned will pass to the next approach. 



###### 3v2_Notebook_Modeling

[.../PropensityToPayModel/html_files/3v2_Notebook_Modeling.html]



The second approach,  is to select the best candidate and performance strategies for an unbalanced dataset and evaluate the models. 



After running the model's set selected in this phase and exploring different techniques, here are some general comments on the model:

- Undersampling method performance the worst as expected. 
- There is not too much variation between the results on the original input and the oversampling method. 
- The Random Forest algorithm has a considering improve on AUC for the train 0.61 to 0.82, I must mention that the hyperparameters were a bit different than in the first approach, however, I don't like the difference between the AUC_train: 0.82  and AUC_val: 0.62. 
- Regarding F1 score, Logistic Regression performs better than the other models by getting f1_train: 0.34 and f1_val:0.36. The AUC_train was 0.60 and AUC_val 0.63. 
- The XGBoost, gave AUC_train 0.63 and AUC_val: 0.55, and f1_train:0.42 and f1_val: 0.21. However, looking at the coffusion matrix, got the best results TP:871% , FP: 9.2% , FN:12.9%, TN:90.8%. However, accuracy is not appropriate when the data is imbalanced. Because the model can achieve higher accuracy by just predicting accurately the majority class while performing poorly on the minority class which in most cases is the class we care about the most.
- Regarding the dataset conditions, I am going to select these three models for the next step, considering as the best option the Logistic Regression, because has the most consistent realists regarding the difference between train and val. The second option is the Random Forest because has the best  f1 score.



###### 3v3_Notebook_Modeling

[.../PropensityToPayModel/html_files/3v3_Notebook_Modeling.html]



The third and last approach is to select the best candidate for the second approach and build models applying strategies to find the best hyperparameters. 



The main metric to evaluate the model selection is the AUC because of the unbalance condition as it was mentioned in the above section. 



After tunning the hyperparameters, the results ranked the Random Forest algorithm as the best model with an AUC_train of 0.797 and an AUC_val of 0.63. The f1_train 0.56 and the f1_val 0.37. However, the confusion matrix results worst that in the second approach. On the other hand, the Logistic Regression shows result similars to the second approach. Finally, the XGB will be set aside.



## Discussion



I would like to mention a couple of points open for discussios:



##### The variables: 

​	- The variable "month", has a small degree of importance for the model, so even if the variable was not use on the best approach, might have a low impact on the model. It will really know when the process is repeated without this variable.



​	- The variable that has a higher degree of importance for the models was c_0039_K6622, d0012, c0019.



##### The models:

​	- A set of algorithms of machine learning were tested, going from a simple approach to an elaborated one. This process helps to narrow down the options to explore in the open cases.

​	- From my perspective, it is good to have at least two options at the point of the process of tunning hyperparameter, this action provides the options to have a better choice by comparing results. 

​	- The model selected could have a better performance by doing a bit more hyperparameter exploration by applying edge technology that can be found in AWS or Azure.



##### The structure of the repo:

​	- Even if this was not used, I think it is handled if all the projects we work on it have a strucure and a very well described and repetitive process to processing the data and analyzing the variables.



##### The points that can help to improve the model

​	- A good strategy to build models it is to have a clear business objective and make decisions based on which insights might want to be draw eventually. 



​	- The data collection is the key for the consistency of the model. From my perspective, might help the model by including variables on demography, culture, etc. We are all human beings and at the end of the day, we might have patterns as collectives, which can be used to develop strategies to be assertive on the approaches with the debtors. 







