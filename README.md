# KappaVoid

A team of (aspiring?) Data Scientists having adventures at Kaggle. Here we will describe our approach to the Titanic problem.

## Titanic Problem 

Based on the sinking of the RMS Titanic, that ended up killing 1502 out of 2224 passengers and crew. One of the reasons for such loss was that there were not enough lifeboats for the passengers and crew. Some groups of people were more likely to survive than others. In this challenge you are requested to **analyse** data applying *machine learning* and **predict** which passenger survived the tragedy
**Link: https://www.kaggle.com/c/titanic**

## Models Implemented

Right now we have implemented 9 *Machine Learning* models.

1) RandomForest
2) LinearSVC
3) Stochasthic
4) Gradient Descent
5) Gaussian Naive Bayes
6) K-Neighbors
7) Perceptron
8) DecisionTree
9) Logistic Regression

## Data Insights

Here are some insights we had analysing the Data.
- Pclass, Sex, Cabin and Embarked are **Categorical _features_**.
- Comparing *Genders*, Females are way more likely to survive.
- The fares didn't contribute much to the model
- We decided to unite Age and Pclass due to the correlation with results
- Names are unique in the dataset, so they are useless without preprocessing
- Dividing age *feature* by groups is important to improve Machine Learning performance. 
