# Udacity Capstone Project - Customer Segmentation and Prediction

### Problem Statement
The main aim of this project is to create a Customer Segmentation Report for Arvato Financial Solutions. This project is a part of Udacity Machine Learning Engineer Nanodegree Capstone Project in collaboration with Bertelsmann Arvato. 
This challenge is a real-life problem, provided by Arvato Financial Solutions, where the problem statement is:

***How can their client, a mail-order company, gets new clients efficiently using data-driven approach for targeted marketing?***

### Approach

Demographics information (like: age, income, wealth, education, assets, cars, houses, family, etc.) of the general population of Germany and of the existing customers of the mail-order company, was provided as datasets by Arvato. The data is protected and not allowed to be used by general public under Arvato's terms and conditions.

This demographic information is to be used to identify customer segments of mail-order company, with the aim to improve their targeted marketing compaigns and predict new customer conversion.

Customer Segmentation was performed using ML unsupervised learning techniques to identify the clusters of the population that best describe the core customer base of the company. 

After analyzing the core customer base, ML supervised learning techniques were applied on marketing campaign data to predict people who are likely to become new customers.

## Installation

The packages that need to be installed are mentioned in the requirement.txt file.

>> pip install -r requirement.txt

cmake = 3.18.4.post1
xgboost = 1.4.1
lightgbm = 3.2.1
scikit-learn = 0.23.2
scikit-optimize = 0.8.1
kneed = 0.7.0
numpy = 1.19.5

## Data
There are four data files associated with this project:

-   `Udacity_AZDIAS_052018.csv`: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
-   `Udacity_CUSTOMERS_052018.csv`: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
-   `Udacity_MAILOUT_052018_TRAIN.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
-   `Udacity_MAILOUT_052018_TEST.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).
-   `DIAS_Attributes_Values_2017.xlsx`: Values- level iInformation about attributes used in data.
-   `DIAS_Information_Levels_Attributes_2017_Komplett.xlsx`: Top-level information about attributes used in data.


## Code Structure
```
.
├── cross_validation_results --> (Not Uploaded) Folder contains csv which store best parameters and score of each iteration of bayes-optized model tuning.
├── models ----------------> (Not Uploaded) Folder of saved Models used for prediction 
├── Avrato_Project.ipynb --> Main Project Workbook
├── ML_Modeling.ipynb -----> Playground for evaluating different models
├── helper.py--------------> Contains different helper functions used in Workbooks
├── requirement.txt ------------------> Python packages installation file
├── Readme.md-------------------------> README File
```
## Steps to Complete This Project

The project has three major steps: the customer segmentation report, the supervised learning model, and the Kaggle Competition.

#### 1. Customer Segmentation Report

This section will be similar to the corresponding project in Term 1 of the program, but the datasets now include more features that you can potentially use. You'll begin the project by using unsupervised learning methods to analyze attributes of established customers and the general population in order to create customer segments.

#### 2. Supervised Learning Model

You'll have access to a third dataset with attributes from targets of a mail order campaign. You'll use the previous analysis to build a machine learning model that predicts whether or not each individual will respond to the campaign.

#### 3. Kaggle Competition

Once you've chosen a model, you'll use it to make predictions on the campaign data as part of a Kaggle Competition. You'll rank the individuals by how likely they are to convert to being a customer, and see how your modeling skills measure up against your fellow students.

## Detailed Break-down of Steps

**Part 0: Get to Know the Data**
- Import the data
- Exploratory Data Analysis (EDA) 
- Data Cleaning and Preprocessing
-- Covert unknown values to NaN
-- Convert special characters to NaN
-- Delete columns with higher percentage of missing values
- Feature Engineering 
- Feature Scaling (MinMax)


**Part 1: Customer Segmentation Report**
- Dimensionality Reduction using PCA
- K-means Clustering
- Customer Segmentation report using Cluster Mapping
- Component Makeup Analysis

**Part 2: Supervised Learning Model**
- Import the training data
- Perform data pre-processing steps: data cleaning, features engineering.
- Normalize the train data and save the scaler to be used later for test data.
- Select features based on feature importance
- Creating Baseline Models
- Selecting Model with high performance (using ROC-AUC metric) and low resource requirements
- Hyper-Parameter Tuning using Bayes Optimizer Cross Validation
- Save the model to be used in next step


**Part 3: Kaggle Competition**

- Import the testing data
- Perform data pre-processing steps: data cleaning, features engineering and scaling.
- Normalize the test data using scaler created for train data.- 
- Make predictions using the model created in the supervised learning model part.
- Submit predictions to kaggle

 ## Kaggle Leadership Board

The Best Score of **0.80420** was achieved after uploading the prediction to Kaggle Competition. Results can be found on Kaggle official website  [https://www.kaggle.com/c/udacity-arvato-identify-customers/leaderboard](https://www.kaggle.com/c/udacity-arvato-identify-customers/leaderboard)

## Improvement
The model prediction can possibily be improved by 
- Doing better feature engineering and feature selection by gathering more domain knowledge in understanding the features better.
- Using bigger training set by also including the data used for customer segmentation.
- Increasing the number of iterations in the hyper-parameter tuning step.
- Increasing the number of PCA principle-components and maximum number of iterations for K-Means clustering, to generate better clusters and customer-cluster mapping, and later using this as a feature for supervised learning model.

## Author

Sohaib Zafar