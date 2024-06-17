# CognoRise InfoTech Data Science Internship Repository 

Welcome to My Data Science Journey
In this repository, I'll be sharing my progress and experiences as I delve into the world of data science during the CognoRise InfoTech Data Science Internship. Get ready to join me on this exciting learning adventure!

 # Data Science Internship Overview 📅
This internship program spans 30 days, from June 01, 2024, to July 01, 2024. Join me on this journey as I engage in hands-on tasks designed to deepen my understanding of key concepts in data science and machine learning. Let's explore the fascinating world of data science together!  

# Tasks I will be involved in 🎯

# Task 2: Credit Card Fraud Detection 🕵️‍♂️
# Project Report: Credit Card Fraud Detection

## Table of Contents
- [Introduction](#1-introduction)
- [Data Overview](#2-data-overview)
- [Data Preprocessing](#3-data-preprocessing)
- [Model Development](#4-model-development)
- [Results and Analysis](#5-results-and-analysis)
- [Conclusion](#6-conclusion)

## 1. Introduction
Fraudulent activities in credit card transactions pose a significant threat to financial security. This project focuses on developing a robust fraud detection system using machine learning algorithms to safeguard against fraudulent transactions.

## 2. Data Overview
- **Dataset Size:** 284,807 transactions
- **Features:** Time, Amount, V1-V28 (anonymized)
- **No missing values detected**
- **1,081 duplicate rows removed (0.4% of total data)**
- **Class Imbalance:** 99.83% legitimate transactions, 0.17% fraudulent transactions

## 3. Data Preprocessing
- **Handling Class Imbalance:**
  - Over-sampling: Achieved balanced dataset (283,253 transactions each)
  - Under-sampling: Balanced dataset with 473 transactions each
  - SMOTE: Balanced dataset with 283,253 transactions each

- **Advanced Data Preprocessing:** 
The project commenced with a meticulous data preprocessing phase, handling 284,807 transactions with 31 columns. The dataset was thoroughly explored, revealing no missing values but detecting and removing 1,081 duplicate entries, showcasing a deep commitment to data integrity.

- **Innovative Class Imbalance Handling:** 
The addressing of class imbalance, a critical issue in fraud detection, was approached with sophistication through three distinct methods – over-sampling, under-sampling, and SMOTE. These techniques created balanced datasets, enabling the model to learn effectively from both classes.

## 4. Model Development
- **Model Used:** Random Forest Classifier
- **Best Parameters:** 
  - n_estimators: 300
  - min_samples_split: 2
  - min_samples_leaf: 1
  - max_features: 'log2'
  - max_depth: 10
- **Evaluation Metrics:**
  - Precision, Recall
  - F1-score, ROC AUC
- **Training and Test Accuracy:** 99% and 94.7% respectively

- **Robust Model Selection and Optimization:** 
Leveraging the power of Random Forest Classification, the project employed rigorous parameter tuning using RandomizedSearchCV to identify the best hyperparameters for the model. This meticulous optimization ensured the model's performance was fine-tuned for detecting credit card fraud.

## 5. Results and Analysis
- **Precision-Recall Curve Area:** 0.98
- **Average Precision-Recall Score:** 0.98
- **Strong performance in fraud detection with high accuracy and stability**
- **Feature Importance Analysis:** Identified key features influencing fraud detection

- **Comprehensive Model Evaluation:** 
The project's sophistication extended to the model evaluation stage. Various evaluation metrics such as precision, recall, F1-score, accuracy, ROC AUC score, and average precision-recall score were meticulously analyzed. This comprehensive evaluation allowed for a deep understanding of the model's performance on different balanced datasets.

## 6. Conclusion
The developed fraud detection model exhibits remarkable performance in identifying fraudulent credit card transactions. The project involved thorough data preprocessing, effective handling of class imbalance, and model optimization to achieve high accuracy and reliability in fraudulent activity detection.

By combining cutting-edge data preprocessing techniques, sophisticated class imbalance handling, meticulous model selection and optimization, comprehensive model evaluation, and insightful feature importance analysis, the project exemplifies a high level of sophistication and finesse in the domain of credit card fraud detection. The advanced methodologies employed and the in-depth analysis conducted make this project an exemplar of excellence in the field.                                 
 #Task 3: Titanic Survival Prediction🚢
 
 # Titanic Survival Prediction Project README.md

## Project Overview:

Welcome to the Titanic Survival Prediction project README. This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. The project is based on an in-depth analysis of a dataset containing 418 rows and 12 columns, with no missing values. The data includes a mix of integers, floating-point numbers, and objects, resulting in a memory usage of approximately 8.8 KB.

### Features Analysis:

- **SibSp & Parch:** The analysis shows an equal distribution of passengers with and without siblings/spouses aboard, as well as a majority of passengers traveling without parents or children.
- **Embarked:** The majority of passengers embarked from 'Cherbourg' (C), followed by 'Southampton' (S).

### Model Performance:

The model achieved remarkable performance metrics during evaluation:
- **Training & Testing Accuracy:** Both training and testing accuracies reached 100%, indicating a complete comprehension of the training data and perfect predictions on new data.
- **ROC AUC Score & Classification Report:** The model demonstrated perfect precision, recall, and F1-score for both Survived and Not Survived classes.

### Model Evaluation & Recommendations:

The model's exceptional performance metrics raise concerns about overfitting and data leakage. Recommendations include reviewing data preprocessing steps, cross-validation for generalizability, investigating feature importance, exploring other models/ensembles, and re-evaluating statistical test results.

### Data Preprocessing:

- Handled missing values in 'Age' and 'Fare' columns.
- Dropped unnecessary columns like 'Cabin', 'Name', 'PassengerId', and 'Ticket'.
- Encoded categorical features 'Sex' and 'Embarked' using LabelEncoder.

### Exploratory Data Analysis (EDA):

- Various visualizations were conducted to analyze and understand the data patterns and relationships.

### Model Training, Evaluation & Tuning:

- Trained a Random Forest Classifier model achieving 100% accuracy.
- Tuned hyperparameters to enhance model performance.

### Hypothesis Testing:

- Conducted a Mann-Whitney U Test with further investigation needed due to the obtained p-value.

### Conclusion:

The Titanic Survival Prediction project exhibits exceptional model performance. Further steps such as model validation, feature engineering, and additional validation are recommended to ensure robustness and generalization of the model.

For more detailed information, please refer to the complete report provided above.

Thank you for exploring the Titanic Survival Prediction project! 🚢✨

# Task 8: Fake News Prediction 📰

 # Fake News Prediction Project Report

## Introduction
The Fake News Prediction project is dedicated to developing a cutting-edge model that accurately classifies news articles as real or fake based on their content. Leveraging advanced machine learning techniques, the project focuses on preprocessing textual data, engineering relevant features, and building a proficient classification model to fulfill these objectives.

## Dataset Overview
- **Dataset Shape:** The dataset comprises 6060 rows and 8 columns, offering a substantial amount of data for analysis.
- **Null Values:** The dataset is pristine with no missing values, ensuring data integrity.
- **Data Types:** A mix of integer, float, and object data types enriches the dataset's diversity.

## Exploratory Data Analysis
- Informative columns such as text length, capital letters, word count, average word length, and special characters were explored to gather key insights.
- Detailed statistics provided critical information on data distribution and characteristics.

## Text Cleaning
- Implemented a comprehensive text cleaning function involving lowercasing, removal of special characters, stopwords, URLs, and user handles to enhance data quality.

## Modelling and Evaluation
- **Machine Learning Model:** Utilized a Random Forest Classifier for robust classification.
- **Cross-Validation Scores:** Averaged an impressive accuracy rate of around 88%, showcasing the model's capability.
- **Classification Reports:** Precise metrics including precision, recall, and F1-score were obtained for both training and testing data.
- **Best Model Hyperparameters:** Optimized model performance with randomized search for hyperparameters.
- **Evaluation Metrics:** Leveraged the Confusion Matrix, Test AUC Score, Test Log Loss Score, and Matthews Correlation Coefficient for a comprehensive evaluation.

## Feature Importance Analysis
- Identified critical features driving the classification model, emphasizing factors like text length and specific textual components.

## Advanced Analysis Report
### 1. **Project Goal**
Develop a sophisticated model to differentiate real from fake news using advanced machine learning techniques.

### 2. **Data Mastery**
Thoroughly managed a dataset of news articles, ensuring data completeness and integrity.

### 3. **Text Refinement**
Implemented advanced text cleaning techniques to standardize content for improved model performance.

### 4. **Model Marvel**
Utilized a Random Forest Classifier achieving an impressive average accuracy rate of approximately 88%.

### 5. **Performance Analysis**
In-depth evaluation using metrics like Confusion Matrix, Test AUC Score, Test Log Loss Score, and Matthews Correlation Coefficient provided profound insights into model effectiveness.

### 6. **Insightful Reports**
Detailed classification reports enhanced understanding of the model's behavior and performance.

### 7. **Hyperparameter Finesse**
Model optimization through randomized search improved predictive accuracy significantly.

### 8. **Feature Focus**
Key features influencing model predictions, such as word count and textual components, were highlighted.

**In conclusion, this project exemplifies an innovative approach using state-of-the-art machine learning to distinguish real and fake news articles, providing valuable insights and leading to advanced applications in news classification.**  


