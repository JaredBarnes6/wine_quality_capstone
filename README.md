# Wine Quality Analysis and Predictive Model
![image](https://user-images.githubusercontent.com/97368604/182192131-b4eb0442-c8ce-42b6-a996-23913be7b08d.png)
Have you ever wondered what goes into the finest quality wines? What separates the higher quality wines from the lower quality wines? Is it the sugar? The alcohol? The age? It is tremendously important for wine companies to know what goes into producing high quality wine and what buyers tend to buy. In this project, I create a predictive model to determine which features of wine greatest affect high wine quality, given a set of limited wine features. 

## Data:


The data set contains features of many different wines. Each row is a different individual wine. Each column is a different feature of the wine. The uncleaned data set contains 6497 rows (wines) and 13 columns (wine features).
List of features: wine type, pH, fixed acidity, volatile acidity, citric acid, residual sugars, chlorides, free sulfur dioxide present, total sulfur dioxide present, density, sulphates, and alcohol content, wine quality
Target Feature: wine quality as "best quality" feature (wines with qualities of 7 or greater on a 10 point scale)
Data Source: https://www.geeksforgeeks.org/wine-quality-prediction-machine-learning/


## Plan:


The goal of this project was to predict wine quality given the 13 features of wine in the data set. Further, I sought to determine which features of wine had the greatest influences on the wine quality. The following is a summary of the steps taken in this project. 
- Cleaned data and dropped rows with null values.
- Exploratory Data Analysis (EDA). 
- Separated “wine quality” feature into the binary "best quality" (1) and not best quality (0). Best quality wines were considered wines with quality levels of 7 and above. 
- Train, test split and scaled data. 
- Compare 3 classification model algorithms:
-- Logistic Regression
-- Random Forest Classifier
-- Gradient Boost
- Hyperparameter tuning using GridSearchCV.
- Identified the 3 most important features influencing wine quality. 



## Data Cleaning: 


All columns of the data set were of the expected data types. There were, however, numerous null values in 7 out of 13 columns. I calculated the percentage of the columns were null, and none were above 0.16 % null. Thus, I felt comfortable dropping all rows containing null values. The data were clean after dropping null values. 


## EDA:


Histograms of all features, with counts of each wine, showed the expected normal distributions. 
I checked some of the features of wine that one might expect to influence quality, such as alcohol content and pH (how acidic or basic the wine is). 
pH - No apparent trend between pH and wine quality. 
Alcohol content - Apparent positive trend between alcohol and wine quality. As alcohol increases so tends the wine quality. Note: wine quality is on a 10 point scale.

![Alt Image text](https://github.com/JaredBarnes6/wine_quality_capstone/blob/main/Charts%20and%20Images/alcohol%20vs%20quality.png)

Figure 1: Wine alcohol content is on the y axis and wine quality on the x. See the positive trend in alcohol and quality. As quality rises, overall, so does alcohol content. 


I calculated correlations between all features with one another. I found that "Total sulfur dioxide" had above a 0.7 Pearson correlation coefficient, making it highly correlated with at least one other feature.


## Feature and Target Manipulation:


 I removed the "Total sulfur dioxide" column, since it showed strong correlation between other columns, meaning that trends exhibited in this column will appear in the column(s) with which it is correlated. "Total sulfur dioxide" is no longer a needed column.  
A new column was created as the target variable: “best quality”. The wine “quality” feature was separated into "best quality" (1) and not best quality (0). Best quality wines were considered wines with quality levels of 7 and above. This ensured that I could identify wines by high and low qualities, relatively speaking, in a classification machine learning Model. 

![Alt Image text](https://github.com/JaredBarnes6/wine_quality_capstone/blob/main/Charts%20and%20Images/best%20quality%20column%20added%20to%20table%20.png)

Figure 2: A data table of all wine features. A new column was created, “best quality,” highlighted in yellow, indicating wines with qualities of 7 or greater (1 for quality >=7 wines and 0 for <7). 


“Quality” as a column was also dropped, since the new target variable is “best quality”. 



## Training/Testing Split, and Scaling:


Splitting: The data set was split into training and testing data - 80% training data, 20% testing data. The following features were used as the X variable in training/testing: wine type, alcohol, sulphates, pH, density, free sulfur dioxide, chlorides, residual sugar, citric acid, volatile acidity, and fixed acidity. The feature, “best quality”, was used as the y variable or target variable. 

Scaling: The X variable was scaled using sklearn’s MinMaxScaler() function. Both training and testing data were scaled for the X variable features. All numeric values were thus scaled between 0 and 1. 


## Model Algorithm Comparison:


I fit 3 models to the data to determine which works best at predicting the target variable using three different algorithms. The following 3 algorithms were applied and compared to one another:
- Random Forest Classifier
- Logistic Regression Classifier
- Gradient Boosting Classifier

To compare the three algorithm models, I calculated the following scores for each:
- Accuracy 
- Mean cross validation test score for test and train data
- Precision
- Recall
- F1
- Mean squared error
- Root mean squared error
Of the three algorithms, the model for the Random Forest algorithm performed the best (0.895 accuracy score, 0.965 cross validation test score), slightly better than the Gradient Boost model (0.858 accuracy score, 0.858 cross validation). Both Gradient Boost and Random Forest outperformed Logistic Regression (0.820 accuracy score, 0.813 cross validation).

![Alt Image text](https://github.com/JaredBarnes6/wine_quality_capstone/blob/main/Charts%20and%20Images/Algorithm%20comparison.png)

Figure 3: Two tables - the first table shows the algorithm and model accuracy, and the second table shows the algorithm and the 5-fold cross validation scores for training and testing data.
 
## Hyperparameter Tuning:


I used GridSearchCV to determine hyperparameters for the Random Forest model with 5-fold cross validation. This took a relatively long time to run, and the following parameters were determined to be optimal:  entropy model, estimators = 700, number of sample splits = 4, with a random state at 1.See Figure 4 below.



![Alt Image text](https://github.com/JaredBarnes6/wine_quality_capstone/blob/main/Charts%20and%20Images/Random%20Forest%20parameters%20tuned.png)

Figure 4: An image showing the results from GridSearchCV for the Random Forest Classifier model. These results are the optimal hyperparameters for the model, calculated using 5-fold cross validation.


## Model Effectiveness:


I determined the ROC-AUC score for the tuned model to determine its effectiveness (true positive rate and false positive rate curve, found in Figure 5 below). Its score was calculated to be 0.932. In Figure 5, we see that the curve trends closely to the upper left corner of the graph, representing its high true positive rate. The model is effective in producing many true positives. If the curve trended toward the 50/50 line shown in red, the model would not be effective and would need tuning or replacing. If the curve trended toward the false positive side of the graph, the model would simply be predicting the opposite answer than desired, and the model could be fixed relatively easily by changing the binary response to its opposite. 


![Alt Image text](https://github.com/JaredBarnes6/wine_quality_capstone/blob/main/Charts%20and%20Images/30.png)

Figure 5: The ROC curve for the Random Forest model is shown. The curve shown in blue represents the model’s predictions of the test data. The red line is the 50/50 line, representing where the curve would be if the model predicted randomly (an ineffective predicting model). We see that the curve trends toward the true positive rate, meaning that our model is effectively predicting the data most of the time. 



## Calculating Feature Weights:

I made predictions and identified features importances using the “feature_importances” function on the tuned Random Forest Classifier model. The following were the three most important features influencing best wine quality (see Figure 6 below):
    Wine Type
    Alcohol Content
    Sulphates

![Alt Image text](https://github.com/JaredBarnes6/wine_quality_capstone/blob/main/Charts%20and%20Images/Features%20and%20their%20weights%20to%20best%20wine%20quality%20target.png)

Figure 6: A horizontal bar chart of each feature’s importance to “best quality”. The features are sorted in descending order from most to least important. The importance calculation is relative to all features. 



## Recommendations and Going Further:     


If a wine company were to use this model, they could infer that those three aspects of their wines - wine type, alcohol content, and sulphates - are the most important to focus on in attaining the highest quality wines. Further, they could identify what wine types, alcohol contents, and sulphate levels truly corresponded to those wine quality scores of >=7 on a 10 point scale. 

Moving forward, I would determine the following in order to develop the project even more:
- Obtain more data: I would attempt to obtain more wine data from the same sources to correspond to each existing wine in the data set. Specifically, I would obtain data on the grapes used for each wine - amount of sugar in the grapes, grape type, grape age, etc. - to determine the effects on alcohol and quality. I would also obtain data on each wine’s price and sales, relative to other wines in the data set, in order to target optimal wine price. Finally, I would obtain each wine’s age and type of barrel/container used to store. With more wine features, this model could produce much stronger results.
- More algorithms: I would test more algorithms for modeling, such as Naive Bayes and XGBoost, to see if a better model could be attained.
- Specifics of important features: I would identify specific attributes/quantities of the 3 most important features that lead to the best wine quality, so as to provide a more detailed recommendation for wine companies looking to improve their overall qualities.



