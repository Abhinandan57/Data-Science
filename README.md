# Feature Engineering
- Feature Engineering is the process of using domain knowledge of the data to create features or variables that make machine learning algorithms work.
- Feature Engineering includes :
1. filling missing values within a variable
2. encoding categorical variables into numbers
3. variable transformation
4. creating or extracting new features from the available variables in the dataset

__Missing Value Treatment__
- There are 3 main mechanisms that lead to missing data. Understanding the missing data mechanisms may help us choose the right missing data imputation technique.
1. Missing data Completely At Random (MCAR)  
2. Missing data At Random (MAR)  
3. Missing data Not At Random (MNAR)

__Missing Completely at Random (MCAR) :__  A variable is missing completely at random (MCAR) if the probability of being missing is the same for all the observations. When data is MCAR, there is absolutely no relationship between the data missing and any other values, observed or missing, within the dataset. In other words, those missing data points are a random subset of the data. There is nothing systematic going on, that makes some data more likely to be missing than other.

__Missing Data Not At Random (MNAR) (Systematic missing values) :__ When data is MNAR, there is absolutely some relationship between the data missing and any other values, observed or missing, within the dataset.

## TECHNIQUES TO HANDLE MISSING VALUES OF CONTINUOUS FEATURES:
__1. Mean/Median/Mode imputation__
When to apply this technique? Mean/Median imputation has the assumption that the data are missing completely at random (MCAR). We solve this by replacing the NaN with the most frequent occurance of the variable.

__Advantages of Mean/Median Imputaion:__  
- Easy to implement (Robust to outliers)
- Faster way to obtain the complete dataset

__Disadvantages of Mean/Median Imputation:__  
- Change or distortion in the original variance
- Impacts Correlation

__2. Random Sample imputation__
- This technique consists of taking random observation from the dataset and we use this observation to replace the NaN values.

When to use this technique? It assumes that the data are missing completely at random (MCAR). So in cases of MCAR, we will be using this technique.

__Advantages of Random Sample Imputation :__  
- Easy to implement
- There is no distortion in variance

__Disadvantage of Random Sample Imputation :__  
- In every situation, randomness will not work

__3. Capturing NaN values with a new feature__
When to use this technique? It works well if the data are not missing completely at random (MNAR).

__Advantages of Capturing NaN values with a new feature :__  
- Easy to implement
- Captures the importance of missing values

__Disadvantage of Capturing NaN values with a new feature :__  
- Creates additional features, which leads to __Curse of Dimensionality__

__4. End of Distribution imputation__
- If there is suspicion that the missing value is not at random then capturing that information is important. In this scenario, one would want to replace missing data with values that are at the tails of the distribution of the variable.

__Advantages of End of Distribution imputation :__  
- Easy to implement
- Captures the importance of missing data

__Disadvantages of End of Distribution imputation :__  
- Distorts the original distribution of the variable
- If missingness is not important, it may mask the predictive power of the original variable by distorting its distribution
- If the number of NA is big, it will mask true outliers in the distribution
- If the number of NA is small, the replaced NA may be considered as an outlier and pre-processed in subsequent feature engineering.

__5. Arbitrary value imputation__
- It consists of replacing NaN by an arbitrary value. This technique was derived from kaggle competition.
- Arbitrary value means the value should not be more frequently present.
- Last outliers are taken in this technique to fill the NaN. It may be least outlier or max outlier.

__Advantages of Arbitrary Value imputation :__  
- Easy to implement
- Captures the importance of missingness if there is one

__Disadvantages of Arbitrary Value imputation :__  
- Distorts the original distribution of the variable
- If missingness is not important, it may mask the predictive power of the original variable by distorting its distribution
- Hard to decide which value to use

## TECHNIQUES TO HANDLE MISSING VALUES OF CATEGORICAL FEATURES:
__1. Frequent category imputation__
__Advantages of Frequent Category imputation :__  
- Easy to implement
- Faster way to implement

__Disadvantages of Frequent Category imputation :__  
- Since we are using the most frequent labels, it may use them in an over represented way, if there are many NaN values.
- It distorts the relationship of the most frequent label

__2. Adding a variable to capture NaN__
- Whenever you have a situation where you have lot of missing values, try to use this particular technique w.r.t one feature.

## TECHNIQUES TO HANDLE MISSING VALUES OF CATEGORICAL FEATURES
__1. One Hot Encoding__
__2. Ordinal Number Encoding__
__3. Count or Frequency Encoding__
__4. Target Guided Ordinal Encoding__
__5. Mean Encoding__
__6. Probability Ratio Encoding__

# Feature Transformation
## Why transformation of features are required?
- Every point has some vector(magnitude & direction). If there is a huge difference between the vectors, then probably the calculation like finding of the global minima by gradient descent algorithm in linear regression, finding of the eucledian distance by the algorithms like KNN, KMeans, Hierarchical clustering will take more time. Hence we need to scale down this particular value, for which transformation of features is done.
- Transformation or scaling or standardization is not required for each & every machine learning algorithm. Some of the machine learning algorithms like linear regression, logistic regression, KNN, KMeans, Hierarchical Means where either Gradient Descent or Eucledian Distance concepts are used, you need to perform transformation.
- __We do not require transformation in ensemble techniques like Decision Tree, Random forest, xgboost, Adaboost.__
- In Deep learning ANN, CNN, RNN we require standardization, sacling.
- By scaling the data, the accuracy and performance increases.

## Transformation techniques:
__1. Normalization and Standardization__
### Standardization :
- Here we try to bring all the variables or features to a similar scale. We will transform all the variables considering the mean value as 0 and standard deviation as 1.
- z score= (x - x_mean)/std
- __StandardScaler helps you to find out the centre medium value which is just like a Gaussian ditribution. But if you have outliers, at that time it may get affected.__
- __If your data w.r.t standard normal distribution then use StandardScaler.__ 

__2. MinMaxScaler__
- It scales the values between 0 and 1.
- x_scaled = (x - x.min)/(x.max - x.min)
- It works well with Deep Learning Techniques i.e. CNN. People also use it in ML.
- __If your data is not following standard normal distribution, then use MinMaxScaler.__

__3. RobustScaler__
- It is used to scale the features to median and quantiles.
- Scaling using median and quantiles consists of subtracting the median to all the observations and then dividing by the interquantile range. The interquantile range is the difference between the 75th and 25th quantile.
- IQR = 75th quantile - 25th quantile
- x_scaled = (x - x.median) / IQR
- This technique is more robust to outliers.

__4. Guassian Transformation__
- Suppose your features are not normally distributed, then we can apply some mathematical equation to convert it into a normally distributed or gaussian distributed. And for that we use Gaussian Transformation.
- Why normal distribution is required? Becuase some of the ML algorithms like linear regression, logistic regression works pretty much well when your data is Gaussian distributed. This results in good accuracy and performance.
- When the data is not normally distributed, then we can apply below techniques to convert it into normally distributed:
1. Logarithmic Transformation
2. Reciprocal Transformation
3. Square Root Transformation
4. Exponential Transformation
5. BoxCox Transformation

# Outlier Treatment Techniques
1. Using 3 Standard Deviation
2. Using z-score
3. Using percentile
4. Using IQR
- IQR = Q3 - Q1  
where Q3 = 75th Percentile, Q1 = 25th percentile
- To remove outlier, we should have the lower & upper limit.  
- lower_limit = Q1 - 1.5*IQR  
- upper_limit = Q3 + 1.5*IQR

### Which Machine Learning Models are sensitive to outliers?
__1. Naive Bayes Classifier :__ Not sensitive to outliers  
__2. SVM :__ Not sensitive to outliers  
__3. Linear Regression :__ Sensitive to outliers  
__4. Logistic Regression :__ Sensitive to outliers  
__5. Decision Tree Regressor or Classifier :__ Not sensitive to outliers  
__6. Ensemble (Random Forest, XGBoost, Gradient Boosting) :__ Not sensitive to outliers  
__7. KNN :__ Not sensitive to outliers  
__8. KMeans :__ Sensitive to outliers  
__9. Hierarchical :__ Sensitive to outliers  
__10. PCA :__ Sensitive to outliers  
__11. Latent Dirichlet Allocation (LDA) :__ Sensitive to outliers  
__12. DBScan :__ Sensitive to outliers  
__13. Neural Networks :__ Sensitive to outliers
