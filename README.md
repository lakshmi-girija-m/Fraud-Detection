# Fraud Detection for e-commerce transactions

### About the Project: <h3>
This project is about predicting whether an e-commerce transaction is a fraud transaction or not. The dataset can be found [here](https://www.kaggle.com/vbinh002/fraud-ecommerce). There are two datasets, where one dataset contains user information like signup time, purchase time, age sex, purchase value, IP address etc. Other dataset contains the IP address range and the associated country. By mapping the IP address of given transactions with the IP address ranges in second dataset, we can get the country name. 
  
### Data Analysis: <h3>
The features signup time and purchase time were highly correlated with the target variable. A new feature 'time_difference' was created which is the difference of time between sign up time and purchase time. As Device ID and purchase value had low correlation with target variable, they were removed. There were no duplicate records as well. As the data was imbalanced, it was handled using SMOTE. Label encoding and One hot encoding was performed on the dataset. For both label and one hot encoded data, PCA with 6 features and LDA was performed. 
  
### Machine Learning: <h3>
Since this is a classification problem, various classification algorithms were used with some hyperparameter tuning. 
| Algorithm      | Parameters     | F1 score  |
| ------------- |:-------------:| -----:|
| Logistic Regression  | 'class_weight': ["balanced"] | 0.2706 |
| Decision Tree      | 'max_depth': [10, 40, 50, 100, None]   | 0.5415 |
| Naive Bayes | ---      | 0.2789|
| Random Forest | 'n_estimators': [50, 100, 150], 'max_depth': [None], 'n_jobs': [-1]     | 0.6668 |
| KNN | 'n_neighbors': [2, 3, 4]   | 0.4908 |
| XGBoost | 'n_estimators':[50, 100, 150], 'n_jobs': [-1]    | 0.6951 |
  
Out of them, XGBoost performed well on one hot encoded data and transformed using PCA with F1 score of 69% for parameters n_estimators=150 and n_jobs=-1. Genetic algorithms was also used for furthur optimization. 

### Deployment: <h3>
The application was deployed on Heroku. Batch prediction was also implemented. The user can input a CSV file with required features. A CSV file along with class for each row can be dowloaded. A user can also opt to input required features using a form. When th user clicks on submit button, it will display if the transaction is a fraud transaction or not a fraud transaction.
Link to Deployed Application : https://detecting-fraud.herokuapp.com/

<video controls>
  <source src="Images/Fraud Detection.mp4" type="video/mp4">
</video>

### Installing required librarires: <h3>
* Installing xgboost:
```
pip install xgboost
```
