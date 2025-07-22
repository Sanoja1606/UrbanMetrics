# UrbanMetrics
RentInsight-India is a Streamlit-powered web app to explore rental trends and predict house/apartment rents across major Indian cities. Using real-world data, EDA, and ML , it delivers actionable insights through interactive visualizations and user-friendly predictions.    


#Objective
The objective of this project is to develop a predictive model for estimating house rents based on various features such as location, furnishing status, number of bedrooms, and more. 

The project includes:
1.Performing Exploratory Data Analysis (EDA) to gain insights.
2.Preprocessing the data.
3.Selecting and tuning machine learning algorithms.
4.Deploying the model using Streamlit for user interaction.


#Dataset Glossary
BHK: Number of Bedrooms, Hall, Kitchen.
Rent: Rent of the property.
Size: Size of the property in square feet.
Floor: The floor on which the property is located.
Area Type: Size category (Super Area, Carpet Area, Build Area).
Area Locality: Locality where the property is located.
City: City of the property.
Furnishing Status: Whether the property is Furnished, Semi-Furnished, or Unfurnished.
Tenant Preferred: Type of tenant preferred (e.g., Family, Bachelor).
Bathroom: Number of Bathrooms.
Point of Contact: Whom to contact for more details.

#Learning Objectives
By working with the House Rental Dataset, you will learn:
Exploratory Data Analysis (EDA): Visualize distributions, relationships, and outliers in the dataset.
Data Preprocessing: Handle missing values, outliers, and encode categorical variables.
Model Selection & Tuning: Choose the best machine learning models for house rent prediction, tune their hyperparameters, and evaluate their performance.
Deployment with Streamlit: Deploy the trained model using Streamlit for real-time predictions.

#Accuracy Model
After training and fine-tuning the machine learning model on the House Rental Dataset, the best-performing model was evaluated on test data. The evaluation metrics show excellent performance, with an accuracy score of 98%.

#Accuracy Visualization:
![After training and tuning the machine learning model on the Home Rental Dataset, the performance of the best model is evaluated using accuracy as the evaluation metric. The model accuracy on the test set is 98%, which shows how well the model predicts the rental price based on the given features.](image.png)

#Model Accuracy Table:

Model	Accuracy Score	RMSE
XGBRegressor	0.984	2698
LGBMRegressor	0.986	2547

#Model Evaluation
The LGBMRegressor model achieved the highest performance with an accuracy score of 98% and an RMSE of 2547, indicating strong predictive power for house rental prices. The relatively high RMSE value suggests the model is generally accurate but may have occasional large errors.

#Model Deployment
The model has been deployed using Streamlit, enabling users to input features such as the number of bedrooms, city, and floor details, and receive a predicted rental price.

#Conclusion
This project demonstrates the application of machine learning techniques to predict house rental prices based on various features. The LGBMRegressor model performs exceptionally well with an accuracy of 98%, making it suitable for real-time use in the housing market.

Through this project, significant insights into rental pricing were gained, and the model was deployed via Streamlit for user access. 


