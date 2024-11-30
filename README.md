# House Price Prediction Using Linear Regression

This project demonstrates the application of a **Linear Regression** model to predict house prices based on several features, such as **Square Feet**, **Bedrooms**, and **Bathrooms**. The dataset is simulated with basic house attributes and their corresponding prices. The goal of this project is to understand how different house features contribute to the price and evaluate the model's performance using various metrics.

## Overview

### Dataset
The dataset includes the following attributes:
- **Square_Feet**: Size of the house in square feet.
- **Bedrooms**: Number of bedrooms in the house.
- **Bathrooms**: Number of bathrooms in the house.
- **Price**: The price of the house in USD.

### Goal
Use **Linear Regression** to model the relationship between house attributes (features) and price (target variable).

## Steps Involved

### 1. Data Preparation:
- Created a dataset of house attributes and prices.
- Split the dataset into training and testing sets.

### 2. Model Training:
- Used the `LinearRegression` model from **scikit-learn** to train the model on the training dataset.

### 3. Prediction & Evaluation:
- Made predictions on the testing dataset.
- Evaluated the model using **Mean Squared Error (MSE)** and **R-squared (R²)** to assess the accuracy of the predictions.

### 4. Visualization:
- Visualized **feature importance** by plotting the model coefficients.
- Plotted **Actual vs Predicted** house prices to assess the model’s predictive accuracy.
- Analyzed prediction errors (residuals) using a **histogram** with a Kernel Density Estimate (KDE).
- Visualized feature correlations using a **correlation heatmap**.
- Used a **pairplot** to show relationships between multiple features and their impact on the target variable.

## Key Visualizations
- **Feature Importance**: A bar chart showcasing the importance of each feature based on the regression coefficients.
- **Actual vs Predicted Prices**: A scatter plot comparing the true house prices with the predicted prices, with a red line representing the perfect prediction.
- **Residual Analysis**: A histogram with a KDE plot to visualize the distribution of residuals (prediction errors).
- **Correlation Heatmap**: A heatmap to understand the correlations between features.
- **Pairplot**: A pairwise plot of features to visualize their relationships.

## Requirements

To run this project locally, you will need the following libraries:

- **NumPy**: `pip install numpy`
- **Pandas**: `pip install pandas`
- **Scikit-learn**: `pip install scikit-learn`
- **Matplotlib**: `pip install matplotlib`
- **Seaborn**: `pip install seaborn`

## Conclusion

This project provides a simple yet effective demonstration of using **Linear Regression** to predict house prices based on various features. The visualizations and evaluation metrics give us insight into the performance of the model and the relationships between the house attributes and their prices. This project can be extended by using more complex models or additional features to improve predictions.
