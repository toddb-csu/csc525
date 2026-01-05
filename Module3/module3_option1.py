# Todd Bartoszkiewicz
# CSC525: Introduction to Machine Learning
# Module 3: Critical Thinking Option #1
#
# Option #1: Simple Linear Regression in Scikit Learn
# Linear regression is a supervised learning algorithm that predicts a dependent variable value based on an independent
# variable by fitting a linear equation to the data.
#
# There are several advantages to linear regression, mainly high efficiency. This efficiency can easily lead to
# overfitting the data, however.
#
# For your assignment, you will build a linear regression model in Python.
#
# The Boston housing dataset can be loaded in scikit-learn using the command load_boston ( ) after from sklearn.datasets
# import load_boston.
#
# Using this data, our model should be able to predict the value of a house using the features given in the dataset.
#
# The Boston housing dataset is a common target for regression analysis, feel free to use Google to conduct your own
# research for this assignment. Feel free to also use the following documentation and resources:
#
# Linear regression https://scipy-cookbook.readthedocs.io/items/LinearRegression.html.
# Linear Regression on Boston Housing Dataset https://github.com/animesh-agarwal/Machine-Learning-Datasets/blob/master/boston-housing/Linear_Regression.ipynb.
# Submission should include an executable Python file demonstrating the prediction of housing prices.
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    print("Loading Boston dataset.")
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    # The target is median value in $1000s
    y = pd.Series(boston.target, name='MEDV')
    print("Success! Boston dataset loaded.")
    print(f"Dataset shape: {X.shape}")
    print(X.head())
    print(f"Target name: {y.name}")
    print(y.head())
    print(f"Features: {list(X.columns)}")
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Make predictions on test set
    y_pred = model.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("=== Linear Regression Results ===")
    print(f"Mean Squared Error (MSE)   : {mse:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"R^2 Score                   : {r2:.4f}")
    print()

    # Show coefficients
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_.round(4)
    }).sort_values('Coefficient', ascending=False)

    print("Model coefficients:")
    print(coef_df.to_string(index=False))
    print(f"\nIntercept (constant term): {model.intercept_:.3f}")
    print()

    # Example prediction
    example_idx = 6
    sample_house = X_test.iloc[[example_idx]]
    true_price = y_test.iloc[example_idx]
    predicted_price = y_pred[example_idx]

    print("Example prediction:")
    print(f"Input features:\n{sample_house.to_string()}")
    print(f"True price   : ${true_price:,.0f}")
    print(f"Predicted price: ${predicted_price:,.0f}")

    # Visualize predicted vs actual prices
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='teal', label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction line')
    plt.xlabel('Actual Median House Value ($1000s)')
    plt.ylabel('Predicted Median House Value ($1000s)')
    plt.title('Linear Regression – Boston Housing Price Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('boston_linear_regression.png')
    plt.show()
