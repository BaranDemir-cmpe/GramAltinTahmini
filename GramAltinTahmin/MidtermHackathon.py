import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Baran Çakmak Demir 210315002

# Load the data
data = pd.read_excel('C:\Users\baran\OneDrive\Masaüstü\GramAltin_5yillikVeri_241106.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)


#Task 1
def linear_regression_prediction(data, period_days):
    periods = range(0, len(data), period_days)
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data['Price'], label="Actual Price", color="blue")

    for start in periods:
        end = min(start + period_days, len(data))
        x = np.arange(0, end - start)
        y = data['Price'].iloc[start:end]

        # Linear regression (y = ax + b)
        a, b = np.polyfit(x, y, 1)
        y_pred = a * x + b

        plt.plot(data.index[start:end], y_pred, linestyle="--", label=f"Linear Regression {start // period_days + 1}")

    plt.title(f"Linear Regression Prediction Model (Period: {period_days} days)")
    plt.xlabel("Date")
    plt.ylabel("Gram Altin Price")
    plt.legend()
    plt.show()


#Task 2
def polynomial_regression_prediction(data, period_days, degree):
    periods = range(0, len(data), period_days)
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data['Price'], label="Actual Price", color="blue")

    for start in periods:
        end = min(start + period_days, len(data))
        x = np.arange(0, end - start)
        y = data['Price'].iloc[start:end]

        # Polynomial regression (degree n)
        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x)

        plt.plot(data.index[start:end], y_pred, linestyle="--",
                 label=f"Poly Regression (Degree {degree}) {start // period_days + 1}")

    plt.title(f"Polynomial Regression Prediction Model (Period: {period_days} days, Degree: {degree})")
    plt.xlabel("Date")
    plt.ylabel("Gram Altin Price")
    plt.legend()
    plt.show()


#Task 3
def test_model(data, test_data, period_days, degree=None):
    predictions = []
    periods = range(0, len(data), period_days)

    for start in periods:
        end = min(start + period_days, len(data))
        x = np.arange(0, end - start)
        y = data['Price'].iloc[start:end]

        if degree is None:
            # Linear regression
            a, b = np.polyfit(x, y, 1)
            y_pred = a * np.arange(len(test_data))
        else:
            # Polynomial regression
            coeffs = np.polyfit(x, y, degree)
            y_pred = np.polyval(coeffs, np.arange(len(test_data)))

        predictions.extend(y_pred)

    return predictions[:len(test_data)]


# Usage Examples:
# Task 1 - Linear Regression with 30 days period
linear_regression_prediction(data, 30)

# Task 2 - Polynomial Regression with 30 days period and degree 2
polynomial_regression_prediction(data, 30, 2)

# Task 3 - Testing the model with 30 days period for both Linear and Polynomial (e.g., degree 2)
test_data = pd.read_excel('C:\Users\baran\OneDrive\Masaüstü\GramAltin_5yillikVeri_241106.xlsx', sheet_name="Test")
test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data.set_index('Date', inplace=True)

# Linear model prediction
linear_predictions = test_model(data, test_data, 30)

# Polynomial model prediction (degree 2)
polynomial_predictions = test_model(data, test_data, 30, degree=2)

# Add predictions to the test data for analysis
test_data['Linear_Predictions'] = linear_predictions
test_data['Polynomial_Predictions'] = polynomial_predictions
print(test_data[['Price', 'Linear_Predictions', 'Polynomial_Predictions']])
