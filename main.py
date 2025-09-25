import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_movies = 200
data = {
    'RottenTomatoes': np.random.randint(30, 100, num_movies),
    'IMDb': np.random.randint(40, 100, num_movies),
    'BoxOffice': np.random.randint(1000000, 100000000, num_movies) 
}
df = pd.DataFrame(data)
# Add some noise to the relationship to make it more realistic
df['BoxOffice'] += np.random.normal(0, 10000000, num_movies) #add gaussian noise
# --- 2. Data Cleaning and Preparation ---
#Check for missing values (although unlikely with synthetic data, good practice)
print("Missing values check:")
print(df.isnull().sum())
# --- 3. Analysis and Modeling ---
# Define features (X) and target (y)
X = df[['RottenTomatoes', 'IMDb']]
y = df['BoxOffice']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# --- 4. Visualization ---
# Scatter plot of Rotten Tomatoes vs. Box Office
plt.figure(figsize=(8, 6))
sns.scatterplot(x='RottenTomatoes', y='BoxOffice', data=df)
plt.title('Rotten Tomatoes vs. Box Office Revenue')
plt.xlabel('Rotten Tomatoes Score')
plt.ylabel('Box Office Revenue')
plt.savefig('rotten_tomatoes_vs_boxoffice.png')
print("Plot saved to rotten_tomatoes_vs_boxoffice.png")
# Scatter plot of IMDb vs. Box Office
plt.figure(figsize=(8, 6))
sns.scatterplot(x='IMDb', y='BoxOffice', data=df)
plt.title('IMDb Score vs. Box Office Revenue')
plt.xlabel('IMDb Score')
plt.ylabel('Box Office Revenue')
plt.savefig('imdb_vs_boxoffice.png')
print("Plot saved to imdb_vs_boxoffice.png")
#Plot Actual vs Predicted Box Office Revenue
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Box Office Revenue")
plt.ylabel("Predicted Box Office Revenue")
plt.title("Actual vs Predicted Box Office Revenue")
plt.savefig("actual_vs_predicted.png")
print("Plot saved to actual_vs_predicted.png")