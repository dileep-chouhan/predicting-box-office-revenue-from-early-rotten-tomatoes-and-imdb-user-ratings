# Predicting Box Office Revenue from Early Rotten Tomatoes and IMDb User Ratings

**Overview:**

This project aims to develop a predictive model for estimating a film's opening weekend box office revenue based on early aggregated user ratings from Rotten Tomatoes and IMDb.  By leveraging these readily available data points, the model seeks to improve studio budgeting and marketing resource allocation strategies. The analysis involves data cleaning, exploratory data analysis (EDA), feature engineering, model selection, training, and evaluation.  The final model provides a prediction of opening weekend revenue given early user ratings.

**Technologies Used:**

* Python 3
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn


**How to Run:**

1. **Clone the repository:**  `git clone <repository_url>`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Run the main script:** `python main.py`


**Example Output:**

The script will print key statistical analysis to the console, including model performance metrics (e.g., R-squared, Mean Squared Error).  Additionally, the script generates several visualization plots (e.g., scatter plots showing the relationship between ratings and box office revenue, and potentially a residual plot for model diagnostics), which are saved as PNG files in the `output` directory.  These visualizations aid in understanding the model's performance and the relationships between the variables.  The exact filenames of the generated plots might vary depending on the analysis performed.


**Project Structure:**

* `data/`: Contains the input datasets.
* `src/`: Contains the source code for the project.
* `output/`: Contains the generated plots and any other output files.
* `requirements.txt`: Lists the project's dependencies.
* `main.py`: The main script to run the analysis.
* `README.md`: This file.


**Contributing:**

Contributions are welcome! Please feel free to open an issue or submit a pull request.


**License:**

[Specify your license here, e.g., MIT License]