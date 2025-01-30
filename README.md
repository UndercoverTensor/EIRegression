# Embedded Interpretable Regressor

A regression model for Machine Learning designed to perform **interpretable regression** without compromising prediction accuracy. It leverages a combination of an **interpretable classifier** (Dempster-Shafer using Gradient Descent classifier) and any compatible **regression model** (e.g., RandomForestRegressor, XGBRegressor, GradientBoostingRegressor) to predict continuous target variables.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Running Experiments](#running-experiments)
5. [Reproducing Results](#reproducing-results)
6. [Contributing](#contributing)
7. [License](#license)

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/EIRegression.git
   ```
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv env
   source env/bin/activate   # On Linux/Mac
   # OR
   env\Scripts\activate      # On Windows
   ```
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Below is a simple usage example with **Gradient Boosting**:

1. **Import** required modules and **prepare** data:
   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import r2_score, mean_absolute_error

   from EIRegressor.EmbeddedInterpreter import EmbeddedInterpreter
   from sklearn.ensemble import GradientBoostingRegressor

   data = pd.read_csv("data/insurance.csv")    # Your dataset
   target = "charges"

   # Data Preprocessing
   data = data.apply(pd.to_numeric, errors='ignore')
   data = pd.get_dummies(data, drop_first=True)

   # Split features & target
   X, y = data.drop(target, axis=1).values, data[target].values
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
   ```

2. **Define** the regressor and its parameters:
   ```python
   regressor = GradientBoostingRegressor
   reg_args = {"loss": "absolute_error", "n_estimators": 300}
   ```

3. **Create** and **fit** an Embedded Interpreter model:
   ```python
   eiReg = EmbeddedInterpreter(
       regressor=regressor,
       n_buckets=3,
       bucketing_method="quantile",
       reg_default_args=reg_args,    # Default arguments
       max_iter=4000,
       lossfn="MSE",
       min_dloss=0.0001,
       lr=0.005,
       precompute_rules=True
   )

   eiReg.fit(
       X_train, y_train,
       add_single_rules=True, single_rules_breaks=3,
       add_multi_rules=True,
       column_names=data.drop(target, axis=1).columns
   )
   ```

4. **Predict** and **evaluate**:
   ```python
   y_pred = eiReg.predict(X_test)
   print("R²:", r2_score(y_test, y_pred))
   print("MAE:", mean_absolute_error(y_test, y_pred))

   # Extract rules if needed
   rules_text = eiReg.get_rules_text()  # returns a string of all rules
   print(rules_text)
   ```

---

## Project Structure

A **high-level** overview of the major directories/files:

```
EIRegression/
├── EIRegressor/
│   ├── EmbeddedInterpreter.py  # Core Embedded Interpreter logic
│   ├── model_optimizer.py      # ModelOptimizer for hyperparameter tuning
│   └── ...                     # Additional utilities
├── experiments/
│   ├── datasets/
│   │   └── ...                 # Various datasets (bank32NH, housing, etc.)
│   ├── XGBRegression/
│   ├── RFRegression/
│   ├── rule_analysis_experiments/
│   ├── bank32NH.py
│   ├── concrete.py
│   ├── delta_elevators.py
│   ├── house_16H.py
│   ├── housing.py
│   ├── insurance.py
│   ├── movies.py
│   ├── ...
│   └── run_all_experiments.py  # Collects and runs multiple experiments
├── requirements.txt
├── README.md  # You are here
└── ...
```

**Key directories/files** include:

- **`EIRegressor/`**: Contains the **EmbeddedInterpreter** logic and auxiliary modules like **ModelOptimizer**.
- **`experiments/`**:
  - **`datasets/`**: Various datasets used for demonstration (e.g., `insurance.csv`, `housing.csv`, etc.).
  - **`bank32NH.py`, `concrete.py`, `delta_elevators.py`, etc.**: Scripts to run experiments on specific datasets.
  - **`run_all_experiments.py`** and others: Scripts that orchestrate multiple experiments sequentially.

---

## Running Experiments

Each script (e.g., `bank32NH.py`, `concrete.py`, `delta_elevators.py`) is designed with **dynamic path handling**, allowing you to run them from the **project root** without manual path adjustments.

### Running Individual Experiments

To run a specific dataset experiment:

```bash
cd EIRegression
python experiments/bank32NH.py
```

Replace `bank32NH.py` with the desired script name (e.g., `concrete.py`, `housing.py`, etc.).

### Running All Experiments

To execute all experiments sequentially:

```bash
python experiments/run_all_experiments.py
```

This script will generate results directories (e.g., `results_debugging/`, `results_fixed/`, etc.) under `experiments/` based on the specific configurations of each experiment.

---

## Reproducing Results

To **reproduce the experiment results**, follow these steps:

1. **Ensure all dependencies are installed**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Navigate to the project root**:
   ```bash
   cd EIRegression
   ```
3. **Run the desired experiments**:
   - **Individual** dataset experiment:
     ```bash
     python experiments/<script_of_choice>.py
     ```
     Replace `<script_of_choice>` with the specific experiment script (e.g., `bank32NH.py`, `concrete.py`, etc.).
   - **All** experiments at once (for example, using XGBoost or Random Forest):
     ```bash
     python experiments/run_all_XGB_experiments.py
     # or
     python experiments/run_all_experiments.py
     ```
4. **Locate results**:
   - Results and logs are saved in directories like `experiments/RFRegression/results/`, `experiments/results/<dataset_name>`, or `experiments/rule_analysis/results/`, depending on the script you run.
   - Each script prints the exact path to its `rules/` subfolder and `.json` summary files upon execution.

5. **Inspect the rule sets**:
   - Many experiments generate text files containing **interpretable rules** within a `rules/` subdirectory.
   - These files include lines like:
     ```
     # RULE 1: If feature_X <= 5.0 AND feature_Y > 2.0 THEN ...
     ```
   - Additionally, check the `.json` summary files for metrics such as R², MAE, MSE, etc.

---

## Contributing

We welcome **pull requests** and **issue reporting**! Please ensure that new or updated functionality:

- **Follows** existing style and code patterns.
- **Includes** docstrings and comments where necessary.
- **Passes** existing tests (if applicable).

Feel free to fork the repository and submit your enhancements!

---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to **use**, **modify**, and **distribute** this project in both commercial and non-commercial settings. See [LICENSE](LICENSE) for more details.

---

**Enjoy** interpretable regression with **EmbeddedInterpreter**! For any questions or feedback, please open an **issue** or **pull request**. We appreciate your contributions.
