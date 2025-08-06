#  Superstore Sales Prediction with Multiple ML Models

##  Project Overview

This project uses machine learning algorithms to predict **Profit** based on sales data from a retail superstore. We applied several regression models and compared their performance to find the best one.

Dataset used: [Superstore Dataset (Kaggle)](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

---

##  Data Preprocessing

- **Scaling:** Applied `StandardScaler()` to normalize the features.
- **Train-Test Split:**
  - `test_size = 0.2`
  - `random_state = 42` (for reproducibility)

```python
scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2, random_state=42)
```

---

##  Models Used

We evaluated the following regression models:

| Model                     | Library                |
| ------------------------- | ---------------------- |
| RandomForestRegressor     | `sklearn.ensemble`     |
| GradientBoostingRegressor | `sklearn.ensemble`     |
| LinearRegression          | `sklearn.linear_model` |
| DecisionTreeRegressor     | `sklearn.tree`         |
| KNeighborsRegressor       | `sklearn.neighbors`    |
| XGBRegressor              | `xgboost`              |

---

##  Model Training & Evaluation

```python
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

modeles = {
    "RandomForestRegressor": RandomForestRegressor(),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "XGBRegressor": XGBRegressor()
}

for name, model in modeles.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    score = r2_score(y_train, y_pred)
    print(f"{name} accuracy score is {score:.2f}")
```

---best_model=DecisionTreeRegressor()
best_model.fit(x_train,y_train)
y_pre=best_model.predict(x_test)
score1=r2_score(y_test , y_pre)
print(f"DecisionTreeRegressor accuracy score is {score1}")

##  Results

The models were trained and evaluated using the **R² score** on the training set. The one with the highest score is considered the best fit (but cross-validation is recommended for better generalization).

---



---

##  Dependencies

```bash
pip install scikit-learn xgboost pandas matplotlib seaborn
```

---

##  Conclusion

This project demonstrates how to:

- Preprocess and scale data
- Apply and compare multiple regression models
- Use XGBoost for advanced performance
- Prepare a foundation for predictive sales analytics

 Ideal for business intelligence, sales forecasting, and decision support systems.

---

> Built with ❤️ by Ahmed Hamdy

