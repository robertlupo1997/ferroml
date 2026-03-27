# Cross-Library Benchmark Results

Generated: 2026-03-27T14:07:08.673102
Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
Python: 3.12.3
NumPy: 2.4.2
sklearn: 1.8.0
xgboost: 3.2.0
lightgbm: 4.6.0

## Summary Table

| Algorithm | Library | Task | N | Features | Fit (ms) | Predict (ms) | Score |
|-----------|---------|------|---|----------|----------|--------------|-------|
| LinearRegression | ferroml | regression | 10000 | 20 | 4.2 | 0.1 | 0.9993 |
| LinearRegression | sklearn | regression | 10000 | 20 | 2.2 | 0.1 | 0.9993 |
| Ridge | ferroml | regression | 10000 | 20 | 1.7 | 0.1 | 0.9993 |
| Ridge | sklearn | regression | 10000 | 20 | 1.0 | 0.1 | 0.9993 |
| Lasso | ferroml | regression | 10000 | 20 | 0.8 | 0.1 | 0.3889 |
| Lasso | sklearn | regression | 10000 | 20 | 0.7 | 0.1 | 0.3889 |
| LogisticRegression | ferroml | classification | 10000 | 20 | 16.8 | 0.1 | 0.8175 |
| LogisticRegression | sklearn | classification | 10000 | 20 | 8.0 | 0.1 | 0.8175 |
| DecisionTreeRegressor | ferroml | regression | 10000 | 20 | 77.5 | 0.2 | 0.5021 |
| DecisionTreeRegressor | sklearn | regression | 10000 | 20 | 88.5 | 0.2 | 0.5020 |
| DecisionTreeClassifier | ferroml | classification | 10000 | 20 | 85.1 | 0.2 | 0.8660 |
| DecisionTreeClassifier | sklearn | classification | 10000 | 20 | 109.9 | 0.2 | 0.8580 |
| RandomForestRegressor | ferroml | regression | 10000 | 20 | 274.5 | 2.6 | 0.4494 |
| RandomForestRegressor | sklearn | regression | 10000 | 20 | 5455.7 | 18.6 | 0.7863 |
| RandomForestRegressor | xgboost | regression | 10000 | 20 | 1913.4 | 1.9 | 0.7785 |
| RandomForestRegressor | lightgbm | regression | 10000 | 20 | 89.9 | 1.6 | 0.5537 |
| RandomForestClassifier | ferroml | classification | 10000 | 20 | 298.4 | 2.4 | 0.8910 |
| RandomForestClassifier | sklearn | classification | 10000 | 20 | 1463.1 | 15.5 | 0.9300 |
| RandomForestClassifier | xgboost | classification | 10000 | 20 | 567.2 | 0.9 | 0.9210 |
| RandomForestClassifier | lightgbm | classification | 10000 | 20 | 84.0 | 1.7 | 0.8650 |
| GradientBoostingRegressor | ferroml | regression | 10000 | 20 | 4982.6 | 1.4 | 0.9298 |
| GradientBoostingRegressor | sklearn | regression | 10000 | 20 | 5612.3 | 4.5 | 0.9296 |
| GradientBoostingRegressor | xgboost | regression | 10000 | 20 | 81.6 | 0.5 | 0.9323 |
| GradientBoostingRegressor | lightgbm | regression | 10000 | 20 | 46.8 | 1.8 | 0.9336 |
| GradientBoostingClassifier | ferroml | classification | 10000 | 20 | 4802.8 | 1.7 | 0.9060 |
| GradientBoostingClassifier | sklearn | classification | 10000 | 20 | 5864.2 | 4.0 | 0.9285 |
| GradientBoostingClassifier | xgboost | classification | 10000 | 20 | 76.1 | 0.6 | 0.9335 |
| GradientBoostingClassifier | lightgbm | classification | 10000 | 20 | 48.6 | 2.1 | 0.9390 |
| HistGradientBoostingRegressor | ferroml | regression | 10000 | 20 | 299.3 | 4.2 | 0.9301 |
| HistGradientBoostingRegressor | sklearn | regression | 10000 | 20 | 138.6 | 2.0 | 0.9313 |
| HistGradientBoostingRegressor | xgboost | regression | 10000 | 20 | 88.4 | 0.6 | 0.9323 |
| HistGradientBoostingRegressor | lightgbm | regression | 10000 | 20 | 47.9 | 1.8 | 0.9336 |
| HistGradientBoostingClassifier | ferroml | classification | 10000 | 20 | 271.5 | 4.1 | 0.9310 |
| HistGradientBoostingClassifier | sklearn | classification | 10000 | 20 | 137.2 | 2.0 | 0.9355 |
| HistGradientBoostingClassifier | xgboost | classification | 10000 | 20 | 76.1 | 0.7 | 0.9335 |
| HistGradientBoostingClassifier | lightgbm | classification | 10000 | 20 | 50.8 | 2.0 | 0.9390 |
| KNeighborsClassifier | ferroml | classification | 10000 | 20 | 2.0 | 27.5 | 0.9425 |
| KNeighborsClassifier | sklearn | classification | 10000 | 20 | 0.5 | 12.9 | 0.9425 |
| SVC | ferroml | classification | 5000 | 20 | 143.6 | 19.5 | 0.9640 |
| SVC | sklearn | classification | 5000 | 20 | 93.1 | 33.1 | 0.9560 |
| GaussianNB | ferroml | classification | 10000 | 20 | 0.6 | 0.2 | 0.7995 |
| GaussianNB | sklearn | classification | 10000 | 20 | 1.7 | 0.4 | 0.7995 |
| StandardScaler | ferroml | preprocessing | 10000 | 20 | 0.3 | 0.1 | N/A |
| StandardScaler | sklearn | preprocessing | 10000 | 20 | 1.0 | 0.1 | N/A |
| PCA | ferroml | preprocessing | 10000 | 20 | 0.6 | 0.1 | N/A |
| PCA | sklearn | preprocessing | 10000 | 20 | 0.6 | 0.1 | N/A |
| KMeans | ferroml | clustering | 10000 | 20 | 4.6 | 0.1 | 159775.1787 |
| KMeans | sklearn | clustering | 10000 | 20 | 17.8 | 0.4 | 159775.1787 |

## FerroML vs Others: Fit-Time Speedup

Values > 1.0 mean FerroML is faster.

| Algorithm | N | vs sklearn | vs xgboost | vs lightgbm |
|-----------|---|-----------|-----------|------------|
| LinearRegression | 10000 | 0.52x | N/A | N/A |
| Ridge | 10000 | 0.59x | N/A | N/A |
| Lasso | 10000 | 0.94x | N/A | N/A |
| LogisticRegression | 10000 | 0.48x | N/A | N/A |
| DecisionTreeRegressor | 10000 | 1.14x | N/A | N/A |
| DecisionTreeClassifier | 10000 | 1.29x | N/A | N/A |
| RandomForestRegressor | 10000 | 19.87x | 6.97x | 0.33x |
| RandomForestClassifier | 10000 | 4.90x | 1.90x | 0.28x |
| GradientBoostingRegressor | 10000 | 1.13x | 0.02x | 0.01x |
| GradientBoostingClassifier | 10000 | 1.22x | 0.02x | 0.01x |
| HistGradientBoostingRegressor | 10000 | 0.46x | 0.30x | 0.16x |
| HistGradientBoostingClassifier | 10000 | 0.51x | 0.28x | 0.19x |
| KNeighborsClassifier | 10000 | 0.24x | N/A | N/A |
| SVC | 5000 | 0.65x | N/A | N/A |
| GaussianNB | 10000 | 2.72x | N/A | N/A |
| StandardScaler | 10000 | 3.44x | N/A | N/A |
| PCA | 10000 | 1.01x | N/A | N/A |
| KMeans | 10000 | 3.90x | N/A | N/A |
