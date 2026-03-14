# Cross-Library Benchmark Results

Generated: 2026-03-14T14:20:33.242921
Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
Python: 3.12.3
NumPy: 2.4.2
sklearn: 1.8.0
xgboost: 3.2.0
lightgbm: 4.6.0

## Summary Table

| Algorithm | Library | Task | N | Features | Fit (ms) | Predict (ms) | Score |
|-----------|---------|------|---|----------|----------|--------------|-------|
| LinearRegression | ferroml | regression | 1000 | 20 | 0.9 | 0.0 | 0.9996 |
| LinearRegression | sklearn | regression | 1000 | 20 | 0.6 | 0.1 | 0.9996 |
| Ridge | ferroml | regression | 1000 | 20 | 0.2 | 0.0 | 0.9996 |
| Ridge | sklearn | regression | 1000 | 20 | 0.5 | 0.1 | 0.9996 |
| Lasso | ferroml | regression | 1000 | 20 | 0.1 | 0.0 | 0.4862 |
| Lasso | sklearn | regression | 1000 | 20 | 0.4 | 0.1 | 0.4862 |
| LogisticRegression | ferroml | classification | 1000 | 20 | 0.7 | 0.0 | 0.8300 |
| LogisticRegression | sklearn | classification | 1000 | 20 | 3.2 | 0.1 | 0.8350 |
| DecisionTreeRegressor | ferroml | regression | 1000 | 20 | 7.1 | 0.0 | 0.0246 |
| DecisionTreeRegressor | sklearn | regression | 1000 | 20 | 8.9 | 0.1 | -0.0511 |
| DecisionTreeClassifier | ferroml | classification | 1000 | 20 | 5.3 | 0.0 | 0.8550 |
| DecisionTreeClassifier | sklearn | classification | 1000 | 20 | 8.4 | 0.1 | 0.8400 |
| RandomForestRegressor | ferroml | regression | 1000 | 20 | 24.0 | 1.0 | 0.3928 |
| RandomForestRegressor | sklearn | regression | 1000 | 20 | 485.6 | 7.0 | 0.6434 |
| RandomForestRegressor | xgboost | regression | 1000 | 20 | 1260.7 | 0.9 | 0.6540 |
| RandomForestRegressor | lightgbm | regression | 1000 | 20 | 32.9 | 0.7 | 0.4637 |
| RandomForestClassifier | ferroml | classification | 1000 | 20 | 20.8 | 0.6 | 0.9250 |
| RandomForestClassifier | sklearn | classification | 1000 | 20 | 173.9 | 5.7 | 0.9500 |
| RandomForestClassifier | xgboost | classification | 1000 | 20 | 157.4 | 0.6 | 0.9200 |
| RandomForestClassifier | lightgbm | classification | 1000 | 20 | 36.0 | 0.8 | 0.8500 |
| GradientBoostingRegressor | ferroml | regression | 1000 | 20 | 353.5 | 1.0 | 0.7690 |
| GradientBoostingRegressor | sklearn | regression | 1000 | 20 | 490.7 | 0.7 | 0.7686 |
| GradientBoostingRegressor | xgboost | regression | 1000 | 20 | 80.8 | 0.4 | 0.7816 |
| GradientBoostingRegressor | lightgbm | regression | 1000 | 20 | 24.4 | 0.6 | 0.8176 |
| GradientBoostingClassifier | ferroml | classification | 1000 | 20 | 337.5 | 0.8 | 0.8850 |
| GradientBoostingClassifier | sklearn | classification | 1000 | 20 | 548.3 | 0.7 | 0.9400 |
| GradientBoostingClassifier | xgboost | classification | 1000 | 20 | 53.5 | 0.5 | 0.9450 |
| GradientBoostingClassifier | lightgbm | classification | 1000 | 20 | 21.5 | 0.9 | 0.9500 |
| HistGradientBoostingRegressor | ferroml | regression | 1000 | 20 | 1167.9 | 0.6 | 0.8203 |
| HistGradientBoostingRegressor | sklearn | regression | 1000 | 20 | 100.4 | 1.1 | 0.8095 |
| HistGradientBoostingRegressor | xgboost | regression | 1000 | 20 | 781.9 | 0.9 | 0.7816 |
| HistGradientBoostingRegressor | lightgbm | regression | 1000 | 20 | 23.4 | 0.6 | 0.8176 |
| HistGradientBoostingClassifier | ferroml | classification | 1000 | 20 | 1206.2 | 0.7 | 0.9350 |
| HistGradientBoostingClassifier | sklearn | classification | 1000 | 20 | 94.3 | 0.8 | 0.9400 |
| HistGradientBoostingClassifier | xgboost | classification | 1000 | 20 | 917.8 | 1.1 | 0.9450 |
| HistGradientBoostingClassifier | lightgbm | classification | 1000 | 20 | 23.8 | 0.7 | 0.9500 |
| KNeighborsClassifier | ferroml | classification | 1000 | 20 | 0.1 | 1.5 | 0.9050 |
| KNeighborsClassifier | sklearn | classification | 1000 | 20 | 0.3 | 1.4 | 0.9050 |
| SVC | ferroml | classification | 1000 | 20 | 14.2 | 2.1 | 0.9550 |
| SVC | sklearn | classification | 1000 | 20 | 6.8 | 2.2 | 0.9450 |
| GaussianNB | ferroml | classification | 1000 | 20 | 0.1 | 0.0 | 0.8300 |
| GaussianNB | sklearn | classification | 1000 | 20 | 0.7 | 0.1 | 0.8300 |
| StandardScaler | ferroml | preprocessing | 1000 | 20 | 0.0 | 0.0 | N/A |
| StandardScaler | sklearn | preprocessing | 1000 | 20 | 0.4 | 0.1 | N/A |
| PCA | ferroml | preprocessing | 1000 | 20 | 0.4 | 0.0 | N/A |
| PCA | sklearn | preprocessing | 1000 | 20 | 0.6 | 0.1 | N/A |
| KMeans | ferroml | clustering | 1000 | 20 | 29.3 | 0.0 | 16074.9071 |
| KMeans | sklearn | clustering | 1000 | 20 | 11.9 | 0.1 | 16074.9071 |
| LinearRegression | ferroml | regression | 5000 | 20 | 1.8 | 0.0 | 0.9997 |
| LinearRegression | sklearn | regression | 5000 | 20 | 1.5 | 0.1 | 0.9997 |
| Ridge | ferroml | regression | 5000 | 20 | 0.8 | 0.0 | 0.9997 |
| Ridge | sklearn | regression | 5000 | 20 | 0.9 | 0.1 | 0.9997 |
| Lasso | ferroml | regression | 5000 | 20 | 0.3 | 0.0 | 0.6045 |
| Lasso | sklearn | regression | 5000 | 20 | 0.6 | 0.1 | 0.6045 |
| LogisticRegression | ferroml | classification | 5000 | 20 | 13.6 | 0.0 | 0.8350 |
| LogisticRegression | sklearn | classification | 5000 | 20 | 5.4 | 0.1 | 0.8350 |
| DecisionTreeRegressor | ferroml | regression | 5000 | 20 | 39.3 | 0.1 | 0.5046 |
| DecisionTreeRegressor | sklearn | regression | 5000 | 20 | 43.8 | 0.2 | 0.5043 |
| DecisionTreeClassifier | ferroml | classification | 5000 | 20 | 42.1 | 0.1 | 0.9020 |
| DecisionTreeClassifier | sklearn | classification | 5000 | 20 | 53.4 | 0.1 | 0.8980 |
| RandomForestRegressor | ferroml | regression | 5000 | 20 | 136.4 | 2.2 | 0.4104 |
| RandomForestRegressor | sklearn | regression | 5000 | 20 | 2664.4 | 13.7 | 0.7893 |
| RandomForestRegressor | xgboost | regression | 5000 | 20 | 3120.6 | 2.0 | 0.7806 |
| RandomForestRegressor | lightgbm | regression | 5000 | 20 | 76.8 | 1.2 | 0.6235 |
| RandomForestClassifier | ferroml | classification | 5000 | 20 | 136.0 | 1.9 | 0.9080 |
| RandomForestClassifier | sklearn | classification | 5000 | 20 | 760.5 | 9.9 | 0.9400 |
| RandomForestClassifier | xgboost | classification | 5000 | 20 | 1663.9 | 2.0 | 0.9350 |
| RandomForestClassifier | lightgbm | classification | 5000 | 20 | 70.1 | 1.3 | 0.8920 |
| GradientBoostingRegressor | ferroml | regression | 5000 | 20 | 2378.8 | 1.0 | 0.9178 |
| GradientBoostingRegressor | sklearn | regression | 5000 | 20 | 2795.7 | 2.5 | 0.9181 |
| GradientBoostingRegressor | xgboost | regression | 5000 | 20 | 163.2 | 0.4 | 0.9181 |
| GradientBoostingRegressor | lightgbm | regression | 5000 | 20 | 39.4 | 1.1 | 0.9225 |
| GradientBoostingClassifier | ferroml | classification | 5000 | 20 | 2312.3 | 1.0 | 0.9170 |
| GradientBoostingClassifier | sklearn | classification | 5000 | 20 | 2870.9 | 2.5 | 0.9560 |
| GradientBoostingClassifier | xgboost | classification | 5000 | 20 | 283.7 | 0.6 | 0.9560 |
| GradientBoostingClassifier | lightgbm | classification | 5000 | 20 | 38.7 | 1.5 | 0.9540 |
| HistGradientBoostingRegressor | ferroml | regression | 5000 | 20 | 2047.4 | 3.0 | 0.9211 |
| HistGradientBoostingRegressor | sklearn | regression | 5000 | 20 | 212.3 | 1.9 | 0.9247 |
| HistGradientBoostingRegressor | xgboost | regression | 5000 | 20 | 1273.9 | 1.6 | 0.9181 |
| HistGradientBoostingRegressor | lightgbm | regression | 5000 | 20 | 43.5 | 1.3 | 0.9225 |
| HistGradientBoostingClassifier | ferroml | classification | 5000 | 20 | 1852.2 | 3.1 | 0.9510 |
| HistGradientBoostingClassifier | sklearn | classification | 5000 | 20 | 122.3 | 1.7 | 0.9520 |
| HistGradientBoostingClassifier | xgboost | classification | 5000 | 20 | 187.9 | 1.2 | 0.9560 |
| HistGradientBoostingClassifier | lightgbm | classification | 5000 | 20 | 37.1 | 1.6 | 0.9540 |
| KNeighborsClassifier | ferroml | classification | 5000 | 20 | 1.1 | 8.2 | 0.9540 |
| KNeighborsClassifier | sklearn | classification | 5000 | 20 | 0.4 | 4.2 | 0.9540 |
| SVC | ferroml | classification | 5000 | 20 | 405.9 | 39.1 | 0.9690 |
| SVC | sklearn | classification | 5000 | 20 | 99.7 | 33.1 | 0.9560 |
| GaussianNB | ferroml | classification | 5000 | 20 | 0.3 | 0.1 | 0.8150 |
| GaussianNB | sklearn | classification | 5000 | 20 | 1.3 | 0.3 | 0.8150 |
| StandardScaler | ferroml | preprocessing | 5000 | 20 | 0.1 | 0.0 | N/A |
| StandardScaler | sklearn | preprocessing | 5000 | 20 | 0.6 | 0.1 | N/A |
| PCA | ferroml | preprocessing | 5000 | 20 | 1.9 | 0.0 | N/A |
| PCA | sklearn | preprocessing | 5000 | 20 | 0.6 | 0.1 | N/A |
| KMeans | ferroml | clustering | 5000 | 20 | 33.8 | 0.1 | 80200.9181 |
| KMeans | sklearn | clustering | 5000 | 20 | 15.4 | 0.2 | 80200.9181 |

## FerroML vs Others: Fit-Time Speedup

Values > 1.0 mean FerroML is faster.

| Algorithm | N | vs sklearn | vs xgboost | vs lightgbm |
|-----------|---|-----------|-----------|------------|
| LinearRegression | 1000 | 0.68x | N/A | N/A |
| Ridge | 1000 | 2.37x | N/A | N/A |
| Lasso | 1000 | 6.39x | N/A | N/A |
| LogisticRegression | 1000 | 4.37x | N/A | N/A |
| DecisionTreeRegressor | 1000 | 1.25x | N/A | N/A |
| DecisionTreeClassifier | 1000 | 1.58x | N/A | N/A |
| RandomForestRegressor | 1000 | 20.23x | 52.52x | 1.37x |
| RandomForestClassifier | 1000 | 8.35x | 7.56x | 1.73x |
| GradientBoostingRegressor | 1000 | 1.39x | 0.23x | 0.07x |
| GradientBoostingClassifier | 1000 | 1.62x | 0.16x | 0.06x |
| HistGradientBoostingRegressor | 1000 | 0.09x | 0.67x | 0.02x |
| HistGradientBoostingClassifier | 1000 | 0.08x | 0.76x | 0.02x |
| KNeighborsClassifier | 1000 | 2.70x | N/A | N/A |
| SVC | 1000 | 0.48x | N/A | N/A |
| GaussianNB | 1000 | 8.97x | N/A | N/A |
| StandardScaler | 1000 | 23.53x | N/A | N/A |
| PCA | 1000 | 1.43x | N/A | N/A |
| KMeans | 1000 | 0.41x | N/A | N/A |
| LinearRegression | 5000 | 0.80x | N/A | N/A |
| Ridge | 5000 | 1.12x | N/A | N/A |
| Lasso | 5000 | 1.88x | N/A | N/A |
| LogisticRegression | 5000 | 0.40x | N/A | N/A |
| DecisionTreeRegressor | 5000 | 1.11x | N/A | N/A |
| DecisionTreeClassifier | 5000 | 1.27x | N/A | N/A |
| RandomForestRegressor | 5000 | 19.53x | 22.88x | 0.56x |
| RandomForestClassifier | 5000 | 5.59x | 12.24x | 0.52x |
| GradientBoostingRegressor | 5000 | 1.18x | 0.07x | 0.02x |
| GradientBoostingClassifier | 5000 | 1.24x | 0.12x | 0.02x |
| HistGradientBoostingRegressor | 5000 | 0.10x | 0.62x | 0.02x |
| HistGradientBoostingClassifier | 5000 | 0.07x | 0.10x | 0.02x |
| KNeighborsClassifier | 5000 | 0.39x | N/A | N/A |
| SVC | 5000 | 0.25x | N/A | N/A |
| GaussianNB | 5000 | 4.31x | N/A | N/A |
| StandardScaler | 5000 | 8.89x | N/A | N/A |
| PCA | 5000 | 0.33x | N/A | N/A |
| KMeans | 5000 | 0.46x | N/A | N/A |
