# Cross-Library Benchmark Results

Generated: 2026-03-24T23:46:44.005313
Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
Python: 3.12.3
NumPy: 2.4.2
sklearn: 1.8.0
xgboost: 3.2.0
lightgbm: 4.6.0

## Summary Table

| Algorithm | Library | Task | N | Features | Fit (ms) | Predict (ms) | Score |
|-----------|---------|------|---|----------|----------|--------------|-------|
| LinearRegression | ferroml | regression | 1000 | 20 | 0.3 | 0.0 | 0.9996 |
| LinearRegression | sklearn | regression | 1000 | 20 | 0.7 | 0.1 | 0.9996 |
| Ridge | ferroml | regression | 1000 | 20 | 0.2 | 0.0 | 0.9996 |
| Ridge | sklearn | regression | 1000 | 20 | 0.6 | 0.1 | 0.9996 |
| Lasso | ferroml | regression | 1000 | 20 | 0.1 | 0.0 | 0.4862 |
| Lasso | sklearn | regression | 1000 | 20 | 0.5 | 0.1 | 0.4862 |
| LogisticRegression | ferroml | classification | 1000 | 20 | 0.5 | 0.0 | 0.8300 |
| LogisticRegression | sklearn | classification | 1000 | 20 | 3.2 | 0.1 | 0.8350 |
| DecisionTreeRegressor | ferroml | regression | 1000 | 20 | 6.1 | 0.0 | 0.0246 |
| DecisionTreeRegressor | sklearn | regression | 1000 | 20 | 9.2 | 0.1 | -0.0309 |
| DecisionTreeClassifier | ferroml | classification | 1000 | 20 | 5.9 | 0.0 | 0.8550 |
| DecisionTreeClassifier | sklearn | classification | 1000 | 20 | 9.2 | 0.1 | 0.8350 |
| RandomForestRegressor | ferroml | regression | 1000 | 20 | 24.0 | 0.7 | 0.4091 |
| RandomForestRegressor | sklearn | regression | 1000 | 20 | 497.2 | 6.5 | 0.6434 |
| RandomForestRegressor | xgboost | regression | 1000 | 20 | 1652.4 | 0.9 | 0.6540 |
| RandomForestRegressor | lightgbm | regression | 1000 | 20 | 35.4 | 0.7 | 0.4637 |
| RandomForestClassifier | ferroml | classification | 1000 | 20 | 21.2 | 0.5 | 0.9100 |
| RandomForestClassifier | sklearn | classification | 1000 | 20 | 180.7 | 6.1 | 0.9500 |
| RandomForestClassifier | xgboost | classification | 1000 | 20 | 402.9 | 0.6 | 0.9200 |
| RandomForestClassifier | lightgbm | classification | 1000 | 20 | 36.4 | 0.8 | 0.8500 |
| GradientBoostingRegressor | ferroml | regression | 1000 | 20 | 377.4 | 0.9 | 0.7690 |
| GradientBoostingRegressor | sklearn | regression | 1000 | 20 | 491.2 | 0.8 | 0.7686 |
| GradientBoostingRegressor | xgboost | regression | 1000 | 20 | 316.9 | 0.4 | 0.7816 |
| GradientBoostingRegressor | lightgbm | regression | 1000 | 20 | 21.8 | 0.6 | 0.8176 |
| GradientBoostingClassifier | ferroml | classification | 1000 | 20 | 345.9 | 0.8 | 0.8850 |
| GradientBoostingClassifier | sklearn | classification | 1000 | 20 | 546.2 | 0.7 | 0.9400 |
| GradientBoostingClassifier | xgboost | classification | 1000 | 20 | 293.7 | 0.5 | 0.9450 |
| GradientBoostingClassifier | lightgbm | classification | 1000 | 20 | 24.1 | 1.0 | 0.9500 |
| HistGradientBoostingRegressor | ferroml | regression | 1000 | 20 | 97.4 | 0.5 | 0.8203 |
| HistGradientBoostingRegressor | sklearn | regression | 1000 | 20 | 90.8 | 1.1 | 0.8095 |
| HistGradientBoostingRegressor | xgboost | regression | 1000 | 20 | 432.2 | 0.8 | 0.7816 |
| HistGradientBoostingRegressor | lightgbm | regression | 1000 | 20 | 19.9 | 0.8 | 0.8176 |
| HistGradientBoostingClassifier | ferroml | classification | 1000 | 20 | 100.7 | 0.5 | 0.9350 |
| HistGradientBoostingClassifier | sklearn | classification | 1000 | 20 | 101.9 | 1.0 | 0.9400 |
| HistGradientBoostingClassifier | xgboost | classification | 1000 | 20 | 480.8 | 0.5 | 0.9450 |
| HistGradientBoostingClassifier | lightgbm | classification | 1000 | 20 | 24.4 | 0.9 | 0.9500 |
| KNeighborsClassifier | ferroml | classification | 1000 | 20 | 0.2 | 1.2 | 0.9050 |
| KNeighborsClassifier | sklearn | classification | 1000 | 20 | 0.3 | 1.4 | 0.9050 |
| SVC | ferroml | classification | 1000 | 20 | 26.1 | 2.4 | 0.9550 |
| SVC | sklearn | classification | 1000 | 20 | 8.1 | 2.5 | 0.9450 |
| GaussianNB | ferroml | classification | 1000 | 20 | 0.1 | 0.0 | 0.8300 |
| GaussianNB | sklearn | classification | 1000 | 20 | 0.8 | 0.2 | 0.8300 |
| StandardScaler | ferroml | preprocessing | 1000 | 20 | 0.0 | 0.0 | N/A |
| StandardScaler | sklearn | preprocessing | 1000 | 20 | 0.4 | 0.1 | N/A |
| PCA | ferroml | preprocessing | 1000 | 20 | 0.1 | 0.0 | N/A |
| PCA | sklearn | preprocessing | 1000 | 20 | 0.5 | 0.1 | N/A |
| KMeans | ferroml | clustering | 1000 | 20 | 0.9 | 0.0 | 16074.9071 |
| KMeans | sklearn | clustering | 1000 | 20 | 13.4 | 0.2 | 16074.9071 |
| LinearRegression | ferroml | regression | 5000 | 20 | 1.2 | 0.0 | 0.9997 |
| LinearRegression | sklearn | regression | 5000 | 20 | 1.6 | 0.1 | 0.9997 |
| Ridge | ferroml | regression | 5000 | 20 | 1.0 | 0.0 | 0.9997 |
| Ridge | sklearn | regression | 5000 | 20 | 1.0 | 0.1 | 0.9997 |
| Lasso | ferroml | regression | 5000 | 20 | 0.5 | 0.1 | 0.6045 |
| Lasso | sklearn | regression | 5000 | 20 | 0.8 | 0.1 | 0.6045 |
| LogisticRegression | ferroml | classification | 5000 | 20 | 13.9 | 0.0 | 0.8350 |
| LogisticRegression | sklearn | classification | 5000 | 20 | 5.8 | 0.1 | 0.8350 |
| DecisionTreeRegressor | ferroml | regression | 5000 | 20 | 42.5 | 0.1 | 0.5046 |
| DecisionTreeRegressor | sklearn | regression | 5000 | 20 | 47.6 | 0.2 | 0.5083 |
| DecisionTreeClassifier | ferroml | classification | 5000 | 20 | 41.8 | 0.1 | 0.9020 |
| DecisionTreeClassifier | sklearn | classification | 5000 | 20 | 56.5 | 0.1 | 0.8980 |
| RandomForestRegressor | ferroml | regression | 5000 | 20 | 136.0 | 1.7 | 0.3712 |
| RandomForestRegressor | sklearn | regression | 5000 | 20 | 2785.7 | 13.2 | 0.7893 |
| RandomForestRegressor | xgboost | regression | 5000 | 20 | 2552.5 | 1.4 | 0.7806 |
| RandomForestRegressor | lightgbm | regression | 5000 | 20 | 75.5 | 1.1 | 0.6235 |
| RandomForestClassifier | ferroml | classification | 5000 | 20 | 137.7 | 1.8 | 0.9130 |
| RandomForestClassifier | sklearn | classification | 5000 | 20 | 762.8 | 10.2 | 0.9400 |
| RandomForestClassifier | xgboost | classification | 5000 | 20 | 1087.3 | 1.5 | 0.9350 |
| RandomForestClassifier | lightgbm | classification | 5000 | 20 | 63.8 | 1.2 | 0.8920 |
| GradientBoostingRegressor | ferroml | regression | 5000 | 20 | 2397.7 | 1.1 | 0.9178 |
| GradientBoostingRegressor | sklearn | regression | 5000 | 20 | 2863.4 | 2.6 | 0.9181 |
| GradientBoostingRegressor | xgboost | regression | 5000 | 20 | 470.8 | 1.0 | 0.9181 |
| GradientBoostingRegressor | lightgbm | regression | 5000 | 20 | 41.3 | 1.3 | 0.9225 |
| GradientBoostingClassifier | ferroml | classification | 5000 | 20 | 2370.5 | 1.1 | 0.9170 |
| GradientBoostingClassifier | sklearn | classification | 5000 | 20 | 2953.3 | 2.4 | 0.9560 |
| GradientBoostingClassifier | xgboost | classification | 5000 | 20 | 64.1 | 0.5 | 0.9560 |
| GradientBoostingClassifier | lightgbm | classification | 5000 | 20 | 38.2 | 1.3 | 0.9540 |
| HistGradientBoostingRegressor | ferroml | regression | 5000 | 20 | 231.8 | 2.3 | 0.9211 |
| HistGradientBoostingRegressor | sklearn | regression | 5000 | 20 | 128.5 | 1.3 | 0.9247 |
| HistGradientBoostingRegressor | xgboost | regression | 5000 | 20 | 80.6 | 0.5 | 0.9181 |
| HistGradientBoostingRegressor | lightgbm | regression | 5000 | 20 | 39.5 | 1.4 | 0.9225 |
| HistGradientBoostingClassifier | ferroml | classification | 5000 | 20 | 215.9 | 2.2 | 0.9510 |
| HistGradientBoostingClassifier | sklearn | classification | 5000 | 20 | 126.9 | 1.4 | 0.9520 |
| HistGradientBoostingClassifier | xgboost | classification | 5000 | 20 | 352.3 | 0.9 | 0.9560 |
| HistGradientBoostingClassifier | lightgbm | classification | 5000 | 20 | 41.5 | 1.8 | 0.9540 |
| KNeighborsClassifier | ferroml | classification | 5000 | 20 | 1.0 | 9.4 | 0.9540 |
| KNeighborsClassifier | sklearn | classification | 5000 | 20 | 0.4 | 4.3 | 0.9540 |
| SVC | ferroml | classification | 5000 | 20 | 1033.4 | 39.1 | 0.9690 |
| SVC | sklearn | classification | 5000 | 20 | 105.0 | 36.9 | 0.9560 |
| GaussianNB | ferroml | classification | 5000 | 20 | 0.4 | 0.1 | 0.8150 |
| GaussianNB | sklearn | classification | 5000 | 20 | 1.2 | 0.2 | 0.8150 |
| StandardScaler | ferroml | preprocessing | 5000 | 20 | 0.2 | 0.1 | N/A |
| StandardScaler | sklearn | preprocessing | 5000 | 20 | 0.7 | 0.1 | N/A |
| PCA | ferroml | preprocessing | 5000 | 20 | 0.4 | 0.1 | N/A |
| PCA | sklearn | preprocessing | 5000 | 20 | 0.5 | 0.1 | N/A |
| KMeans | ferroml | clustering | 5000 | 20 | 4.0 | 0.1 | 80200.9181 |
| KMeans | sklearn | clustering | 5000 | 20 | 14.9 | 0.2 | 80200.9181 |
| LinearRegression | ferroml | regression | 10000 | 20 | 2.2 | 0.1 | 0.9993 |
| LinearRegression | sklearn | regression | 10000 | 20 | 2.5 | 0.1 | 0.9993 |
| Ridge | ferroml | regression | 10000 | 20 | 1.9 | 0.1 | 0.9993 |
| Ridge | sklearn | regression | 10000 | 20 | 1.2 | 0.1 | 0.9993 |
| Lasso | ferroml | regression | 10000 | 20 | 0.8 | 0.1 | 0.3889 |
| Lasso | sklearn | regression | 10000 | 20 | 0.8 | 0.1 | 0.3889 |
| LogisticRegression | ferroml | classification | 10000 | 20 | 24.7 | 0.1 | 0.8175 |
| LogisticRegression | sklearn | classification | 10000 | 20 | 8.5 | 0.1 | 0.8175 |
| DecisionTreeRegressor | ferroml | regression | 10000 | 20 | 84.3 | 0.2 | 0.5021 |
| DecisionTreeRegressor | sklearn | regression | 10000 | 20 | 96.3 | 0.2 | 0.5036 |
| DecisionTreeClassifier | ferroml | classification | 10000 | 20 | 91.8 | 0.2 | 0.8660 |
| DecisionTreeClassifier | sklearn | classification | 10000 | 20 | 119.2 | 0.2 | 0.8635 |
| RandomForestRegressor | ferroml | regression | 10000 | 20 | 303.5 | 3.0 | 0.4217 |
| RandomForestRegressor | sklearn | regression | 10000 | 20 | 5889.5 | 21.4 | 0.7863 |
| RandomForestRegressor | xgboost | regression | 10000 | 20 | 1863.7 | 1.1 | 0.7785 |
| RandomForestRegressor | lightgbm | regression | 10000 | 20 | 93.4 | 1.4 | 0.5537 |
| RandomForestClassifier | ferroml | classification | 10000 | 20 | 323.2 | 2.7 | 0.8935 |
| RandomForestClassifier | sklearn | classification | 10000 | 20 | 1603.8 | 16.6 | 0.9300 |
| RandomForestClassifier | xgboost | classification | 10000 | 20 | 1471.4 | 1.8 | 0.9210 |
| RandomForestClassifier | lightgbm | classification | 10000 | 20 | 90.6 | 1.7 | 0.8650 |
| GradientBoostingRegressor | ferroml | regression | 10000 | 20 | 5593.4 | 1.6 | 0.9298 |
| GradientBoostingRegressor | sklearn | regression | 10000 | 20 | 6157.0 | 4.9 | 0.9296 |
| GradientBoostingRegressor | xgboost | regression | 10000 | 20 | 422.3 | 1.2 | 0.9323 |
| GradientBoostingRegressor | lightgbm | regression | 10000 | 20 | 50.4 | 1.8 | 0.9336 |
| GradientBoostingClassifier | ferroml | classification | 10000 | 20 | 5435.5 | 1.4 | 0.9060 |
| GradientBoostingClassifier | sklearn | classification | 10000 | 20 | 6373.1 | 4.4 | 0.9285 |
| GradientBoostingClassifier | xgboost | classification | 10000 | 20 | 226.1 | 1.0 | 0.9335 |
| GradientBoostingClassifier | lightgbm | classification | 10000 | 20 | 52.7 | 2.2 | 0.9390 |
| HistGradientBoostingRegressor | ferroml | regression | 10000 | 20 | 415.6 | 5.5 | 0.9301 |
| HistGradientBoostingRegressor | sklearn | regression | 10000 | 20 | 146.6 | 2.1 | 0.9313 |
| HistGradientBoostingRegressor | xgboost | regression | 10000 | 20 | 557.0 | 0.6 | 0.9323 |
| HistGradientBoostingRegressor | lightgbm | regression | 10000 | 20 | 51.9 | 1.8 | 0.9336 |
| HistGradientBoostingClassifier | ferroml | classification | 10000 | 20 | 399.9 | 4.4 | 0.9310 |
| HistGradientBoostingClassifier | sklearn | classification | 10000 | 20 | 142.7 | 2.3 | 0.9355 |
| HistGradientBoostingClassifier | xgboost | classification | 10000 | 20 | 359.4 | 1.3 | 0.9335 |
| HistGradientBoostingClassifier | lightgbm | classification | 10000 | 20 | 53.0 | 2.1 | 0.9390 |
| KNeighborsClassifier | ferroml | classification | 10000 | 20 | 2.3 | 29.9 | 0.9425 |
| KNeighborsClassifier | sklearn | classification | 10000 | 20 | 0.4 | 13.7 | 0.9425 |
| SVC | ferroml | classification | 5000 | 20 | 1015.4 | 39.6 | 0.9690 |
| SVC | sklearn | classification | 5000 | 20 | 99.8 | 34.5 | 0.9560 |
| GaussianNB | ferroml | classification | 10000 | 20 | 0.8 | 0.2 | 0.7995 |
| GaussianNB | sklearn | classification | 10000 | 20 | 1.9 | 0.4 | 0.7995 |
| StandardScaler | ferroml | preprocessing | 10000 | 20 | 0.4 | 0.1 | N/A |
| StandardScaler | sklearn | preprocessing | 10000 | 20 | 1.1 | 0.2 | N/A |
| PCA | ferroml | preprocessing | 10000 | 20 | 0.7 | 0.1 | N/A |
| PCA | sklearn | preprocessing | 10000 | 20 | 0.7 | 0.1 | N/A |
| KMeans | ferroml | clustering | 10000 | 20 | 14.9 | 0.1 | 159775.1787 |
| KMeans | sklearn | clustering | 10000 | 20 | 18.9 | 0.3 | 159775.1787 |

## FerroML vs Others: Fit-Time Speedup

Values > 1.0 mean FerroML is faster.

| Algorithm | N | vs sklearn | vs xgboost | vs lightgbm |
|-----------|---|-----------|-----------|------------|
| LinearRegression | 1000 | 2.56x | N/A | N/A |
| Ridge | 1000 | 2.51x | N/A | N/A |
| Lasso | 1000 | 5.76x | N/A | N/A |
| LogisticRegression | 1000 | 6.52x | N/A | N/A |
| DecisionTreeRegressor | 1000 | 1.52x | N/A | N/A |
| DecisionTreeClassifier | 1000 | 1.57x | N/A | N/A |
| RandomForestRegressor | 1000 | 20.72x | 68.86x | 1.48x |
| RandomForestClassifier | 1000 | 8.53x | 19.03x | 1.72x |
| GradientBoostingRegressor | 1000 | 1.30x | 0.84x | 0.06x |
| GradientBoostingClassifier | 1000 | 1.58x | 0.85x | 0.07x |
| HistGradientBoostingRegressor | 1000 | 0.93x | 4.44x | 0.20x |
| HistGradientBoostingClassifier | 1000 | 1.01x | 4.78x | 0.24x |
| KNeighborsClassifier | 1000 | 1.63x | N/A | N/A |
| SVC | 1000 | 0.31x | N/A | N/A |
| GaussianNB | 1000 | 8.71x | N/A | N/A |
| StandardScaler | 1000 | 9.44x | N/A | N/A |
| PCA | 1000 | 3.57x | N/A | N/A |
| KMeans | 1000 | 15.17x | N/A | N/A |
| LinearRegression | 5000 | 1.31x | N/A | N/A |
| Ridge | 5000 | 1.00x | N/A | N/A |
| Lasso | 5000 | 1.41x | N/A | N/A |
| LogisticRegression | 5000 | 0.41x | N/A | N/A |
| DecisionTreeRegressor | 5000 | 1.12x | N/A | N/A |
| DecisionTreeClassifier | 5000 | 1.35x | N/A | N/A |
| RandomForestRegressor | 5000 | 20.48x | 18.76x | 0.55x |
| RandomForestClassifier | 5000 | 5.54x | 7.90x | 0.46x |
| GradientBoostingRegressor | 5000 | 1.19x | 0.20x | 0.02x |
| GradientBoostingClassifier | 5000 | 1.25x | 0.03x | 0.02x |
| HistGradientBoostingRegressor | 5000 | 0.55x | 0.35x | 0.17x |
| HistGradientBoostingClassifier | 5000 | 0.59x | 1.63x | 0.19x |
| KNeighborsClassifier | 5000 | 0.38x | N/A | N/A |
| SVC | 5000 | 0.10x | N/A | N/A |
| GaussianNB | 5000 | 3.17x | N/A | N/A |
| StandardScaler | 5000 | 3.83x | N/A | N/A |
| PCA | 5000 | 1.13x | N/A | N/A |
| KMeans | 5000 | 3.70x | N/A | N/A |
| LinearRegression | 10000 | 1.13x | N/A | N/A |
| Ridge | 10000 | 0.61x | N/A | N/A |
| Lasso | 10000 | 1.01x | N/A | N/A |
| LogisticRegression | 10000 | 0.34x | N/A | N/A |
| DecisionTreeRegressor | 10000 | 1.14x | N/A | N/A |
| DecisionTreeClassifier | 10000 | 1.30x | N/A | N/A |
| RandomForestRegressor | 10000 | 19.40x | 6.14x | 0.31x |
| RandomForestClassifier | 10000 | 4.96x | 4.55x | 0.28x |
| GradientBoostingRegressor | 10000 | 1.10x | 0.08x | 0.01x |
| GradientBoostingClassifier | 10000 | 1.17x | 0.04x | 0.01x |
| HistGradientBoostingRegressor | 10000 | 0.35x | 1.34x | 0.12x |
| HistGradientBoostingClassifier | 10000 | 0.36x | 0.90x | 0.13x |
| KNeighborsClassifier | 10000 | 0.18x | N/A | N/A |
| GaussianNB | 10000 | 2.49x | N/A | N/A |
| StandardScaler | 10000 | 3.03x | N/A | N/A |
| PCA | 10000 | 0.90x | N/A | N/A |
| KMeans | 10000 | 1.27x | N/A | N/A |
