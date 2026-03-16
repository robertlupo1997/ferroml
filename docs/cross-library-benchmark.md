# Cross-Library Benchmark Results

Generated: 2026-03-15T18:06:04.155528
Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
Python: 3.12.3
NumPy: 2.4.2
sklearn: 1.8.0
xgboost: 3.2.0
lightgbm: 4.6.0

## Summary Table

| Algorithm | Library | Task | N | Features | Fit (ms) | Predict (ms) | Score |
|-----------|---------|------|---|----------|----------|--------------|-------|
| LinearRegression | ferroml | regression | 1000 | 20 | 3.6 | 0.0 | 0.9996 |
| LinearRegression | sklearn | regression | 1000 | 20 | 1.3 | 0.1 | 0.9996 |
| Ridge | ferroml | regression | 1000 | 20 | 0.4 | 0.0 | 0.9996 |
| Ridge | sklearn | regression | 1000 | 20 | 1.3 | 0.1 | 0.9996 |
| Lasso | ferroml | regression | 1000 | 20 | 0.1 | 0.0 | 0.4862 |
| Lasso | sklearn | regression | 1000 | 20 | 1.1 | 0.1 | 0.4862 |
| LogisticRegression | ferroml | classification | 1000 | 20 | 0.7 | 0.0 | 0.8300 |
| LogisticRegression | sklearn | classification | 1000 | 20 | 19.5 | 0.1 | 0.8350 |
| DecisionTreeRegressor | ferroml | regression | 1000 | 20 | 8.4 | 0.0 | 0.0246 |
| DecisionTreeRegressor | sklearn | regression | 1000 | 20 | 9.8 | 0.2 | 0.0149 |
| DecisionTreeClassifier | ferroml | classification | 1000 | 20 | 9.1 | 0.0 | 0.8550 |
| DecisionTreeClassifier | sklearn | classification | 1000 | 20 | 12.4 | 0.2 | 0.8400 |
| RandomForestRegressor | ferroml | regression | 1000 | 20 | 44.7 | 2.6 | 0.3913 |
| RandomForestRegressor | sklearn | regression | 1000 | 20 | 833.1 | 12.5 | 0.6434 |
| RandomForestRegressor | xgboost | regression | 1000 | 20 | 21838.4 | 23.9 | 0.6540 |
| RandomForestRegressor | lightgbm | regression | 1000 | 20 | 60.2 | 1.1 | 0.4637 |
| RandomForestClassifier | ferroml | classification | 1000 | 20 | 29.4 | 1.5 | 0.8600 |
| RandomForestClassifier | sklearn | classification | 1000 | 20 | 239.4 | 7.6 | 0.9500 |
| RandomForestClassifier | xgboost | classification | 1000 | 20 | 18994.9 | 36.0 | 0.9200 |
| RandomForestClassifier | lightgbm | classification | 1000 | 20 | 53.8 | 2.3 | 0.8500 |
| GradientBoostingRegressor | ferroml | regression | 1000 | 20 | 410.1 | 1.2 | 0.7690 |
| GradientBoostingRegressor | sklearn | regression | 1000 | 20 | 579.9 | 0.9 | 0.7686 |
| GradientBoostingRegressor | xgboost | regression | 1000 | 20 | 17594.3 | 0.8 | 0.7816 |
| GradientBoostingRegressor | lightgbm | regression | 1000 | 20 | 28.9 | 0.9 | 0.8176 |
| GradientBoostingClassifier | ferroml | classification | 1000 | 20 | 371.8 | 1.3 | 0.8850 |
| GradientBoostingClassifier | sklearn | classification | 1000 | 20 | 1081.6 | 1.3 | 0.9400 |
| GradientBoostingClassifier | xgboost | classification | 1000 | 20 | 29762.6 | 0.9 | 0.9450 |
| GradientBoostingClassifier | lightgbm | classification | 1000 | 20 | 28.3 | 1.1 | 0.9500 |
| HistGradientBoostingRegressor | ferroml | regression | 1000 | 20 | 136.8 | 0.5 | 0.8203 |
| HistGradientBoostingRegressor | sklearn | regression | 1000 | 20 | 107.1 | 1.2 | 0.8095 |
| HistGradientBoostingRegressor | xgboost | regression | 1000 | 20 | 484.6 | 1.1 | 0.7816 |
| HistGradientBoostingRegressor | lightgbm | regression | 1000 | 20 | 24.4 | 0.7 | 0.8176 |
| HistGradientBoostingClassifier | ferroml | classification | 1000 | 20 | 124.6 | 0.6 | 0.9350 |
| HistGradientBoostingClassifier | sklearn | classification | 1000 | 20 | 104.5 | 0.9 | 0.9400 |
| HistGradientBoostingClassifier | xgboost | classification | 1000 | 20 | 342.5 | 1.4 | 0.9450 |
| HistGradientBoostingClassifier | lightgbm | classification | 1000 | 20 | 28.3 | 0.9 | 0.9500 |
| KNeighborsClassifier | ferroml | classification | 1000 | 20 | 0.2 | 1.4 | 0.9050 |
| KNeighborsClassifier | sklearn | classification | 1000 | 20 | 0.3 | 1.8 | 0.9050 |
| SVC | ferroml | classification | 1000 | 20 | 32.3 | 2.3 | 0.9550 |
| SVC | sklearn | classification | 1000 | 20 | 8.0 | 2.9 | 0.9450 |
| GaussianNB | ferroml | classification | 1000 | 20 | 0.1 | 0.0 | 0.8300 |
| GaussianNB | sklearn | classification | 1000 | 20 | 1.0 | 0.2 | 0.8300 |
| StandardScaler | ferroml | preprocessing | 1000 | 20 | 0.0 | 0.0 | N/A |
| StandardScaler | sklearn | preprocessing | 1000 | 20 | 0.4 | 0.1 | N/A |
| PCA | ferroml | preprocessing | 1000 | 20 | 0.5 | 0.0 | N/A |
| PCA | sklearn | preprocessing | 1000 | 20 | 0.5 | 0.1 | N/A |
| KMeans | ferroml | clustering | 1000 | 20 | 1.4 | 0.0 | 16074.9071 |
| KMeans | sklearn | clustering | 1000 | 20 | 11.9 | 0.1 | 16074.9071 |
| LinearRegression | ferroml | regression | 5000 | 20 | 2.2 | 0.0 | 0.9997 |
| LinearRegression | sklearn | regression | 5000 | 20 | 1.4 | 0.1 | 0.9997 |
| Ridge | ferroml | regression | 5000 | 20 | 0.9 | 0.0 | 0.9997 |
| Ridge | sklearn | regression | 5000 | 20 | 0.9 | 0.1 | 0.9997 |
| Lasso | ferroml | regression | 5000 | 20 | 0.3 | 0.0 | 0.6045 |
| Lasso | sklearn | regression | 5000 | 20 | 1.0 | 0.1 | 0.6045 |
| LogisticRegression | ferroml | classification | 5000 | 20 | 16.6 | 0.0 | 0.8350 |
| LogisticRegression | sklearn | classification | 5000 | 20 | 7.8 | 0.1 | 0.8350 |
| DecisionTreeRegressor | ferroml | regression | 5000 | 20 | 42.7 | 0.1 | 0.5046 |
| DecisionTreeRegressor | sklearn | regression | 5000 | 20 | 45.3 | 0.1 | 0.4910 |
| DecisionTreeClassifier | ferroml | classification | 5000 | 20 | 42.7 | 0.1 | 0.9020 |
| DecisionTreeClassifier | sklearn | classification | 5000 | 20 | 57.2 | 0.2 | 0.8970 |
| RandomForestRegressor | ferroml | regression | 5000 | 20 | 148.4 | 2.5 | 0.4190 |
| RandomForestRegressor | sklearn | regression | 5000 | 20 | 2826.8 | 14.0 | 0.7893 |
| RandomForestRegressor | xgboost | regression | 5000 | 20 | 6408.7 | 1.5 | 0.7806 |
| RandomForestRegressor | lightgbm | regression | 5000 | 20 | 86.7 | 1.6 | 0.6235 |
| RandomForestClassifier | ferroml | classification | 5000 | 20 | 165.9 | 2.1 | 0.9180 |
| RandomForestClassifier | sklearn | classification | 5000 | 20 | 840.4 | 12.1 | 0.9400 |
| RandomForestClassifier | xgboost | classification | 5000 | 20 | 1899.8 | 28.0 | 0.9350 |
| RandomForestClassifier | lightgbm | classification | 5000 | 20 | 99.9 | 1.7 | 0.8920 |
| GradientBoostingRegressor | ferroml | regression | 5000 | 20 | 2590.2 | 1.4 | 0.9178 |
| GradientBoostingRegressor | sklearn | regression | 5000 | 20 | 2999.7 | 2.9 | 0.9181 |
| GradientBoostingRegressor | xgboost | regression | 5000 | 20 | 990.3 | 6.1 | 0.9181 |
| GradientBoostingRegressor | lightgbm | regression | 5000 | 20 | 51.5 | 1.6 | 0.9225 |
| GradientBoostingClassifier | ferroml | classification | 5000 | 20 | 2570.5 | 1.7 | 0.9170 |
| GradientBoostingClassifier | sklearn | classification | 5000 | 20 | 3118.6 | 2.7 | 0.9560 |
| GradientBoostingClassifier | xgboost | classification | 5000 | 20 | 503.8 | 0.8 | 0.9560 |
| GradientBoostingClassifier | lightgbm | classification | 5000 | 20 | 50.2 | 1.6 | 0.9540 |
| HistGradientBoostingRegressor | ferroml | regression | 5000 | 20 | 450.0 | 2.7 | 0.9211 |
| HistGradientBoostingRegressor | sklearn | regression | 5000 | 20 | 263.3 | 2.7 | 0.9247 |
| HistGradientBoostingRegressor | xgboost | regression | 5000 | 20 | 642.2 | 2.8 | 0.9181 |
| HistGradientBoostingRegressor | lightgbm | regression | 5000 | 20 | 57.4 | 1.2 | 0.9225 |
| HistGradientBoostingClassifier | ferroml | classification | 5000 | 20 | 414.8 | 2.4 | 0.9510 |
| HistGradientBoostingClassifier | sklearn | classification | 5000 | 20 | 162.1 | 2.1 | 0.9520 |
| HistGradientBoostingClassifier | xgboost | classification | 5000 | 20 | 13228.0 | 1.6 | 0.9560 |
| HistGradientBoostingClassifier | lightgbm | classification | 5000 | 20 | 50.8 | 1.8 | 0.9540 |
| KNeighborsClassifier | ferroml | classification | 5000 | 20 | 0.9 | 10.2 | 0.9540 |
| KNeighborsClassifier | sklearn | classification | 5000 | 20 | 0.4 | 5.0 | 0.9540 |
| SVC | ferroml | classification | 5000 | 20 | 1825.8 | 40.2 | 0.9690 |
| SVC | sklearn | classification | 5000 | 20 | 103.9 | 35.3 | 0.9560 |
| GaussianNB | ferroml | classification | 5000 | 20 | 0.3 | 0.1 | 0.8150 |
| GaussianNB | sklearn | classification | 5000 | 20 | 1.3 | 0.2 | 0.8150 |
| StandardScaler | ferroml | preprocessing | 5000 | 20 | 0.1 | 0.0 | N/A |
| StandardScaler | sklearn | preprocessing | 5000 | 20 | 0.9 | 0.1 | N/A |
| PCA | ferroml | preprocessing | 5000 | 20 | 2.1 | 0.1 | N/A |
| PCA | sklearn | preprocessing | 5000 | 20 | 1.0 | 0.1 | N/A |
| KMeans | ferroml | clustering | 5000 | 20 | 5.6 | 0.0 | 80200.9181 |
| KMeans | sklearn | clustering | 5000 | 20 | 19.1 | 0.2 | 80200.9181 |
| LinearRegression | ferroml | regression | 10000 | 20 | 5.1 | 0.0 | 0.9993 |
| LinearRegression | sklearn | regression | 10000 | 20 | 2.9 | 0.1 | 0.9993 |
| Ridge | ferroml | regression | 10000 | 20 | 2.3 | 0.0 | 0.9993 |
| Ridge | sklearn | regression | 10000 | 20 | 1.5 | 0.1 | 0.9993 |
| Lasso | ferroml | regression | 10000 | 20 | 0.8 | 0.0 | 0.3889 |
| Lasso | sklearn | regression | 10000 | 20 | 1.2 | 0.1 | 0.3889 |
| LogisticRegression | ferroml | classification | 10000 | 20 | 21.4 | 0.1 | 0.8175 |
| LogisticRegression | sklearn | classification | 10000 | 20 | 9.0 | 0.2 | 0.8175 |
| DecisionTreeRegressor | ferroml | regression | 10000 | 20 | 88.0 | 0.3 | 0.5021 |
| DecisionTreeRegressor | sklearn | regression | 10000 | 20 | 98.6 | 0.2 | 0.4988 |
| DecisionTreeClassifier | ferroml | classification | 10000 | 20 | 98.9 | 0.1 | 0.8660 |
| DecisionTreeClassifier | sklearn | classification | 10000 | 20 | 125.9 | 0.2 | 0.8675 |
| RandomForestRegressor | ferroml | regression | 10000 | 20 | 378.3 | 3.2 | 0.4242 |
| RandomForestRegressor | sklearn | regression | 10000 | 20 | 6109.1 | 23.2 | 0.7863 |
| RandomForestRegressor | xgboost | regression | 10000 | 20 | 4705.1 | 3.8 | 0.7785 |
| RandomForestRegressor | lightgbm | regression | 10000 | 20 | 109.6 | 1.9 | 0.5537 |
| RandomForestClassifier | ferroml | classification | 10000 | 20 | 394.9 | 3.8 | 0.8915 |
| RandomForestClassifier | sklearn | classification | 10000 | 20 | 1689.9 | 17.8 | 0.9300 |
| RandomForestClassifier | xgboost | classification | 10000 | 20 | 41007.5 | 24.2 | 0.9210 |
| RandomForestClassifier | lightgbm | classification | 10000 | 20 | 107.4 | 2.0 | 0.8650 |
| GradientBoostingRegressor | ferroml | regression | 10000 | 20 | 6310.4 | 2.4 | 0.9298 |
| GradientBoostingRegressor | sklearn | regression | 10000 | 20 | 6511.6 | 5.2 | 0.9296 |
| GradientBoostingRegressor | xgboost | regression | 10000 | 20 | 447.8 | 2.3 | 0.9323 |
| GradientBoostingRegressor | lightgbm | regression | 10000 | 20 | 58.5 | 2.3 | 0.9336 |
| GradientBoostingClassifier | ferroml | classification | 10000 | 20 | 5633.7 | 1.9 | 0.9060 |
| GradientBoostingClassifier | sklearn | classification | 10000 | 20 | 6561.5 | 4.7 | 0.9285 |
| GradientBoostingClassifier | xgboost | classification | 10000 | 20 | 1144.1 | 1.1 | 0.9335 |
| GradientBoostingClassifier | lightgbm | classification | 10000 | 20 | 61.4 | 2.5 | 0.9390 |
| HistGradientBoostingRegressor | ferroml | regression | 10000 | 20 | 660.7 | 4.8 | 0.9301 |
| HistGradientBoostingRegressor | sklearn | regression | 10000 | 20 | 177.0 | 2.2 | 0.9313 |
| HistGradientBoostingRegressor | xgboost | regression | 10000 | 20 | 350.6 | 1.0 | 0.9323 |
| HistGradientBoostingRegressor | lightgbm | regression | 10000 | 20 | 62.5 | 2.4 | 0.9336 |
| HistGradientBoostingClassifier | ferroml | classification | 10000 | 20 | 628.9 | 4.7 | 0.9310 |
| HistGradientBoostingClassifier | sklearn | classification | 10000 | 20 | 169.0 | 2.5 | 0.9355 |
| HistGradientBoostingClassifier | xgboost | classification | 10000 | 20 | 286.3 | 0.8 | 0.9335 |
| HistGradientBoostingClassifier | lightgbm | classification | 10000 | 20 | 57.4 | 2.3 | 0.9390 |
| KNeighborsClassifier | ferroml | classification | 10000 | 20 | 2.3 | 32.8 | 0.9425 |
| KNeighborsClassifier | sklearn | classification | 10000 | 20 | 0.6 | 14.7 | 0.9425 |
| SVC | ferroml | classification | 5000 | 20 | 1727.1 | 39.9 | 0.9690 |
| SVC | sklearn | classification | 5000 | 20 | 103.7 | 36.6 | 0.9560 |
| GaussianNB | ferroml | classification | 10000 | 20 | 0.7 | 0.2 | 0.7995 |
| GaussianNB | sklearn | classification | 10000 | 20 | 2.3 | 0.4 | 0.7995 |
| StandardScaler | ferroml | preprocessing | 10000 | 20 | 0.2 | 0.1 | N/A |
| StandardScaler | sklearn | preprocessing | 10000 | 20 | 1.5 | 0.2 | N/A |
| PCA | ferroml | preprocessing | 10000 | 20 | 4.9 | 0.1 | N/A |
| PCA | sklearn | preprocessing | 10000 | 20 | 0.7 | 0.1 | N/A |
| KMeans | ferroml | clustering | 10000 | 20 | 19.5 | 0.1 | 159775.1787 |
| KMeans | sklearn | clustering | 10000 | 20 | 23.1 | 0.6 | 159775.1787 |

## FerroML vs Others: Fit-Time Speedup

Values > 1.0 mean FerroML is faster.

| Algorithm | N | vs sklearn | vs xgboost | vs lightgbm |
|-----------|---|-----------|-----------|------------|
| LinearRegression | 1000 | 0.36x | N/A | N/A |
| Ridge | 1000 | 3.39x | N/A | N/A |
| Lasso | 1000 | 11.13x | N/A | N/A |
| LogisticRegression | 1000 | 26.11x | N/A | N/A |
| DecisionTreeRegressor | 1000 | 1.17x | N/A | N/A |
| DecisionTreeClassifier | 1000 | 1.37x | N/A | N/A |
| RandomForestRegressor | 1000 | 18.65x | 488.96x | 1.35x |
| RandomForestClassifier | 1000 | 8.14x | 646.06x | 1.83x |
| GradientBoostingRegressor | 1000 | 1.41x | 42.90x | 0.07x |
| GradientBoostingClassifier | 1000 | 2.91x | 80.04x | 0.08x |
| HistGradientBoostingRegressor | 1000 | 0.78x | 3.54x | 0.18x |
| HistGradientBoostingClassifier | 1000 | 0.84x | 2.75x | 0.23x |
| KNeighborsClassifier | 1000 | 1.75x | N/A | N/A |
| SVC | 1000 | 0.25x | N/A | N/A |
| GaussianNB | 1000 | 13.28x | N/A | N/A |
| StandardScaler | 1000 | 16.96x | N/A | N/A |
| PCA | 1000 | 1.07x | N/A | N/A |
| KMeans | 1000 | 8.30x | N/A | N/A |
| LinearRegression | 5000 | 0.62x | N/A | N/A |
| Ridge | 5000 | 1.07x | N/A | N/A |
| Lasso | 5000 | 3.03x | N/A | N/A |
| LogisticRegression | 5000 | 0.47x | N/A | N/A |
| DecisionTreeRegressor | 5000 | 1.06x | N/A | N/A |
| DecisionTreeClassifier | 5000 | 1.34x | N/A | N/A |
| RandomForestRegressor | 5000 | 19.04x | 43.17x | 0.58x |
| RandomForestClassifier | 5000 | 5.06x | 11.45x | 0.60x |
| GradientBoostingRegressor | 5000 | 1.16x | 0.38x | 0.02x |
| GradientBoostingClassifier | 5000 | 1.21x | 0.20x | 0.02x |
| HistGradientBoostingRegressor | 5000 | 0.59x | 1.43x | 0.13x |
| HistGradientBoostingClassifier | 5000 | 0.39x | 31.89x | 0.12x |
| KNeighborsClassifier | 5000 | 0.44x | N/A | N/A |
| SVC | 5000 | 0.06x | N/A | N/A |
| GaussianNB | 5000 | 3.76x | N/A | N/A |
| StandardScaler | 5000 | 9.46x | N/A | N/A |
| PCA | 5000 | 0.50x | N/A | N/A |
| KMeans | 5000 | 3.41x | N/A | N/A |
| LinearRegression | 10000 | 0.57x | N/A | N/A |
| Ridge | 10000 | 0.66x | N/A | N/A |
| Lasso | 10000 | 1.44x | N/A | N/A |
| LogisticRegression | 10000 | 0.42x | N/A | N/A |
| DecisionTreeRegressor | 10000 | 1.12x | N/A | N/A |
| DecisionTreeClassifier | 10000 | 1.27x | N/A | N/A |
| RandomForestRegressor | 10000 | 16.15x | 12.44x | 0.29x |
| RandomForestClassifier | 10000 | 4.28x | 103.85x | 0.27x |
| GradientBoostingRegressor | 10000 | 1.03x | 0.07x | 0.01x |
| GradientBoostingClassifier | 10000 | 1.16x | 0.20x | 0.01x |
| HistGradientBoostingRegressor | 10000 | 0.27x | 0.53x | 0.09x |
| HistGradientBoostingClassifier | 10000 | 0.27x | 0.46x | 0.09x |
| KNeighborsClassifier | 10000 | 0.24x | N/A | N/A |
| GaussianNB | 10000 | 3.43x | N/A | N/A |
| StandardScaler | 10000 | 6.36x | N/A | N/A |
| PCA | 10000 | 0.15x | N/A | N/A |
| KMeans | 10000 | 1.19x | N/A | N/A |
