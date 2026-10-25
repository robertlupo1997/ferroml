# Cross-Library Benchmark Results

FerroML 1.0.2 (release wheel built from the 1.0.2 source). Every time and score below is the median of three full runs of `scripts/benchmark_cross_library.py --sizes 10000`, each of which already takes the median of 3 repeats. Single runs moved small timings (under ~30 ms) by up to 2-3x on this 4-core cloud VM, so treat those ratios as rough.

Generated: 2026-10-25T20:03:34.465105
Platform: Linux-6.18.44-fc-v153-x86_64-with-glibc2.39
Python: 3.13.17
NumPy: 2.5.4
sklearn: 1.9.1
xgboost: 3.4.1
lightgbm: 4.7.0

## Summary Table

| Algorithm | Library | Task | N | Features | Fit (ms) | Predict (ms) | Score |
|-----------|---------|------|---|----------|----------|--------------|-------|
| LinearRegression | ferroml | regression | 10000 | 20 | 4.5 | 0.1 | 0.9993 |
| LinearRegression | sklearn | regression | 10000 | 20 | 3.8 | 0.1 | 0.9993 |
| Ridge | ferroml | regression | 10000 | 20 | 2.5 | 0.1 | 0.9993 |
| Ridge | sklearn | regression | 10000 | 20 | 2.1 | 0.2 | 0.9993 |
| Lasso | ferroml | regression | 10000 | 20 | 1.1 | 0.1 | 0.3889 |
| Lasso | sklearn | regression | 10000 | 20 | 1.4 | 0.1 | 0.3889 |
| LogisticRegression | ferroml | classification | 10000 | 20 | 29.2 | 0.2 | 0.8175 |
| LogisticRegression | sklearn | classification | 10000 | 20 | 13.0 | 0.3 | 0.8175 |
| DecisionTreeRegressor | ferroml | regression | 10000 | 20 | 89.4 | 0.3 | 0.5021 |
| DecisionTreeRegressor | sklearn | regression | 10000 | 20 | 110.5 | 0.4 | 0.4972 |
| DecisionTreeClassifier | ferroml | classification | 10000 | 20 | 87.6 | 0.3 | 0.8660 |
| DecisionTreeClassifier | sklearn | classification | 10000 | 20 | 150.8 | 0.4 | 0.8610 |
| RandomForestRegressor | ferroml | regression | 10000 | 20 | 2554.3 | 10.9 | 0.7851 |
| RandomForestRegressor | sklearn | regression | 10000 | 20 | 7032.7 | 28.4 | 0.7863 |
| RandomForestRegressor | xgboost | regression | 10000 | 20 | 2680.9 | 2.9 | 0.7791 |
| RandomForestRegressor | lightgbm | regression | 10000 | 20 | 155.3 | 3.8 | 0.5537 |
| RandomForestClassifier | ferroml | classification | 10000 | 20 | 837.7 | 8.3 | 0.9250 |
| RandomForestClassifier | sklearn | classification | 10000 | 20 | 2036.0 | 23.3 | 0.9300 |
| RandomForestClassifier | xgboost | classification | 10000 | 20 | 925.0 | 2.4 | 0.9225 |
| RandomForestClassifier | lightgbm | classification | 10000 | 20 | 141.7 | 3.5 | 0.8650 |
| GradientBoostingRegressor | ferroml | regression | 10000 | 20 | 5159.6 | 4.6 | 0.9298 |
| GradientBoostingRegressor | sklearn | regression | 10000 | 20 | 7463.5 | 6.2 | 0.9296 |
| GradientBoostingRegressor | xgboost | regression | 10000 | 20 | 145.7 | 1.4 | 0.9301 |
| GradientBoostingRegressor | lightgbm | regression | 10000 | 20 | 101.6 | 4.1 | 0.9336 |
| GradientBoostingClassifier | ferroml | classification | 10000 | 20 | 4906.5 | 4.9 | 0.9060 |
| GradientBoostingClassifier | sklearn | classification | 10000 | 20 | 7576.2 | 5.8 | 0.9285 |
| GradientBoostingClassifier | xgboost | classification | 10000 | 20 | 130.9 | 1.1 | 0.9345 |
| GradientBoostingClassifier | lightgbm | classification | 10000 | 20 | 100.4 | 4.3 | 0.9390 |
| HistGradientBoostingRegressor | ferroml | regression | 10000 | 20 | 437.0 | 6.8 | 0.9301 |
| HistGradientBoostingRegressor | sklearn | regression | 10000 | 20 | 223.9 | 4.2 | 0.9309 |
| HistGradientBoostingRegressor | xgboost | regression | 10000 | 20 | 130.7 | 1.0 | 0.9301 |
| HistGradientBoostingRegressor | lightgbm | regression | 10000 | 20 | 91.5 | 4.1 | 0.9336 |
| HistGradientBoostingClassifier | ferroml | classification | 10000 | 20 | 388.6 | 6.3 | 0.9310 |
| HistGradientBoostingClassifier | sklearn | classification | 10000 | 20 | 217.6 | 3.7 | 0.9350 |
| HistGradientBoostingClassifier | xgboost | classification | 10000 | 20 | 120.2 | 1.4 | 0.9345 |
| HistGradientBoostingClassifier | lightgbm | classification | 10000 | 20 | 88.5 | 4.2 | 0.9390 |
| KNeighborsClassifier | ferroml | classification | 10000 | 20 | 3.1 | 62.8 | 0.9425 |
| KNeighborsClassifier | sklearn | classification | 10000 | 20 | 1.0 | 19.2 | 0.9425 |
| SVC | ferroml | classification | 5000 | 20 | 150.4 | 22.4 | 0.9640 |
| SVC | sklearn | classification | 5000 | 20 | 116.6 | 47.8 | 0.9560 |
| GaussianNB | ferroml | classification | 10000 | 20 | 1.1 | 0.3 | 0.7995 |
| GaussianNB | sklearn | classification | 10000 | 20 | 3.4 | 0.7 | 0.7995 |
| StandardScaler | ferroml | preprocessing | 10000 | 20 | 0.5 | 0.2 | N/A |
| StandardScaler | sklearn | preprocessing | 10000 | 20 | 1.9 | 0.2 | N/A |
| PCA | ferroml | preprocessing | 10000 | 20 | 1.1 | 0.2 | N/A |
| PCA | sklearn | preprocessing | 10000 | 20 | 1.3 | 0.2 | N/A |
| KMeans | ferroml | clustering | 10000 | 20 | 17.3 | 0.2 | 159775.1787 |
| KMeans | sklearn | clustering | 10000 | 20 | 23.6 | 0.4 | 159775.1787 |

## FerroML vs Others: Fit-Time Speedup

Values > 1.0 mean FerroML is faster.

| Algorithm | N | vs sklearn | vs xgboost | vs lightgbm |
|-----------|---|-----------|-----------|------------|
| LinearRegression | 10000 | 0.85x | N/A | N/A |
| Ridge | 10000 | 0.82x | N/A | N/A |
| Lasso | 10000 | 1.33x | N/A | N/A |
| LogisticRegression | 10000 | 0.44x | N/A | N/A |
| DecisionTreeRegressor | 10000 | 1.24x | N/A | N/A |
| DecisionTreeClassifier | 10000 | 1.72x | N/A | N/A |
| RandomForestRegressor | 10000 | 2.75x | 1.05x | 0.06x |
| RandomForestClassifier | 10000 | 2.43x | 1.10x | 0.17x |
| GradientBoostingRegressor | 10000 | 1.45x | 0.03x | 0.02x |
| GradientBoostingClassifier | 10000 | 1.54x | 0.03x | 0.02x |
| HistGradientBoostingRegressor | 10000 | 0.51x | 0.30x | 0.21x |
| HistGradientBoostingClassifier | 10000 | 0.56x | 0.31x | 0.23x |
| KNeighborsClassifier | 10000 | 0.32x | N/A | N/A |
| SVC | 5000 | 0.78x | N/A | N/A |
| GaussianNB | 10000 | 3.06x | N/A | N/A |
| StandardScaler | 10000 | 4.13x | N/A | N/A |
| PCA | 10000 | 1.14x | N/A | N/A |
| KMeans | 10000 | 1.37x | N/A | N/A |

