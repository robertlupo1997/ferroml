# Cross-Library Benchmark Results

Generated: 2026-03-24T22:34:50.283797
Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
Python: 3.12.3
NumPy: 2.4.2
sklearn: 1.8.0
xgboost: 3.2.0
lightgbm: 4.6.0

## Summary Table

| Algorithm | Library | Task | N | Features | Fit (ms) | Predict (ms) | Score |
|-----------|---------|------|---|----------|----------|--------------|-------|
| KMeans | ferroml | clustering | 1000 | 20 | 1.2 | 0.0 | 16074.9071 |
| KMeans | sklearn | clustering | 1000 | 20 | 13.4 | 0.2 | 16074.9071 |
| KMeans | ferroml | clustering | 5000 | 20 | 8.4 | 0.1 | 80200.9181 |
| KMeans | sklearn | clustering | 5000 | 20 | 19.7 | 0.3 | 80200.9181 |
| KMeans | ferroml | clustering | 10000 | 20 | 8.4 | 0.1 | 159775.1787 |
| KMeans | sklearn | clustering | 10000 | 20 | 24.0 | 0.5 | 159775.1787 |

## FerroML vs Others: Fit-Time Speedup

Values > 1.0 mean FerroML is faster.

| Algorithm | N | vs sklearn | vs xgboost | vs lightgbm |
|-----------|---|-----------|-----------|------------|
| KMeans | 1000 | 10.98x | N/A | N/A |
| KMeans | 5000 | 2.35x | N/A | N/A |
| KMeans | 10000 | 2.84x | N/A | N/A |
