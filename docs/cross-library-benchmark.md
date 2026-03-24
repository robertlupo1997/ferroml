# Cross-Library Benchmark Results

Generated: 2026-03-24T12:23:56.301042
Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
Python: 3.12.3
NumPy: 2.4.2
sklearn: 1.8.0
xgboost: 3.2.0
lightgbm: 4.6.0

## Summary Table

| Algorithm | Library | Task | N | Features | Fit (ms) | Predict (ms) | Score |
|-----------|---------|------|---|----------|----------|--------------|-------|
| KMeans | ferroml | clustering | 5000 | 20 | 32.9 | 0.1 | 80200.9181 |
| KMeans | sklearn | clustering | 5000 | 20 | 16.3 | 0.3 | 80200.9181 |

## FerroML vs Others: Fit-Time Speedup

Values > 1.0 mean FerroML is faster.

| Algorithm | N | vs sklearn | vs xgboost | vs lightgbm |
|-----------|---|-----------|-----------|------------|
| KMeans | 5000 | 0.50x | N/A | N/A |
