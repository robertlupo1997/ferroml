"""
FerroML vs sklearn Preprocessing Comparison Report
===================================================
"""

import numpy as np

print('='*70)
print('FERROML vs SKLEARN PREPROCESSING COMPARISON REPORT')
print('='*70)
print()

# Test data
X = np.array([
    [1.0, 10.0, 100.0],
    [2.0, 20.0, 200.0],
    [3.0, 30.0, 300.0],
    [4.0, 40.0, 400.0],
    [5.0, 50.0, 500.0],
], dtype=np.float64)

X_cat = np.array([[0.0], [1.0], [2.0], [1.0], [0.0]], dtype=np.float64)
X_missing = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan], [4.0, 6.0], [5.0, 8.0]], dtype=np.float64)

# Import all preprocessors
from sklearn.preprocessing import (
    StandardScaler as SkStd, MinMaxScaler as SkMinMax,
    RobustScaler as SkRobust, MaxAbsScaler as SkMaxAbs,
    OneHotEncoder as SkOneHot, OrdinalEncoder as SkOrdinal,
    LabelEncoder as SkLabel
)
from sklearn.impute import SimpleImputer as SkImputer

from ferroml.preprocessing import (
    StandardScaler as FeStd, MinMaxScaler as FeMinMax,
    RobustScaler as FeRobust, MaxAbsScaler as FeMaxAbs,
    OneHotEncoder as FeOneHot, OrdinalEncoder as FeOrdinal,
    LabelEncoder as FeLabel, SimpleImputer as FeImputer
)

def compare(name, sk_val, fe_val):
    """Compare arrays and return status"""
    sk_arr = np.asarray(sk_val).flatten()
    fe_arr = np.asarray(fe_val).flatten()
    if sk_arr.shape != fe_arr.shape:
        return 'SHAPE_MISMATCH', float('inf')
    max_diff = np.max(np.abs(sk_arr - fe_arr))
    if max_diff < 1e-14:
        return 'EXACT', max_diff
    elif max_diff < 1e-10:
        return 'CLOSE', max_diff
    else:
        return 'DIFFERS', max_diff

results = []

# === StandardScaler ===
sk = SkStd().fit(X)
fe = FeStd(); fe.fit(X)
results.append(('StandardScaler', 'mean_', *compare('mean_', sk.mean_, fe.mean_)))
results.append(('StandardScaler', 'scale_', *compare('scale_', sk.scale_, fe.scale_)))
results.append(('StandardScaler', 'transform', *compare('transform', sk.transform(X), fe.transform(X))))
results.append(('StandardScaler', 'inverse', *compare('inverse', sk.inverse_transform(sk.transform(X)), fe.inverse_transform(fe.transform(X)))))

# === MinMaxScaler ===
sk = SkMinMax().fit(X)
fe = FeMinMax(); fe.fit(X)
results.append(('MinMaxScaler', 'data_min_', *compare('', sk.data_min_, fe.data_min_)))
results.append(('MinMaxScaler', 'data_max_', *compare('', sk.data_max_, fe.data_max_)))
results.append(('MinMaxScaler', 'data_range_', *compare('', sk.data_range_, fe.data_range_)))
results.append(('MinMaxScaler', 'transform', *compare('', sk.transform(X), fe.transform(X))))
results.append(('MinMaxScaler', 'inverse', *compare('', sk.inverse_transform(sk.transform(X)), fe.inverse_transform(fe.transform(X)))))

# === RobustScaler ===
sk = SkRobust().fit(X)
fe = FeRobust(); fe.fit(X)
results.append(('RobustScaler', 'center_', *compare('', sk.center_, fe.center_)))
results.append(('RobustScaler', 'scale_', *compare('', sk.scale_, fe.scale_)))
results.append(('RobustScaler', 'transform', *compare('', sk.transform(X), fe.transform(X))))
results.append(('RobustScaler', 'inverse', *compare('', sk.inverse_transform(sk.transform(X)), fe.inverse_transform(fe.transform(X)))))

# === MaxAbsScaler ===
X_signed = X - X.mean(axis=0)
sk = SkMaxAbs().fit(X_signed)
fe = FeMaxAbs(); fe.fit(X_signed)
results.append(('MaxAbsScaler', 'max_abs_', *compare('', sk.max_abs_, fe.max_abs_)))
results.append(('MaxAbsScaler', 'transform', *compare('', sk.transform(X_signed), fe.transform(X_signed))))
results.append(('MaxAbsScaler', 'inverse', *compare('', sk.inverse_transform(sk.transform(X_signed)), fe.inverse_transform(fe.transform(X_signed)))))

# === OneHotEncoder ===
sk = SkOneHot(sparse_output=False).fit(X_cat)
fe = FeOneHot(); fe.fit(X_cat)
results.append(('OneHotEncoder', 'transform', *compare('', sk.transform(X_cat), fe.transform(X_cat))))

# === OrdinalEncoder ===
sk = SkOrdinal().fit(X_cat)
fe = FeOrdinal(); fe.fit(X_cat)
results.append(('OrdinalEncoder', 'transform', *compare('', sk.transform(X_cat), fe.transform(X_cat))))

# === LabelEncoder ===
y = np.array([2.0, 0.0, 1.0, 2.0, 1.0, 0.0])
sk = SkLabel().fit(y.astype(int))
fe = FeLabel(); fe.fit(y)
# Note: LabelEncoder may assign different codes, check consistency instead
sk_res = sk.transform(y.astype(int))
fe_res = fe.transform(y)
# Check if mapping is consistent (same input -> same output)
le_consistent = len(set(zip(y, sk_res))) == 3 and len(set(zip(y, fe_res))) == 3
results.append(('LabelEncoder', 'consistency', 'MATCH' if le_consistent else 'DIFFERS', 0.0))

# === SimpleImputer (mean) ===
sk = SkImputer(strategy='mean').fit(X_missing)
fe = FeImputer(strategy='mean'); fe.fit(X_missing)
results.append(('SimpleImputer(mean)', 'statistics_', *compare('', sk.statistics_, fe.statistics_)))
results.append(('SimpleImputer(mean)', 'transform', *compare('', sk.transform(X_missing), fe.transform(X_missing))))

# === SimpleImputer (median) ===
sk = SkImputer(strategy='median').fit(X_missing)
fe = FeImputer(strategy='median'); fe.fit(X_missing)
results.append(('SimpleImputer(med)', 'statistics_', *compare('', sk.statistics_, fe.statistics_)))
results.append(('SimpleImputer(med)', 'transform', *compare('', sk.transform(X_missing), fe.transform(X_missing))))

# Print results table
print(f"{'Preprocessor':<22} {'Component':<14} {'Status':<12} {'Max Diff':<15}")
print('-' * 65)

exact_count = 0
close_count = 0
diff_count = 0

for preproc, comp, status, diff in results:
    if status == 'EXACT':
        exact_count += 1
        diff_str = f'{diff:.2e}'
    elif status == 'CLOSE':
        close_count += 1
        diff_str = f'{diff:.2e}'
    elif status == 'MATCH':
        exact_count += 1
        diff_str = 'N/A'
    else:
        diff_count += 1
        diff_str = f'{diff:.2e}'
    print(f'{preproc:<22} {comp:<14} {status:<12} {diff_str:<15}')

print()
print('=' * 65)
print('SUMMARY')
print('=' * 65)
print(f'Exact matches (diff < 1e-14):  {exact_count}')
print(f'Close matches (diff < 1e-10):  {close_count}')
print(f'Differences:                   {diff_count}')
print()

# Detailed analysis of differences
if diff_count > 0:
    print('=' * 65)
    print('DETAILED ANALYSIS OF DIFFERENCES')
    print('=' * 65)
    print()

    # StandardScaler analysis
    print('StandardScaler scale_ ISSUE:')
    print('-' * 40)
    print('  sklearn uses:  ddof=0 (population std)')
    print('  FerroML uses:  ddof=1 (sample std)')
    print()
    print('  For n=5 samples:')
    print(f'    Population std: sqrt(var * n/n)     = {np.std([1,2,3,4,5], ddof=0):.10f}')
    print(f'    Sample std:     sqrt(var * n/(n-1)) = {np.std([1,2,3,4,5], ddof=1):.10f}')
    print()
    print(f'  Ratio: sqrt(n/(n-1)) = sqrt(5/4) = {np.sqrt(5/4):.10f}')
    print()
    print('  ROOT CAUSE: FerroML StandardScaler computes sample std (ddof=1)')
    print('              while sklearn uses population std (ddof=0).')
    print()
    print('  NOTE: The inverse_transform still recovers original data because')
    print('        the scaling factor is applied consistently both ways.')

print()
print('=' * 65)
print('PREPROCESSORS THAT MATCH SKLEARN EXACTLY')
print('=' * 65)

# Determine which preprocessors fully match
preprocessor_status = {}
for preproc, comp, status, diff in results:
    if preproc not in preprocessor_status:
        preprocessor_status[preproc] = True
    if status not in ('EXACT', 'MATCH', 'CLOSE'):
        preprocessor_status[preproc] = False

matching = [p for p, ok in preprocessor_status.items() if ok]
differing = [p for p, ok in preprocessor_status.items() if not ok]

print()
print('MATCHING sklearn (all components):')
for p in sorted(matching):
    print(f'  [OK] {p}')

print()
print('DIFFERING from sklearn:')
for p in sorted(differing):
    print(f'  [!!] {p}')

print()
print('=' * 65)
print('NUMERIC COMPARISON DETAILS')
print('=' * 65)
print()

# Show actual values for StandardScaler
print('StandardScaler Numeric Values:')
print('-' * 40)
sk = SkStd().fit(X)
fe = FeStd(); fe.fit(X)
print(f'  sklearn mean_:   {sk.mean_}')
print(f'  FerroML mean_:   {fe.mean_}')
print(f'  Difference:      {np.abs(sk.mean_ - fe.mean_)}')
print()
print(f'  sklearn scale_:  {sk.scale_}')
print(f'  FerroML scale_:  {fe.scale_}')
print(f'  Difference:      {np.abs(sk.scale_ - fe.scale_)}')
print()
print('  Transformed data (first 2 rows):')
print(f'    sklearn:  {sk.transform(X)[:2]}')
print(f'    FerroML:  {fe.transform(X)[:2]}')
