import pandas as pd
import numpy as np

# Load
py_res = pd.read_csv("py_all_results.csv")
r_res = pd.read_csv("data/r_all_results.csv")

py_res['Date'] = pd.to_datetime(py_res['Date'])
r_res['Date'] = pd.to_datetime(r_res['Date'])

# Merge
comp = pd.merge(py_res, r_res, on='Date', suffixes=('_py', '_r'))

# Automatically find columns to compare
py_cols = [c for c in py_res.columns if c != 'Date']
r_cols = [c for c in r_res.columns if c != 'Date']

results = []
missing_in_r = []

for p_col in py_cols:
    # Try exact match or with _py suffix
    r_col = p_col
    if r_col not in r_cols:
        # Check if it was renamed to _r
        if f"{p_col}_r" in comp.columns:
            r_col = f"{p_col}_r"
        else:
            missing_in_r.append(p_col)
            continue
    
    col_py = f"{p_col}_py" if f"{p_col}_py" in comp.columns else p_col
    col_r = f"{r_col}_r" if f"{r_col}_r" in comp.columns else r_col
    
    mask = comp[col_py].notna() & comp[col_r].notna()
    diff = np.abs(comp.loc[mask, col_py] - comp.loc[mask, col_r])
    
    if len(diff) == 0:
        results.append({'Indicator': p_col, 'Max Diff': 'N/A', 'Match%': '0.00%'})
        continue
        
    max_d = diff.max()
    match_pct = (diff < 1e-7).mean() * 100
    
    results.append({
        'Indicator': p_col,
        'Max Diff': f"{max_d:.4e}",
        'Match%': f"{match_pct:.2f}%"
    })
df_report = pd.DataFrame(results)
print("Top 20 Mismatches (Sorted by Max Diff):")
print(df_report[df_report['Match%'] != '100.00%'].sort_values('Max Diff', ascending=False).head(20).to_string(index=False))

print("\nOverall Summary:")
# ...

if len(missing_in_r) > 0:
    print(f"Sample missing: {', '.join(missing_in_r[:10])}...")

# Categorization
perfect = df_report[df_report['Match%'] == '100.00%']['Indicator'].tolist()
print(f"\nPerfectly Aligned ({len(perfect)} indicators)")
