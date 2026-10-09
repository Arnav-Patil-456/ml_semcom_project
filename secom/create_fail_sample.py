import os
import joblib
import pandas as pd
import numpy as np

# Load the 562 active feature names and test partition
artifacts_dir = "artifacts"
feature_names = joblib.load(os.path.join(artifacts_dir, "feature_names.joblib"))
test_samples = pd.read_csv(os.path.join(artifacts_dir, "test_samples.csv"))

# Find an actual defective run (Actual_Fail == 1) from holdout samples
fail_rows = test_samples[test_samples["Actual_Fail"] == 1]

if len(fail_rows) > 0:
    # Use real ground-truth failure sensor readings
    fail_sample = fail_rows.drop(columns=["Actual_Fail"]).iloc[[0]].copy()
else:
    # Fallback: create an extreme outlier vector across top features
    fail_sample = test_samples.drop(columns=["Actual_Fail"]).iloc[[0]].copy()
    fail_sample.iloc[0, :20] = fail_sample.iloc[0, :20] * 5.0

# Save as uploadable CSV
output_path = "failing_wafer_sample.csv"
fail_sample[feature_names].to_csv(output_path, index=False)
print(f"[✓] Created '{output_path}' with {len(feature_names)} valid columns!")