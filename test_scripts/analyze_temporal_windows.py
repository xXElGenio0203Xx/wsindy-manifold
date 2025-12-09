"""
Analyze temporal window sizes and compare to R² degradation timeline.
"""

print("="*80)
print("TEMPORAL WINDOW ANALYSIS")
print("="*80)

# Configuration from best_run_extended_test.yaml
dt = 0.1  # seconds per timestep
subsample = 1  # no subsampling
mvar_lag = 20  # lag order

# Training
T_train = 2.0  # seconds
steps_per_train_run = int(T_train / dt)  # 20 steps

# Testing
T_test = 10.0  # seconds
steps_per_test_run = int(T_test / dt)  # 100 steps

print("\n" + "-"*80)
print("CONFIGURATION")
print("-"*80)
print(f"dt = {dt}s (timestep)")
print(f"subsample = {subsample} (no temporal downsampling)")
print(f"mvar_lag = {mvar_lag} (MVAR order)")

print("\n" + "-"*80)
print("TRAINING DATA")
print("-"*80)
print(f"Duration: T_train = {T_train}s")
print(f"Timesteps per run: {steps_per_train_run}")
print(f"Effective dt: {dt * subsample}s (after subsampling)")

# MVAR window calculation
# MVAR needs (lag + 1) consecutive snapshots to form one training sample
# For lag=20: [t-20, t-19, ..., t-1, t] → predicts t+1
samples_per_train_run = steps_per_train_run - mvar_lag
window_duration = mvar_lag * dt * subsample

print(f"\n📊 MVAR Training Windows:")
print(f"   Window size: {mvar_lag} lags × {dt}s = {window_duration}s")
print(f"   Samples per training run: {samples_per_train_run}")
print(f"   (Need {mvar_lag+1} snapshots to make 1 prediction)")

if samples_per_train_run <= 0:
    print(f"\n⚠️  WARNING: No training samples! lag={mvar_lag} ≥ steps={steps_per_train_run}")
else:
    print(f"\n✓ Each training run provides {samples_per_train_run} MVAR training samples")
    print(f"  First sample uses: t=0 to t={window_duration}s → predicts t={window_duration + dt}s")
    print(f"  Last sample uses: t={(samples_per_train_run-1)*dt}s to t={T_train - dt}s → predicts t={T_train}s")

print("\n" + "-"*80)
print("TESTING / PREDICTION")
print("-"*80)
print(f"Duration: T_test = {T_test}s")
print(f"Timesteps per test run: {steps_per_test_run}")
print(f"Evaluation starts at: t={T_train}s (after training period)")
print(f"Evaluation duration: {T_test - T_train}s ({int((T_test - T_train)/dt)} prediction steps)")

print("\n📊 Autoregressive Prediction:")
print(f"   To predict t={T_train + dt}s:")
print(f"      Need history: [{T_train - window_duration}s, {T_train}s]")
print(f"      Window: {window_duration}s of past data")
print(f"   ")
print(f"   To predict t={T_train + 2*dt}s:")
print(f"      Use: [{T_train - window_duration + dt}s, {T_train + dt}s]")
print(f"      (Includes 1 predicted value)")
print(f"   ")
print(f"   To predict t=T_train + k*dt (general):")
print(f"      Window includes k-1 predicted values (if k > lag)")

# Critical analysis
print("\n" + "="*80)
print("CRITICAL ANALYSIS: Prediction vs Training Window Size")
print("="*80)

print(f"\n🔍 Training window duration: {window_duration}s")
print(f"   The model learns dynamics from {window_duration}s time windows")

# Compare to degradation timeline from time_resolved_analysis
degradation_times = {
    "Starts decent": 2.0,
    "Below R²=0.8": 2.0,
    "Below R²=0.5": 3.1,
    "Below R²=0.3": 3.6,
    "Below R²=0.0 (useless)": 4.4,
    "Complete failure": 10.0
}

print(f"\n📉 R² Degradation Timeline (from analysis):")
for label, t in degradation_times.items():
    time_beyond_train = t - T_train
    ratio_to_window = time_beyond_train / window_duration if window_duration > 0 else float('inf')
    print(f"   {label:25s}: t={t:4.1f}s  (+{time_beyond_train:.1f}s beyond training)")
    print(f"                                    = {ratio_to_window:.2f}× training window duration")

print("\n" + "="*80)
print("HYPOTHESIS TESTING")
print("="*80)

print(f"\n💡 Key Insight:")
print(f"   Training window = {window_duration}s = {mvar_lag} timesteps")
print(f"   Model maintains R² > 0.5 for ~{3.1 - T_train:.1f}s beyond training")
print(f"   Ratio: {(3.1 - T_train) / window_duration:.2f}× the training window")

print(f"\n🎯 Interpretation:")
print(f"   1. Model learns patterns that repeat every ~{window_duration}s")
print(f"   2. Can extrapolate for ~{(3.1 - T_train) / window_duration:.1f}× that window ({3.1 - T_train:.1f}s)")
print(f"   3. After {4.4 - T_train:.1f}s ({(4.4 - T_train) / window_duration:.2f}× window), predictions become worse than baseline")

print(f"\n⚠️  Problem:")
print(f"   The MVAR model sees correlations within {window_duration}s windows")
print(f"   When predicting {4.4 - T_train:.1f}s ahead:")
print(f"      • Uses a {window_duration}s history window")
print(f"      • But that history includes {min(mvar_lag, int((4.4 - T_train)/dt))} predicted (not true) values")
print(f"      • Prediction error accumulates → cascade failure")

# Autoregressive contamination
print(f"\n" + "="*80)
print("AUTOREGRESSIVE ERROR ACCUMULATION")
print("="*80)

for t_pred in [2.1, 3.0, 4.0, 5.0, 10.0]:
    steps_ahead = int((t_pred - T_train) / dt)
    if steps_ahead <= 0:
        continue
    
    # At time t_pred, how many predicted values are in the lag window?
    predicted_in_window = min(steps_ahead, mvar_lag)
    true_in_window = max(0, mvar_lag - steps_ahead)
    fraction_predicted = predicted_in_window / mvar_lag
    
    print(f"\nPredicting t={t_pred:.1f}s ({steps_ahead} steps beyond training):")
    print(f"   Lag window contains:")
    print(f"      • {true_in_window}/{mvar_lag} true values ({true_in_window/mvar_lag*100:.0f}%)")
    print(f"      • {predicted_in_window}/{mvar_lag} predicted values ({fraction_predicted*100:.0f}%)")
    print(f"   Window purity: {true_in_window/mvar_lag*100:.0f}% ground truth")
    
    if steps_ahead >= mvar_lag:
        print(f"   ⚠️  FULLY CONTAMINATED: Entire window is predicted data!")

print(f"\n" + "="*80)
print("CONCLUSION")
print("="*80)

print(f"""
The model's {window_duration}s training window explains the degradation pattern:

1. ✅ t=2.0s → 3.1s: Performance acceptable (R² > 0.5)
   • Predicting 0 → {(3.1-T_train)/dt:.0f} steps ahead
   • Lag window still has {(1 - (3.1-T_train)/window_duration)*100:.0f}% → {max(0, (1 - (3.1-T_train)/window_duration))*100:.0f}% ground truth

2. ⚠️  t=3.1s → 4.4s: Rapid degradation (R² = 0.5 → 0.0)
   • Predicting {(3.1-T_train)/dt:.0f} → {(4.4-T_train)/dt:.0f} steps ahead
   • Lag window becomes {(1 - min(1, (4.4-T_train)/window_duration))*100:.0f}% → 0% ground truth

3. ❌ t > 4.4s: Complete failure (R² < 0)
   • Predicting {(4.4-T_train)/dt:.0f}+ steps ahead
   • Entire {window_duration}s lag window is predicted (contaminated) data
   • Model compounds its own errors → cascade failure

RECOMMENDATION:
To improve long-term prediction, you need EITHER:
  A) Longer training trajectories (T_train >> {window_duration}s) to see more dynamics
  B) Shorter lag window (lag << {mvar_lag}) to reduce error accumulation  
  C) Both: Train on longer sequences with smaller lag
""")

print("="*80)
