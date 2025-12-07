# Making ROM-MVAR Robust: Comprehensive Guide

## Current Problems Identified

### 1. **Unstable Dynamics (Critical)**
- **Issue**: MVAR has eigenvalues λ = 1.276 > 1
- **Effect**: Predictions diverge exponentially: (1.276)^100 ≈ 10^10
- **Result**: Test R² = -0.018 (worse than mean baseline)

### 2. **Non-Translation-Invariant POD**
- **Issue**: POD trained on Gaussian blobs at (15,15) only
- **Effect**: Different spatial positions project to wrong latent coordinates
- **Result**: Cannot generalize to translated initial conditions

### 3. **Distribution-Specific POD**
- **Issue**: POD trained only on Gaussian distributions
- **Effect**: Fails on uniform, ring, or other distributions
- **Result**: Cannot generalize beyond training distribution family

---

## Solution 1: Stabilize MVAR (CRITICAL - Do First)

### Problem
Ridge regularization α = 1e-6 is too weak → overfitting → unstable eigenvalues

### Solutions

#### A) Increase Ridge Regularization ⭐ **EASIEST**
```yaml
rom:
  ridge_alpha: 0.01  # Increase from 1e-6 (10,000× stronger)
```
- **Pros**: One line change, guaranteed stability
- **Cons**: May reduce training R² slightly
- **Expected**: Max eigenvalue < 1, stable predictions

#### B) Eigenvalue Constraints
Add constraint during optimization:
```python
# After training, rescale if needed
max_eig = max(np.abs(np.linalg.eigvals(A[i])) for i in range(p))
if max_eig > 0.95:
    scale_factor = 0.95 / max_eig
    A_matrices = scale_factor * A_matrices
```

#### C) Use Constrained Optimization
Replace Ridge regression with:
- Projected gradient descent with spectral constraints
- Alternating Direction Method of Multipliers (ADMM)

---

## Solution 2: Translation-Equivariant POD

### Problem
POD learns "Gaussian blob at (15,15)" not "Gaussian blob anywhere"

### Solutions

#### A) Translation Augmentation ⭐ **RECOMMENDED**
```yaml
train_ic:
  type: "gaussian_spatial_grid"
  # Train on grid of positions
  positions_x: [7.5, 11.25, 15.0, 18.75, 22.5]  # 5 positions
  positions_y: [7.5, 11.25, 15.0, 18.75, 22.5]  # 5 positions
  variances: [1.0, 2.0, 3.0, 4.0, 5.0]          # 5 variances
  n_samples_per_config: 2
  # Total: 5×5 spatial × 5 variances × 2 samples = 250 runs
```

**Pros**: 
- POD learns position-independent features
- Generalizes to any spatial translation
- Easy to implement

**Cons**: 
- More training data needed (250 vs 200 runs)
- Training time ~3-4 hours on Oscar

#### B) Shift to Centered Coordinates
```python
# Before POD projection
def center_density_field(rho):
    """Shift field so center of mass is at origin"""
    y, x = np.indices(rho.shape)
    cx = np.sum(x * rho) / np.sum(rho)
    cy = np.sum(y * rho) / np.sum(rho)
    # Circular shift to center
    shift_x = int(nx/2 - cx)
    shift_y = int(ny/2 - cy)
    rho_centered = np.roll(np.roll(rho, shift_x, axis=1), shift_y, axis=0)
    return rho_centered, (cx, cy)
```

**Pros**: No extra training data needed
**Cons**: More complex prediction pipeline (need to shift back)

#### C) Convolutional Autoencoder (Advanced)
Replace POD with CNN that is naturally translation-equivariant:
```python
# CNN Encoder: (64, 64) → latent
# LSTM/GRU: latent dynamics
# CNN Decoder: latent → (64, 64)
```

**Pros**: Best generalization, learned features
**Cons**: Needs more data, harder to interpret, longer training

---

## Solution 3: Distribution-Robust POD

### Problem
POD trained on Gaussians doesn't understand uniform/ring patterns

### Solutions

#### A) Train on Mixed Distributions ⭐ **RECOMMENDED**
```yaml
train_ic:
  type: "mixed_distributions"
  gaussian_runs: 100
  uniform_runs: 50
  ring_runs: 50
  # Each with varied parameters
```

**Pros**: POD learns general "crowd density" features
**Cons**: Need diverse training data

#### B) Physics-Informed Features
Instead of raw density ρ(x,y), use:
- Density moments: ∫ρ, ∫xρ, ∫x²ρ
- Gradient fields: ∇ρ
- Fourier modes (universal for any distribution)

---

## Solution 4: Nonlinear MVAR

### Problem
Linear MVAR too simple for nonlinear crowd dynamics

### Solutions

#### A) Add Quadratic Terms (SINDy) ⭐ **GOOD BALANCE**
```python
# Linear: x[t] = Σ A[i]·x[t-i]
# Quadratic: x[t] = Σ A[i]·x[t-i] + Σ B[ij]·(x[t-1]⊗x[t-1])

from sklearn.linear_model import Lasso
# Use sparse regression to select important nonlinear terms
```

**Pros**: More expressive, still interpretable
**Cons**: More parameters (need regularization)

#### B) Neural Network MVAR
```python
class NonlinearMVAR(nn.Module):
    def forward(self, x_history):
        # x_history: (p, r) - past p latent states
        h = torch.cat(x_history, dim=-1)  # Concatenate
        h = self.mlp(h)  # MLP: (p*r) → r
        return h
```

**Pros**: Maximum flexibility
**Cons**: Black box, needs lots of data

---

## Practical Implementation Plan

### Phase 1: Quick Stability Fix (1 day)
**Goal**: Get stable predictions

1. **Increase ridge regularization**
   - Change `ridge_alpha: 1e-6` → `0.01` in config
   - Retrain MVAR on existing 200 Gaussian runs
   - Test on same 48 positions
   
**Expected**: Stable predictions, R² should be 0.2-0.5 instead of -0.018

### Phase 2: Spatial Generalization (2-3 days)
**Goal**: Generalize to different positions

2. **Add spatial grid training data**
   ```yaml
   train_ic:
     positions: 4×4 grid (16 positions)
     variances: 5 values
     samples: 2 per config
     Total: 160 runs
   ```
   
3. **Train POD+MVAR on spatially diverse data**
   - Run on Oscar: ~3 hours
   - POD will learn position-independent features
   
4. **Test on held-out spatial positions**

**Expected**: R² > 0.7 for spatial translations

### Phase 3: Distribution Diversity (3-5 days)
**Goal**: Generalize to different IC types

5. **Generate mixed training data**
   - 100 Gaussian (varied positions + variances)
   - 50 Uniform
   - 50 Ring (varied radii)
   
6. **Train POD+MVAR on diverse distributions**

**Expected**: R² > 0.5 for uniform/ring ICs

### Phase 4: Nonlinear Dynamics (1-2 weeks)
**Goal**: Capture nonlinear effects

7. **Implement SINDy-MVAR**
   - Add quadratic terms with sparsity
   - Use cross-validation for regularization
   
**Expected**: R² > 0.8, better long-term predictions

---

## Recommended Next Experiment

### Experiment: `robust_mvar_spatial_v1`

#### Configuration
```yaml
# configs/robust_mvar_spatial_v1.yaml

sim:
  N: 300
  T: 100.0
  # ... same as gaussians_oscar

train_ic:
  type: "gaussian_spatial_grid"
  
  # 4×4 spatial grid
  positions_x: [7.5, 12.5, 17.5, 22.5]
  positions_y: [7.5, 12.5, 17.5, 22.5]
  
  # 5 variances per position
  variances: [1.0, 1.5, 2.0, 3.0, 4.0]
  
  # 2 samples each
  n_samples_per_config: 2
  
  # Total: 4×4×5×2 = 160 runs

rom:
  pod_energy: 0.995
  mvar_lag: 4
  ridge_alpha: 0.01  # ← INCREASED FROM 1e-6

test_ic:
  # Test on held-out positions (interpolation)
  positions_x: [10.0, 15.0, 20.0]
  positions_y: [10.0, 15.0, 20.0]
  variances: [2.0]
  # Total: 3×3 = 9 test runs
```

#### Expected Runtime
- Training simulations: 160 runs × 2 min = 5.3 hours
- POD+MVAR: ~5 minutes
- Test simulations: 9 runs × 2 min = 18 minutes
- Predictions: ~30 seconds
- **Total: ~6 hours on Oscar**

#### Expected Results
- ✅ Stable predictions (all eigenvalues < 1)
- ✅ Spatial interpolation R² > 0.7
- ✅ Works for any position in domain
- ❌ Still fails on uniform/ring (needs Phase 3)

---

## Key Takeaways

### What Worked
- POD captures variance dynamics well (training R² = 0.999)
- MVAR can learn temporal patterns when properly regularized

### What Failed
1. **Weak regularization** → unstable dynamics
2. **Single-position training** → no spatial generalization
3. **Single-distribution training** → no distribution generalization

### What to Do Next
1. **Immediate**: Increase ridge_alpha to 0.01
2. **Short-term**: Train on 4×4 spatial grid (160 runs)
3. **Medium-term**: Add mixed distributions (Gaussian + uniform + ring)
4. **Long-term**: Consider nonlinear MVAR or deep learning

### Bottom Line
Current ROM-MVAR is **scientifically interesting** (demonstrates failure modes) but **not practically useful**. With proper regularization and diverse training data, it can become **robust and generalizable**.
