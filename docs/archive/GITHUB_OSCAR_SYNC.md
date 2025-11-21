# GitHub → Oscar Sync Workflow

## Quick Reference

### On your Mac (Local Development)
```bash
# 1. Make code changes in VS Code
# 2. Commit and push to GitHub
git add .
git commit -m "Description of changes"
git push origin main
```

### On Oscar (via VS Code Remote-SSH)
```bash
# Option A: Full sync with test (recommended)
./sync_and_test.sh

# Option B: Quick update (minimal)
./quick_update.sh

# Option C: Manual steps (see below)
```

---

## Detailed Workflow

### 1️⃣ Connect to Oscar from VS Code

**From your Mac:**
1. Open VS Code
2. Press `Cmd+Shift+P`
3. Select "Remote-SSH: Connect to Host..."
4. Choose `oscar`
5. Approve Duo 2FA
6. Open folder: `/users/emaciaso/src/wsindy-manifold`

---

### 2️⃣ Pull Latest Code

**Option A: Use the sync script (Recommended)**
```bash
./sync_and_test.sh
```

This will:
- Pull latest changes from GitHub
- Activate conda environment
- Verify package installation
- Run a quick test simulation
- Show you output location

**Option B: Manual git pull**
```bash
cd /users/emaciaso/src/wsindy-manifold
git status                    # Check current state
git pull origin main          # Pull latest changes
```

**View what changed:**
```bash
git log -3 --oneline          # See last 3 commits
git diff HEAD~1               # See changes from previous commit
```

---

### 3️⃣ Do You Need to Reinstall?

**Short answer: Usually NO** ✅

Since you installed with `pip install -e .` (editable mode), Python automatically uses your updated source files.

**When you DO need to reinstall:**
- ❗ Changed `pyproject.toml` (added/changed dependencies, entry points, etc.)
- ❗ Added new package dependencies in `requirements.txt`
- ❗ Changed package metadata or structure

**How to reinstall:**
```bash
source setup_oscar_env.sh
pip install -r requirements.txt  # Update dependencies
pip install -e .                 # Reinstall package
```

Or use the script:
```bash
./sync_and_test.sh --reinstall
```

**To verify installation:**
```bash
source setup_oscar_env.sh
python -c "import rectsim; print(rectsim.__file__)"
rectsim --help
```

---

### 4️⃣ Test Your Changes

**Quick interactive test:**
```bash
source setup_oscar_env.sh
rectsim single \
  --config configs/gentle_clustering.yaml \
  --sim.N 50 \
  --sim.T 2.0 \
  --outputs.animate false
```

**Check outputs:**
```bash
ls -lh outputs/single/
```

**Test with different config:**
```bash
./sync_and_test.sh --config configs/long_loose_N200_T400.yaml
```

---

### 5️⃣ Submit Production Job

Once tests pass, submit a full SLURM job:

```bash
# Edit parameters in run_rectsim_single.slurm if needed
sbatch run_rectsim_single.slurm

# Monitor job
squeue -u emaciaso
tail -f slurm_logs/rectsim_<JOB_ID>.out
```

---

## Common Scenarios

### Scenario 1: Fixed a bug in Python code
```bash
# On Mac: commit and push
git add src/rectsim/vicsek_discrete.py
git commit -m "Fix velocity update bug"
git push origin main

# On Oscar: pull and test
./sync_and_test.sh
# ✓ No reinstall needed - editable install uses new code
```

### Scenario 2: Added new dependency
```bash
# On Mac: update requirements.txt, commit, push
echo "scikit-learn==1.3.0" >> requirements.txt
git add requirements.txt
git commit -m "Add scikit-learn dependency"
git push origin main

# On Oscar: pull and reinstall
./sync_and_test.sh --reinstall
# ✓ Reinstall needed to install new dependency
```

### Scenario 3: Changed config file
```bash
# On Mac: edit config, commit, push
# (edit configs/gentle_clustering.yaml)
git add configs/gentle_clustering.yaml
git commit -m "Update clustering parameters"
git push origin main

# On Oscar: pull and test with new config
./sync_and_test.sh --config configs/gentle_clustering.yaml
# ✓ No reinstall needed - just using new config
```

### Scenario 4: Added new CLI command
```bash
# On Mac: modify pyproject.toml [project.scripts], commit, push
# (add new entry point in pyproject.toml)
git add pyproject.toml
git commit -m "Add new CLI command"
git push origin main

# On Oscar: pull and reinstall
./sync_and_test.sh --reinstall
# ✓ Reinstall needed to register new entry point
```

---

## Troubleshooting

### "Already up to date" but files haven't changed
```bash
# Check if you're on the right branch
git branch --show-current

# Force fetch and reset to remote
git fetch origin
git reset --hard origin/main
```

### Import errors after pulling
```bash
# Reinstall package and dependencies
source setup_oscar_env.sh
pip install -r requirements.txt
pip install -e .
```

### "rectsim: command not found"
```bash
# Make sure environment is activated
source setup_oscar_env.sh
which python    # Should show wsindy env path
which rectsim   # Should show wsindy env path
```

### Changes not taking effect
```bash
# 1. Verify you pulled the changes
git log -1 --oneline

# 2. Check the file on Oscar matches GitHub
cat src/rectsim/vicsek_discrete.py | head -20

# 3. Restart Python (if running in interactive session)
# 4. For persistent issues, reinstall
./sync_and_test.sh --reinstall
```

---

## Best Practices

### ✅ DO
- Always pull before starting work on Oscar
- Test interactively with short runs before submitting jobs
- Use `./sync_and_test.sh` for routine updates
- Check git log to see what changed
- Use `--reinstall` when you change dependencies or pyproject.toml

### ❌ DON'T
- Edit code directly on Oscar (edit on Mac, push to GitHub)
- Skip testing after pulling updates
- Forget to activate conda environment
- Submit jobs without verifying changes work

---

## Typical Daily Workflow

**Morning (on Mac):**
```bash
# Work on code in VS Code
# Make changes, test locally if possible
git add .
git commit -m "Implement new feature"
git push origin main
```

**Afternoon (on Oscar via Remote-SSH):**
```bash
# Connect to Oscar in VS Code
# Open terminal (Ctrl+`)

# Sync and test
./sync_and_test.sh

# If tests pass, run production simulations
sbatch run_rectsim_single.slurm

# Monitor
squeue -u emaciaso
```

**Evening (check results):**
```bash
# View outputs
ls -lh outputs/single/
ls -lh slurm_logs/

# Download results to Mac if needed (from Mac terminal)
scp -r oscar:/users/emaciaso/src/wsindy-manifold/outputs/single/ ~/Desktop/
```

---

## Scripts Summary

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `sync_and_test.sh` | Full sync with test | After pushing code changes |
| `sync_and_test.sh --reinstall` | Sync + reinstall | Changed dependencies or pyproject.toml |
| `sync_and_test.sh --config <path>` | Test specific config | Testing new config file |
| `quick_update.sh` | Minimal sync | Quick updates, simple changes |
| `setup_oscar_env.sh` | Activate environment only | Just need conda env |
| `run_rectsim_single.slurm` | Submit SLURM job | Production runs |

---

## Next Steps

After you're comfortable with this workflow:
1. **Parameter sweeps**: Create SLURM array jobs
2. **MVAR pipeline**: Adapt training scripts for Oscar
3. **Batch processing**: Run multiple configs in parallel
4. **Output management**: Organize simulation results
