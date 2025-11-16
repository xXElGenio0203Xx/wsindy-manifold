# Oscar WSindy Workflow - Installation Guide

## ✅ Scripts Created

Two helper scripts have been created in your home directory:

1. **`~/.oscar_wsindy.sh`** - One-command login to Oscar
2. **`~/.oscar_wsindy_sync.sh`** - One-command result sync from Oscar

---

## 📝 Step 1: Add Aliases to Your Shell Config

You need to add these two lines to your shell configuration file.

### If you use **zsh** (default on modern macOS):

```bash
echo 'alias oscar-wsindy="bash $HOME/.oscar_wsindy.sh"' >> ~/.zshrc
echo 'alias oscar-sync="bash $HOME/.oscar_wsindy_sync.sh"' >> ~/.zshrc
source ~/.zshrc
```

### If you use **bash**:

```bash
echo 'alias oscar-wsindy="bash $HOME/.oscar_wsindy.sh"' >> ~/.bashrc
echo 'alias oscar-sync="bash $HOME/.oscar_wsindy_sync.sh"' >> ~/.bashrc
source ~/.bashrc
```

**Not sure which shell you're using?**
```bash
echo $SHELL
```
- If it says `/bin/zsh` → use zsh commands above
- If it says `/bin/bash` → use bash commands above

---

## ✅ Step 2: Verify Installation

After sourcing your shell config, test that the aliases work:

```bash
# Test that aliases are defined
which oscar-wsindy
which oscar-sync

# Should both print the path to your scripts
```

---

## 🚀 Usage

### A. Connect to Oscar and Run Simulations

```bash
# 1. Connect to Oscar (one command!)
oscar-wsindy
# → You'll be prompted for Duo 2FA
# → After approval, you're logged in with:
#    - wsindy conda env activated
#    - In /users/emaciaso/src/wsindy-manifold directory
#    - Ready to run rectsim commands

# 2. On Oscar, sync latest code and test
./sync_and_test.sh

# 3. Run a simulation
rectsim single --config configs/gentle_clustering.yaml --sim.N 200 --sim.T 50.0

# 4. Or submit a SLURM job
sbatch run_rectsim_single.slurm

# 5. Check job status
squeue -u emaciaso

# 6. Exit Oscar when done
exit
```

### B. Sync Results Back to Your Mac

```bash
# On your Mac (NOT on Oscar)
oscar-sync single
# → Copies /users/emaciaso/src/wsindy-manifold/outputs/single
#   to ~/wsindy-results/single/

# Or sync a specific run
oscar-sync gentle_clustering_20251116

# View synced results
cd ~/wsindy-results/single/
ls -lh
open *.png  # View plots
```

---

## 📊 Typical Workflow

```bash
# ═══════════════════════════════════════════════════════════
# On your Mac: Edit code, commit, push to GitHub
# ═══════════════════════════════════════════════════════════
cd ~/Desktop/wsindy-manifold
# (edit Python files in VS Code)
git add .
git commit -m "Improve alignment algorithm"
git push origin main

# ═══════════════════════════════════════════════════════════
# Connect to Oscar
# ═══════════════════════════════════════════════════════════
oscar-wsindy
# (approve Duo 2FA)

# ═══════════════════════════════════════════════════════════
# On Oscar: Pull latest code and run simulations
# ═══════════════════════════════════════════════════════════
./sync_and_test.sh  # Pull from GitHub and test

# Option A: Interactive test
rectsim single --config configs/gentle_clustering.yaml --sim.N 100 --sim.T 10.0

# Option B: Submit SLURM job
sbatch run_rectsim_single.slurm
squeue -u emaciaso  # Monitor

# Wait for job to complete, then exit
exit

# ═══════════════════════════════════════════════════════════
# Back on Mac: Sync results
# ═══════════════════════════════════════════════════════════
oscar-sync single
cd ~/wsindy-results/single/
open *.png  # Analyze results
```

---

## 🔧 Advanced Usage

### Sync specific output directories

```bash
# List available outputs on Oscar first
oscar-wsindy
ls outputs/
exit

# Then sync the one you want
oscar-sync test_run_oscar
oscar-sync gentle_clustering_20251116_123456
```

### Sync multiple runs

```bash
# Create a simple loop
for run in run1 run2 run3; do
    oscar-sync "$run"
done
```

### Re-sync (update only changed files)

```bash
# rsync is smart - running it again only transfers changes
oscar-sync single  # First time: copies everything
# (make changes on Oscar, run more simulations)
oscar-sync single  # Second time: only copies new/changed files
```

---

## 📁 Where Things Are

| Location | Path | Purpose |
|----------|------|---------|
| Oscar repo | `/users/emaciaso/src/wsindy-manifold` | Your code and configs |
| Oscar outputs | `/users/emaciaso/src/wsindy-manifold/outputs/` | Simulation results |
| Mac login script | `~/.oscar_wsindy.sh` | Connect to Oscar |
| Mac sync script | `~/.oscar_wsindy_sync.sh` | Pull results |
| Mac results | `~/wsindy-results/` | Synced outputs |

---

## 🐛 Troubleshooting

### "command not found: oscar-wsindy"
```bash
# Reload your shell config
source ~/.zshrc  # or source ~/.bashrc
```

### "Permission denied" when syncing
```bash
# Make sure scripts are executable
chmod +x ~/.oscar_wsindy.sh
chmod +x ~/.oscar_wsindy_sync.sh
```

### "Remote directory does not exist"
```bash
# Check what outputs exist on Oscar
oscar-wsindy
ls outputs/
exit

# Then sync the correct directory name
oscar-sync <actual_directory_name>
```

### SSH connection hangs
- Check your network connection
- Make sure Duo 2FA is approved
- Try connecting manually first: `ssh emaciaso@ssh.ccv.brown.edu`

---

## 🎯 Summary

**Two commands to rule them all:**

```bash
oscar-wsindy   # Connect to Oscar (with environment ready)
oscar-sync RUN_NAME  # Pull results to Mac
```

No more copy-pasting complex SSH/rsync/conda commands! 🎉
