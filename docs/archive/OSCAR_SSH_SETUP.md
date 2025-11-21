# Oscar SSH Setup Guide

## ✅ What's Configured

### SSH Keys
- **Key type**: Ed25519 (already exists at `~/.ssh/id_ed25519`)
- **Public key installed**: Yes, on Oscar at `~/.ssh/authorized_keys`
- **Benefit**: No more password prompts

### SSH Config (`~/.ssh/config`)
Four host aliases configured:

```bash
# Quick connections
ssh oscar              # Main gateway (requires Duo from off-campus)
ssh oscar-campus       # Campus gateway (may skip Duo on Brown network/VPN)

# For VSCode Remote SSH (in VSCode's "Remote Explorer")
vscode-oscar          # Via main gateway
vscode-oscar-campus   # Via campus gateway
```

### One-Command Workflow
Function added to `~/.bash_profile`:

```bash
oscar-wsindy
```

**What it does:**
1. Starts ssh-agent (if needed)
2. Adds your SSH key (if needed)
3. SSHs to Oscar with agent forwarding
4. Loads miniconda3 module
5. Activates `wsindy` conda environment
6. Changes to `~/wsindy-manifold`
7. Drops you into an interactive bash shell, ready to work

---

## 🚀 Daily Usage

### Quick SSH Session
```bash
ssh oscar
# Duo prompt appears → approve → you're in (no password!)
```

### Start Working on wsindy-manifold
```bash
oscar-wsindy
# Duo prompt → approve → environment ready!
```

### Run Quick Commands
```bash
ssh oscar "cd ~/wsindy-manifold && sacct -j JOBID"
```

### Use VSCode Remote SSH
1. Open VSCode
2. Click Remote Explorer icon (left sidebar)
3. Click "+" to add new SSH target
4. Select `vscode-oscar` or `vscode-oscar-campus`
5. VSCode connects through the configured tunnel

---

## 🔐 About Duo / 2FA

### What Changed
- ✅ **Password prompts removed** (SSH key authentication)
- ✅ **Agent forwarding enabled** (Git works seamlessly)
- ✅ **Keep-alive configured** (fewer disconnects)

### What Didn't Change
- ⚠️ **Duo/2FA still required** (Brown policy, cannot be bypassed)
- From off-campus: `ssh.ccv.brown.edu` always requires Duo
- On campus/VPN: `sshcampus.ccv.brown.edu` *may* skip Duo (Brown decides)

### Why You Can't Bypass Duo
- Brown IT policy requires MFA for all remote access
- Prevents unauthorized access even if SSH keys are compromised
- **This is a good thing** for security

---

## 🛠️ Troubleshooting

### "Permission denied (publickey)"
```bash
# Re-add your key to the agent
ssh-add ~/.ssh/id_ed25519

# Verify it's loaded
ssh-add -L
```

### "ssh-agent not running"
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### Connection drops frequently
The SSH config includes keep-alive settings, but if you still get disconnects:
```bash
# In ~/.ssh/config, increase these values:
ServerAliveInterval 30
ServerAliveCountMax 30
```

### VSCode can't connect
1. Make sure you're using `vscode-oscar` as the host (not the IP)
2. Check that the SSH config is correct: `cat ~/.ssh/config`
3. Try connecting via terminal first: `ssh oscar`

---

## 📝 Quick Reference

| Task | Command |
|------|---------|
| Connect to Oscar | `ssh oscar` |
| Connect (campus/VPN) | `ssh oscar-campus` |
| Start working on project | `oscar-wsindy` |
| Run single command | `ssh oscar "command"` |
| Check job status | `ssh oscar "sacct -j JOBID"` |
| Copy file from Oscar | `scp oscar:~/wsindy-manifold/file.txt .` |
| Copy file to Oscar | `scp file.txt oscar:~/wsindy-manifold/` |
| Sync directory | `rsync -avz oscar:~/wsindy-manifold/outputs/ ./outputs/` |

---

## 🔄 Updating Your Workflow

### Add More Functions
Edit `~/.bash_profile` and add functions like:

```bash
# Check Oscar job queue
oscar-jobs() {
    ssh oscar "squeue -u emaciaso"
}

# Tail latest log file
oscar-logs() {
    ssh oscar "cd ~/wsindy-manifold && tail -f slurm_logs/\$(ls -t slurm_logs/ | head -1)"
}
```

Then reload: `source ~/.bash_profile`

---

## 📚 Brown CCV Documentation

- [CCV SSH Documentation](https://docs.ccv.brown.edu/oscar/connecting-to-oscar/ssh)
- [Oscar User Manual](https://docs.ccv.brown.edu/oscar/)
- [SLURM on Oscar](https://docs.ccv.brown.edu/oscar/submitting-jobs/)

---

**Setup completed**: November 21, 2025  
**Username**: emaciaso  
**Project**: wsindy-manifold
