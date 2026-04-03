#!/bin/bash
# install_oscar_bashrc_guard.sh
#
# Run ONCE from your Mac to append a login-node guard to your OSCAR ~/.bashrc.
# After this, every interactive SSH session on Oscar will block 'du' on login
# nodes and remind you to use 'interact' for any heavy work.
#
# Usage:
#   bash shell_scripts/install_oscar_bashrc_guard.sh
# ============================================================================

OSCAR_HOST="${OSCAR_HOST:-ssh.ccv.brown.edu}"
GUARD_TAG="# >>> wsindy-manifold login-node guard <<<"

echo "Installing login-node guard on OSCAR (~/.bashrc) ..."

ssh "$OSCAR_HOST" bash << ENDSSH
set -e

GUARD_TAG='${GUARD_TAG}'

# Idempotent: skip if already installed
if grep -qF "\$GUARD_TAG" ~/.bashrc 2>/dev/null; then
    echo "Guard is already installed in ~/.bashrc — nothing to do."
    exit 0
fi

cat >> ~/.bashrc << 'GUARD'

# >>> wsindy-manifold login-node guard <<<
# Blocks 'du' on login nodes, which are shared and have strict resource limits.
# Installed by install_oscar_bashrc_guard.sh — remove this block to uninstall.
_on_login_node() { hostname 2>/dev/null | grep -qE '^login[0-9]'; }
if _on_login_node; then
    echo ""
    echo "  [OSCAR] Login node detected. Heavy commands (du, find -exec, python training)"
    echo "          are blocked here. Use 'interact' for compute work, 'myquota' for quota."
    echo ""
    du() {
        echo "ERROR: 'du' is blocked on login nodes — it saturates shared memory." >&2
        echo "       Use 'myquota' for quota info, or first run 'interact' to get a compute node." >&2
        echo "       Command was: du \$*" >&2
        return 1
    }
    export -f du
fi
# <<< wsindy-manifold login-node guard <<<
GUARD

echo "Guard installed successfully in ~/.bashrc."
echo "It will take effect on your next SSH login."
ENDSSH
