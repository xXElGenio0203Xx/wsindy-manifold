#!/bin/bash
# _oscar_guard.sh — Source this at the top of any script that runs on Oscar.
#
# What it does:
#   1. Detects login nodes (hostname matches login*) and blocks commands that
#      are known to saturate login-node memory.
#   2. Overrides 'du' with a wrapper that refuses to recurse into large
#      directories (>depth 1) from a login node.
#   3. Prints a reminder banner about login-node rules.
#
# Usage:
#   source shell_scripts/_oscar_guard.sh   # from local machine
#   source ~/wsindy-manifold/shell_scripts/_oscar_guard.sh  # on Oscar
# ============================================================================

_on_login_node() {
    # Returns 0 (true) if the current host is an Oscar login node.
    hostname 2>/dev/null | grep -qE '^login[0-9]'
}

if _on_login_node; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  OSCAR LOGIN NODE — light tasks only                            ║"
    echo "║  Heavy work must go through SLURM or 'interact'.               ║"
    echo "║  BANNED on login nodes: du, find -exec, python training        ║"
    echo "║  For disk usage: myquota   OR   interact then du               ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""

    # Override 'du' so it refuses to run on the login node with a deep path
    du() {
        echo "ERROR: 'du' is blocked on login nodes — it saturates memory." >&2
        echo "       Use 'myquota' for quota info, or run inside 'interact'." >&2
        echo "       Command was: du $*" >&2
        return 1
    }
    export -f du
fi
