#!/bin/bash
# Quick update: pull code and run a fast test
# Usage: ./quick_update.sh

cd /users/emaciaso/src/wsindy-manifold
git pull
source setup_oscar_env.sh
rectsim single --config configs/vicsek_morse_base.yaml --sim.N 50 --sim.T 2.0 --outputs.animate_traj false --outputs.animate_density false
