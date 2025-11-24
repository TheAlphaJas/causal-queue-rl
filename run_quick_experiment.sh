#!/bin/bash
# Quick experiment runner for Linux/Mac
# Trains all 3 algorithms with 15k steps

echo "========================================"
echo "Running Quick RL Experiment"
echo "========================================"
echo

python run_experiments.py --total-steps 15000 --seed 42

echo
echo "========================================"
echo "Experiment Complete!"
echo "Check the experiments_* folder for results"
echo "========================================"

