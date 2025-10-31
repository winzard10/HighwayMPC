# Highway MPC Simulation

This project implements a **lane-following highway controller** using **Linear Time-Varying MPC (LTV-MPC)** in C++, with visualization in Python.  
The workflow is: **build → run simulations → visualize results**.

---

## Quick Commands

```bash
rm -rf build
cmake -S . -B build
cmake --build build -j

# Run simulations
./build/sim_lane_follow
./build/centerline_demo data/lane_centerlines.csv
./build/sim_lane_follow --lane-from right --lane-to left --t-change 5 --T-change 4

# Visualize results
python3 viz_lane_follow.py
python3 viz_lane_follow.py sim_log.csv --circle-at dmin

cd build
cmake ..
make
cd ..
./build/sim_lane_follow 
python3 viz_lane_follow.py