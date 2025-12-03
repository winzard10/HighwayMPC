# A Study on Model Predictive Control Architectures for Passenger Autonomous Vehicles

A full C++/Python simulation framework for **highway autonomous driving** using **Linear Time-Varying Model Predictive Control (LTV-MPC)**, dynamic bicycle models, tire forces, obstacle-aware corridor planning, and visualization tools.

## Features

- Dynamic bicycle model with Magic Formula tire forces, dynamic load transfer
- LTV-MPC controller using OsqpEigen
- Tracks lateral/heading error
- Steering-rate and jerk penalization
- Speed & curvature preview
- Obstacle-aware corridor planning
- Lane-change trajectories (time-scheduled)
- ACC / gap-keeping module
- Python visualization for trajectory, tire forces, friction circle, understeer, etc.

## Dependencies

- C++20
- cmake 3.16
- Eigen3
- OSQP + OsqpEigen
- Python 3.10: numpy, matplotlib
- Tested on Ubuntu 22.04

**Eigen3**
```bash
sudo apt install build-essential cmake libeigen3-dev
```

**OSQP**
```bash
sudo apt install libosqp-dev
pip install osqp
pip install osqp-eigen
```

**Python Dependencies**
```bash
pip install numpy matplotlib
```

## Simulation

```bash
# Load or sample map
./build/centerline_demo data/lane_centerlines.csv

# Run simulations with no lane change
./build/sim_lane_follow

# Run simulations with lane change
./build/sim_lane_follow --lane-from right --lane-to left --t-change 5 --T-change 4

# Visualize results
python3 viz_lane_follow.py sim_log.csv

# Useful copy and paste command
cd build
cmake ..
make
cd ..
./build/sim_lane_follow --lane-from right --lane-to left --t-change 5 --T-change 4
python3 viz_lane_follow.py

```

## Citation 

ROB 590: Directed Study

Model Predictive Control for Highway Autonomous Driving:
A Study on Model Predictive Control Architectures for Passenger Autonomous Vehicles by **Phurithat Tangsripairoje**, Department of Robotics, University of Michigan
Supervised by Prof. Tulga Ersal, Department of Mechanical Engineering, University of Michigan