# Accelerated Gradient Descent for High Frequency MPC

**Authors:** Jianghan Zhang, Armand Jordana, Ludovic Righetti | **NYU**

We employ Accelerated Gradient Descent (AGD) with ADAM for high-frequency Model Predictive Control (MPC), demonstrating that first-order methods can match the efficacy of second-order methods with simpler computation and faster iteration times. This repository contains the AGD implementation, performance comparisons, and practical applications on real robots.

All the solvers are implemented based on the API of Crocoddyl (v2). In other words, AGD take as input a crocoddyl.ShootingProblem.

**Features:**
- AGD matches second-order method performance at 1kHz control rates for a 7-Dof manipulator.
- Validated on a 7-DOF torque-controlled manipulator with real-world task scenarios.

[View the Repository](https://github.com/JianghanZHang/AGD)

**Installation:**
conda create -n agd_env

conda activate agd_env

conda install mim-solvers crocoddyl pinocchio -c conda-forge

git clone https://github.com/JianghanZHang/AGD.git

cd AGD && mkdir build && cd build

cmake .. [-DCMAKE_BUILD_TYPE=Release] [-DCMAKE_INSTALL_PREFIX=...]

make [-j6] && make install
