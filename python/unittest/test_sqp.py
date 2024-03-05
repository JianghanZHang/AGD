import crocoddyl
import numpy as np
import time
import example_robot_data
import pinocchio
import mim_solvers

# In this example test, we will solve the reaching-goal task with the Kinova arm.
# For that, we use the forward dynamics (with its analytical derivatives) developed
# inside crocoddyl; it is described inside DifferentialActionModelFreeFwdDynamics class.
# Finally, we use an Euler sympletic integration scheme.

# First, let's load create the state and actuation models
robot = example_robot_data.load("ur5")
robot_model = robot.model
state = crocoddyl.StateMultibody(robot_model)
actuation = crocoddyl.ActuationModelFull(state)
# q0 = kinova.model.referenceConfigurations["arm_up"]
q0 = np.zeros(robot_model.nq)
x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])

# Create a cost model per the running and terminal action model.
nu = state.nv
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)
constraintManager = crocoddyl.ConstraintModelManager(state)

# Note that we need to include a cost model (i.e. set of cost functions) in
# order to fully define the action model for our optimal control problem.
# For this particular example, we formulate three running-cost functions:
# goal-tracking cost, state and control regularization; and one terminal-cost:
# goal cost. First, let's create the common cost functions.

framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
    state,
    # robot_model.getFrameId("j2s6s200_end_effector"), # This is for kinova
    robot_model.getFrameId("ee_link"),  # This is for ur5
    pinocchio.SE3(np.eye(3), np.array([0.3, 0.3, 0.3])),
    nu,
)

frameTranslationConstraint = crocoddyl.ResidualModelFrameTranslation(
    state,
    # robot_model.getFrameId("j2s6s200_end_effector"), # This is for kinova
    robot_model.getFrameId("ee_link"),  # This is for ur5
    np.array([0.3, 0.3, 0.]),
    nu,
)

frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
    state,
    # robot_model.getFrameId("j2s6s200_end_effector"), # This is for kinova
    robot_model.getFrameId("ee_link"),  # This is for ur5
    np.array([0.3, 0.3, 0.3]),
    nu,
)
# frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state,robot_model.getFrameId("j2s6s200_end_effector"), np.array([0.2, 0.2, 0.5]))

uResidual = crocoddyl.ResidualModelControl(state, nu)
xResidual = crocoddyl.ResidualModelState(state, x0, nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

lb = np.array([-np.inf, -np.inf, -np.inf])
ub = np.array([np.inf, np.inf, np.inf])
Constraint = crocoddyl.ConstraintModelResidual(state, frameTranslationConstraint, lb, ub)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, .1)
# runningCostModel.addCost("xReg", xRegCost, 1e-1)
runningCostModel.addCost("uReg", uRegCost, 1e-4)

constraintManager.addConstraint("zConstaint", Constraint)

terminalCostModel.addCost("gripperPose", goalTrackingCost, .1)

# Next, we need to create an action model for running and terminal knots. The
# forward dynamics (computed using ABA) are implemented
# inside DifferentialActionModelFreeFwdDynamics.
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, runningCostModel, constraintManager
    ),
    dt,
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, terminalCostModel, constraintManager
    ),
    0.0,
)

T = 30 

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
sqp = mim_solvers.SolverSQP(problem)
ddp = mim_solvers.SolverFDDP(problem)

time_ddp = []
time_sqp = []

init_us = problem.quasiStatic([problem.x0] * problem.T)
init_xs = problem.rollout(init_us)


nb_iter = 10000
for i in range(nb_iter):
    time_start = time.time()
    ddp.solve(init_xs, init_us, 1)
    time_end = time.time()
    time_ddp.append(time_end - time_start)

for i in range(nb_iter):
    time_start = time.time()
    sqp.solve(init_xs, init_us, 1)
    time_end = time.time()
    time_sqp.append(time_end - time_start)

print(f"Average time for DDP: {np.mean(time_ddp) * 1000} ms")
print(f"Average time for SQP: {np.mean(time_sqp) * 1000} ms")
