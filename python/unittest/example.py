import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl

def arm_manipulation_problem(T):
    WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
    WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
    signal.signal(signal.SIGINT, signal.SIG_DFL)

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

    # For this optimal control problem, we define 100 knots (or running action
    # models) plus a terminal knot

    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    return problem

def humanoid_taichi_problem(T):
    # Load robot
    robot = example_robot_data.load("talos")
    rmodel = robot.model
    lims = rmodel.effortLimit
    # lims[19:] *= 0.5  # reduced artificially the torque limits
    rmodel.effortLimit = lims

    # Create data structures
    rdata = rmodel.createData()
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFloatingBase(state)

    # Set integration time
    DT = 1e-3
    T = 20
    target = np.array([0.4, 0, 1.2])

    # Initialize reference state, target and reference CoM
    rightFoot = "right_sole_link"
    leftFoot = "left_sole_link"
    endEffector = "gripper_left_joint"
    endEffectorId = rmodel.getFrameId(endEffector)
    rightFootId = rmodel.getFrameId(rightFoot)
    leftFootId = rmodel.getFrameId(leftFoot)
    q0 = rmodel.referenceConfigurations["half_sitting"]
    x0 = np.concatenate([q0, np.zeros(rmodel.nv)])
    pinocchio.forwardKinematics(rmodel, rdata, q0)
    pinocchio.updateFramePlacements(rmodel, rdata)
    rfPos0 = rdata.oMf[rightFootId].translation
    lfPos0 = rdata.oMf[leftFootId].translation
    refGripper = rdata.oMf[rmodel.getFrameId("gripper_left_joint")].translation
    comRef = (rfPos0 + lfPos0) / 2
    comRef[2] = pinocchio.centerOfMass(rmodel, rdata, q0)[2].item()

    # Create two contact models used along the motion
    contactModel1Foot = crocoddyl.ContactModelMultiple(state, actuation.nu)
    contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)
    supportContactModelLeft = crocoddyl.ContactModel6D(
        state,
        leftFootId,
        pinocchio.SE3.Identity(),
        pinocchio.LOCAL,
        actuation.nu,
        np.array([0, 40]),
    )
    supportContactModelRight = crocoddyl.ContactModel6D(
        state,
        rightFootId,
        pinocchio.SE3.Identity(),
        pinocchio.LOCAL,
        actuation.nu,
        np.array([0, 40]),
    )
    contactModel1Foot.addContact(rightFoot + "_contact", supportContactModelRight)
    contactModel2Feet.addContact(leftFoot + "_contact", supportContactModelLeft)
    contactModel2Feet.addContact(rightFoot + "_contact", supportContactModelRight)

    # Cost for self-collision
    maxfloat = sys.float_info.max
    xlb = np.concatenate(
        [
            -maxfloat * np.ones(6),  # dimension of the SE(3) manifold
            rmodel.lowerPositionLimit[7:],
            -maxfloat * np.ones(state.nv),
        ]
    )
    xub = np.concatenate(
        [
            maxfloat * np.ones(6),  # dimension of the SE(3) manifold
            rmodel.upperPositionLimit[7:],
            maxfloat * np.ones(state.nv),
        ]
    )
    bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
    xLimitResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
    xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
    limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)

    # Cost for state and control
    xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv) ** 2
    )
    uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
    xTActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [100] * state.nv) ** 2
    )
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

    # Cost for target reaching: hand and foot
    handTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state, endEffectorId, pinocchio.SE3(np.eye(3), target), actuation.nu
    )
    handTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1] * 3 + [0.0001] * 3) ** 2
    )
    handTrackingCost = crocoddyl.CostModelResidual(
        state, handTrackingActivation, handTrackingResidual
    )

    footTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state, leftFootId, pinocchio.SE3(np.eye(3), np.array([0.0, 0.4, 0.0])), actuation.nu
    )
    footTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1, 1, 0.1] + [1.0] * 3) ** 2
    )
    footTrackingCost1 = crocoddyl.CostModelResidual(
        state, footTrackingActivation, footTrackingResidual
    )
    footTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state,
        leftFootId,
        pinocchio.SE3(np.eye(3), np.array([0.3, 0.15, 0.35])),
        actuation.nu,
    )
    footTrackingCost2 = crocoddyl.CostModelResidual(
        state, footTrackingActivation, footTrackingResidual
    )

    # Cost for CoM reference
    comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)
    comTrack = crocoddyl.CostModelResidual(state, comResidual)

    # Create cost model per each action model. We divide the motion in 3 phases plus its
    # terminal model.
    runningCostModel1 = crocoddyl.CostModelSum(state, actuation.nu)
    runningCostModel2 = crocoddyl.CostModelSum(state, actuation.nu)
    runningCostModel3 = crocoddyl.CostModelSum(state, actuation.nu)
    terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

    # Then let's added the running and terminal cost functions
    runningCostModel1.addCost("gripperPose", handTrackingCost, 1e2)
    runningCostModel1.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel1.addCost("ctrlReg", uRegCost, 1e-4)
    runningCostModel1.addCost("limitCost", limitCost, 1e3)

    runningCostModel2.addCost("gripperPose", handTrackingCost, 1e2)
    runningCostModel2.addCost("footPose", footTrackingCost1, 1e1)
    runningCostModel2.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel2.addCost("ctrlReg", uRegCost, 1e-4)
    runningCostModel2.addCost("limitCost", limitCost, 1e3)

    runningCostModel3.addCost("gripperPose", handTrackingCost, 1e2)
    runningCostModel3.addCost("footPose", footTrackingCost2, 1e1)
    runningCostModel3.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel3.addCost("ctrlReg", uRegCost, 1e-4)
    runningCostModel3.addCost("limitCost", limitCost, 1e3)

    terminalCostModel.addCost("gripperPose", handTrackingCost, 1e2)
    terminalCostModel.addCost("stateReg", xRegTermCost, 1e-3)
    terminalCostModel.addCost("limitCost", limitCost, 1e3)

    # Create the action model
    dmodelRunning1 = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel2Feet, runningCostModel1
    )
    dmodelRunning2 = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel1Foot, runningCostModel2
    )
    dmodelRunning3 = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel1Foot, runningCostModel3
    )
    dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel1Foot, terminalCostModel
    )

    runningModel1 = crocoddyl.IntegratedActionModelEuler(dmodelRunning1, DT)
    runningModel2 = crocoddyl.IntegratedActionModelEuler(dmodelRunning2, DT)
    runningModel3 = crocoddyl.IntegratedActionModelEuler(dmodelRunning3, DT)
    terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)

    # Problem definition
    x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
    problem = crocoddyl.ShootingProblem(
        x0, [runningModel1] * T + [runningModel2] * T + [runningModel3] * T, terminalModel
    )
    return problem
def quadrotor_problem(T):
    hector = example_robot_data.load("hector")
    robot_model = hector.model

    target_pos = np.array([1.0, 0.0, 1.0])
    target_quat = pinocchio.Quaternion(1.0, 0.0, 0.0, 0.0)

    state = crocoddyl.StateMultibody(robot_model)

    d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
    tau_f = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, d_cog, 0.0, -d_cog],
            [-d_cog, 0.0, d_cog, 0.0],
            [-cm / cf, cm / cf, -cm / cf, cm / cf],
        ]
    )
    actuation = crocoddyl.ActuationModelMultiCopterBase(state, tau_f)

    nu = actuation.nu
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)

    # Costs
    xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0.1] * 3 + [1000.0] * 3 + [1000.0] * robot_model.nv)
    )
    uResidual = crocoddyl.ResidualModelControl(state, nu)
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    goalTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state,
        robot_model.getFrameId("base_link"),
        pinocchio.SE3(target_quat.matrix(), target_pos),
        nu,
    )
    goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)
    runningCostModel.addCost("xReg", xRegCost, 1e-6)
    runningCostModel.addCost("uReg", uRegCost, 1e-6)
    runningCostModel.addCost("trackPose", goalTrackingCost, 1e-2)
    terminalCostModel.addCost("goalPose", goalTrackingCost, 3.0)

    dt = 1e-2
    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, runningCostModel
        ),
        dt,
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminalCostModel
        ),
        dt,
    )
    # runningModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
    # runningModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])

    # Creating the shooting problem and the BoxDDP solver
    problem = crocoddyl.ShootingProblem(
        np.concatenate([hector.q0, np.zeros(state.nv)]), [runningModel] * T, terminalModel
    )
    return problem



