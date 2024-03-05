import pdb
import sys
sys.path.append('..')
sys.path.append('../../build/bindings/grg_pywrap')
import pinocchio as pin
import crocoddyl
import numpy as np
from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot
import pybullet as p
import mpc_utils
import grg_pywrap as grg
import time
import matplotlib.pyplot as plt
import eigenpy
from datetime import datetime
import faulthandler
np.set_printoptions(precision=4, linewidth=180)
eigenpy.sharedMemory(False)
# faulthandler.enable()

def solveOCP(solver, x_curr, xs_init, us_init, targets, maxIter, warm_start=True):
    # print('Solving OCP...')

    # pdb.set_trace()
    solver.problem.x0 = x_curr.reshape((14,))

    if warm_start:
    # warm start
        us_init = list(us_init[1:]) + [us_init[-1]]
        xs_init = list(xs_init[1:]) + [xs_init[-1]]
        xs_init[0] = x_curr
    else:
        xs_init = list(xs_init)
        us_init = list(us_init)  
        xs_init[0] = x_curr

    # xs_init = np.array(xs_init)
    # us_init = np.array(us_init)
    # Get OCP nodes
    models = list(solver.problem.runningModels) + [solver.problem.terminalModel]
    for k, model in enumerate(models):
        model.differential.costs.costs["translation"].active = True
        model.differential.costs.costs["translation"].cost.residual.reference = targets[k]
    
    solver.solve(xs_init, us_init, maxIter)
  
    # calculating cost of the current node
    u_curr = solver.us[0]
    runningModel0 = solver.problem.runningModels[0]
    runningData0 = solver.problem.runningDatas[0]
    runningModel0.calc(runningData0, x_curr, u_curr)
    runningCost = runningData0.cost
    # pdb.set_trace()
    totalCost = solver.cost

    # Qu = solver.Vu
    # kkt = np.linalg.norm(Qu, np.inf)
    # print('Solving ocp done')
    return np.array(solver.us), np.array(solver.xs), runningCost, totalCost

def circleTraj(T, t, dt):
    pi = np.pi
    # orientation = pin.Quaternion(np.array([pi, 0., 0., 1.]))
    # orientation = np.eye(3)
    # target = [pin.SE3(orientation, np.array([.4 + (.1 * np.cos(pi * (t + j*dt))),
    #                                        .2 + (.1 * np.sin(pi * (t + j*dt))),
    #                                        .3 + (.1 * np.sin(pi * (t + j*dt)))
    #                                        ])) for j in range(T + 1)]
    targets = np.zeros([T + 1, 3])
    for j in range(T + 1):

        targets[j, 0] = .4  + (.3 * np.cos(pi * (t + j*dt)))
        targets[j, 1] = .4 # + (.3 * np.sin(pi * (t + j*dt)))
        targets[j, 2] = .4  + (.3 * np.sin(pi * (t + j*dt)))

    return targets


if __name__ == '__main__':

    env = BulletEnvWithGround(p.GUI, dt=10e-3)
    
    robot_simulator = IiwaRobot()
    pin_robot = robot_simulator.pin_robot
    q0 = np.array([0.9755, 1.2615, 1.7282, 1.8473, -1.0791, 2.0306, -0.0759])
    v0 = np.zeros(pin_robot.model.nv)
    x0 = np.concatenate([q0, v0])
    env.add_robot(robot_simulator)
    robot_simulator.reset_state(q0, v0)
    robot_simulator.forward_robot(q0, v0)

    state = crocoddyl.StateMultibody(pin_robot.model)
    actuation = crocoddyl.ActuationModelFull(state)
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)
    ee_frame_id = robot_simulator.pin_robot.model.getFrameId("contact")
    ee_translation = np.array([.4, .4, .4])
    
    R = np.array([[0., 1., 0.],
                  [1., 0., 0.],
                  [0., 0., -1.]])
    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, ee_frame_id, ee_translation)
    # frameTranslationResidual = crocoddyl.ResidualModelFramePlacement(
    #     state,
    #     ee_frame_id,  # This is for ur5
    #     pin.SE3(R, ee_translation),
    # )
    
    frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
    uResidual = crocoddyl.ResidualModelControlGrav(state)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    xResidual = crocoddyl.ResidualModelState(state, x0)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    runningCostModel.addCost("stateReg", xRegCost, 1e-1)
    runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
    runningCostModel.addCost("translation", frameTranslationCost, 1e1)

    terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
    terminalCostModel.addCost("translation", frameTranslationCost, 3e1)

    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)
    dt_ocp = 1e-2
    runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt_ocp)
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
    T = 20
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    problem.nthreads = 8
    ddp = crocoddyl.SolverDDP(problem)

    SolverGRG = grg.SolverGRG(problem)
    # SolverGRG = grg.SolverGRG_ADAHESSIAN(problem)


    SolverGRG.beta1 = (0.8)
    SolverGRG.beta2 = (0.8)
    SolverGRG.const_step_length = (0.1)
    SolverGRG.correct_bias = True
    # SolverGRG.use_line_search = True

    solver = SolverGRG

    solver.with_callbacks = True


    time_ = 10.
    t = 0.
    print(env.dt)
    dt_sim = env.dt  # 1e-3
    sim_freq = 1/dt_sim
    dt_mpc = env.dt
    mpc_freq = 1/dt_mpc
    num_step = int(time_ / dt_sim)

    # warm starting us
    x_measured = x0
    xs_init = [x0 for i in range(T + 1)]
    us_init = ddp.problem.quasiStatic(xs_init[:-1])
    targets = circleTraj(T, t, dt_ocp)
    #x_measured.reshape(problem.ndx, 1)
    us, xs, runningCost, totalCost = solveOCP(ddp, x0, xs_init, us_init, targets, 100)
    us = np.array(us)
    xs = np.array(xs)

    mpc_utils.display_ball(ee_translation , RADIUS=.05, COLOR=[1., 0., 0., .6])
    q_measured = q0

    log_rate = 100
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(10, 15))
    desired_ee_positions = []
    measured_ee_positions = []
    times = []
    start_time = datetime.now()  # Capture start time
    
    print("STARTING SIMULATION")
    for i in range(num_step):
       
        start = time.time()
        # tau_gravity = pin.rnea(pin_robot.model, pin_robot.data, q_measured, np.zeros_like(q_measured), np.zeros_like(q_measured))

        if i % (int(sim_freq / mpc_freq)) == 0:
            desired_ee_positions.append(np.array(targets[0]))
            targets = circleTraj(T, t, dt_ocp)
            maxIter = 10
            us, xs, runningCost, totalCost = solveOCP(solver, x_measured, solver.xs, us, targets, maxIter)


            tau = us[0]
            if i % log_rate == 0:
                print(f'at step {i}: tau={tau}')

            # tau += tau_gravity
            robot_simulator.send_joint_command(tau)

            measured_ee_positions.append(np.array(robot_simulator.pin_robot.data.oMf[ee_frame_id].translation))

            env.step()
            q_measured, v_measured = robot_simulator.get_state()
            robot_simulator.forward_robot(q_measured, v_measured)
            x_measured = np.concatenate([q_measured, v_measured])#.reshape(problem.ndx, 1)   
            t += dt_sim
        
        end = time.time()

        times.append(end - start)


    # crocoddyl.stop_watch_report(3)
    
    indices = [i for i, value in enumerate(times) if value > env.dt]

    print(indices)
    
    end_time = datetime.now()  # Capture end time after the loop

    print(f"Loop took {end_time - start_time}")


    measured_ee_positions = np.array(measured_ee_positions)
    desired_ee_positions = np.array(desired_ee_positions)
    error = np.linalg.norm(np.abs(measured_ee_positions - desired_ee_positions), axis=1, ord = 1)

