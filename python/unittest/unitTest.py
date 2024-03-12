import pdb
import crocoddyl
import numpy as np
import sys  
sys.path.append('..')
sys.path.append('../../build/bindings/grg_pywrap')
import grg_pywrap as grg
# from solverNAG_lineSearch import SolverMSls
import time

from example import quadrotor_problem
from example import arm_manipulation_problem
from example import humanoid_taichi_problem
import matplotlib.pyplot as plt
import pdb
import json
import os


class Tester:
    def __init__(self, NX=1, NU=1, T=20, maxIter=10):
        self.NX = NX
        self.NU = NU
        self.T = T

        # self.x0 = np.ones(NX) * 1.
        self.runningModel = crocoddyl.ActionModelLQR(NX, NU)
        self.terminalModel = crocoddyl.ActionModelLQR(NX, NU)

        self.problem_arm_manipulation = arm_manipulation_problem(self.T)
        self.problem_humanoid_taichi = humanoid_taichi_problem(self.T)
        self.problem_quadrotor = quadrotor_problem(self.T)

        self.problem = None

    def test(self, solver, MaxIter):
        
        init_us = self.problem.quasiStatic([self.problem.x0] * self.problem.T)
        init_xs = self.problem.rollout(init_us)
        self.init_us = init_us

        self.Solver1 = grg.SolverGRG_ADAHESSIAN(self.problem)
        DDP = crocoddyl.SolverDDP(self.problem)

        self.Solver1.use_line_search = False
        self.Solver1.mu = 1.0
        self.Solver1.const_step_length = .25
        self.Solver1.beta1 = 0.9
        self.Solver1.beta2 = 0.999

        self.Solver1.with_callbacks = True

        self.Solver1.solve(init_xs, init_us, MaxIter)

        DDP.setCallbacks(
            [
                crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose(),
            ]
            )

        DDP.solve(init_xs, init_us, MaxIter)

        print(DDP.cost)


if __name__ == '__main__':
    T = 50
    tester = Tester(6, 6, T)

    tester.problem = tester.problem_humanoid_taichi

    print(f'problem: {tester.problem}')

    ddp_maxIter = 50

    maxIter = 100

    tester.test("MSls", maxIter)


