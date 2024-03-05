import pdb
import crocoddyl
import numpy as np
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

with open("ALMconfig.json", 'r') as file:
    CONFIG = json.load(file)

class Tester:
    def __init__(self, NX=1, NU=1, T=20, maxIter=10):
        self.NX = NX
        self.NU = NU
        self.T = T

        # self.x0 = np.ones(NX) * 1.
        self.x0 = np.random.uniform(low=-2 * np.pi, high= 2 * np.pi, size=(NX, 1))
        self.runningModel = crocoddyl.ActionModelLQR(NX, NU)
        self.terminalModel = crocoddyl.ActionModelLQR(NX, NU)
        self.problem_lqr = crocoddyl.ShootingProblem(self.x0, [self.runningModel] * T, self.terminalModel)

        self.problem_arm_manipulation = arm_manipulation_problem(self.T)
        self.problem_humanoid_taichi = humanoid_taichi_problem(self.T)
        self.problem_quadrotor = quadrotor_problem(self.T)

        self.problem = None

    def test(self, solver, params=None):

        
        init_us = self.problem.quasiStatic([self.problem.x0] * self.problem.T)
        init_xs = self.problem.rollout(init_us)
        self.init_us = init_us


        Iteration = 1000

        self.Solver1 = grg.SolverGRG(self.problem)

        self.Solver1.solve(init_xs, init_us, Iteration)
        # self.Solver2.solve(init_xs, init_us, Iteration)
        # self.Solver3.solve(init_xs, init_us, Iteration)
        print(f'optimal control form multiple shooting solver with line search: cost= {self.Solver1.costs[-1]},'
    f'\nus[0]:{self.Solver1.us[0]}, xs[0]:{self.Solver1.xs[0]}\nus[-1]:{self.Solver1.us[-1]}, xs[-1]:{self.Solver1.xs[-1]}\n\n')
        
        fig1, (ax1, ax2, ax3, ax4, ax6) = plt.subplots(5, sharex=True, figsize=(10, 15))

        fig1.suptitle(f'Solver metrics, # of iterations={Iteration}, T={T}', fontsize=16)
        
        color = 'tab:blue'
        ax1.set_ylabel('Objective', color='tab:red')
        ax1.plot(self.Solver1.costs[:], color='tab:red', label = 'ADAM',linestyle='-')
        # ax1.plot(self.Solver2.costs[:], color='tab:green', label = 'Vanilla',linestyle='-')
        # ax1.plot(self.Solver3.costs[:], color='tab:blue', label = 'Nesterov',linestyle='-')
        # ax1.plot(self.Solver4.costs[:], color='tab:pink', label = 'Vanilla_ALM',linestyle='-')
        # ax1.plot(self.Solver5.costs[:], color='y', label = 'NAG_ALM',linestyle='-')
        # ax1.plot(self.Solver6.costs[:], color='tab:orange', label = 'ADAM_ALM',linestyle='-')
        ax1.legend(loc='upper right')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlabel('Iteration')  # Set the x-axis label
        ax1.grid(True)

        color = 'tab:red'
        ax2.set_ylabel('Constraint violations (log10)', color='tab:red')
        ax2.plot(np.log10(self.Solver1.Infeasibilities[:]), 'tab:red', label = 'ADAM', linestyle='-')
        # ax2.plot(np.log10(self.Solver2.Infeasibilities[:]), color='tab:green', label = 'Vanilla',linestyle='-')
        # ax2.plot(np.log10(self.Solver3.Infeasibilities[:]), color='tab:blue', label = 'Nesterov',linestyle='-')
        # ax2.plot(np.log10(self.Solver4.Infeasibilities[:]), color='tab:pink', label = 'Vanilla_ALM',linestyle='-')
        # ax2.plot(np.log10(self.Solver5.Infeasibilities[:]), color='y', label = 'NAG_ALM',linestyle='-')
        # ax2.plot(np.log10(self.Solver6.Infeasibilities[:]), color='tab:orange', label = 'ADAM_ALM',linestyle='-')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        ax2.set_xlabel('Iteration')  # Set the x-axis label
        ax2.grid(True)

        color = 'tab:red'
        ax4.set_ylabel('gradients', color='tab:red')
        ax4.plot(np.log10(self.Solver1.gradients[:]), color='tab:red', label = 'ADAM', linestyle='-')
        # ax4.plot(np.log10(self.Solver2.gradients[:]), color='tab:green', label = 'Vanilla', linestyle='-')
        # ax4.plot(np.log10(self.Solver3.gradients[:]), color='tab:blue', label = 'Nesterov', linestyle='-')
        # ax4.plot(np.log10(self.Solver4.gradients[:]), color='tab:pink', label = 'Vanilla_ALM', linestyle='-')
        # ax4.plot(np.log10(self.Solver5.gradients[:]), color='y', label = 'NAG_ALM',linestyle='-')
        # ax4.plot(np.log10(self.Solver6.gradients[:]), color='tab:orange', label = 'ADAM_ALM',linestyle='-')
        ax4.tick_params(axis='y', labelcolor=color)
        ax4.legend(loc='upper right')
        ax4.set_xlabel('Iteration')  # Set the x-axis label
        ax4.grid(True)

        color = 'tab:red'
        ax3.set_ylabel('KKT (log10)', color='tab:red')
        ax3.plot(np.log10(self.Solver1.KKTs[:]), color='tab:red', label = 'ADAM', linestyle='-')
        # ax3.plot(np.log10(self.Solver2.KKTs[:]), color='tab:green', label = 'Vanilla', linestyle='-')
        # ax3.plot(np.log10(self.Solver3.KKTs[:]), color='tab:blue', label = 'Nesterov', linestyle='-')
        # ax3.plot(np.log10(self.Solver4.KKTs[:]), color='tab:pink', label = 'Vanilla_ALM', linestyle='-')
        # ax3.plot(np.log10(self.Solver5.KKTs[:]), color='y', label = 'NAG_ALM',linestyle='-')
        # ax3.plot(np.log10(self.Solver6.KKTs[:]), color='tab:orange', label = 'ADAM_ALM',linestyle='-')
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.legend(loc='upper right')
        ax3.set_xlabel('Iteration')  # Set the x-axis label
        ax3.grid(True)


        
        color = 'tab:red'
        ax6.set_ylabel('alphas', color='tab:red')
        ax6.plot(self.Solver1.alphas[:], color='tab:red', label = 'ADAM', linestyle='-')
        # ax6.plot(self.Solver2.alphas[:], color='tab:green', label = 'Vanilla', linestyle='-')
        # ax6.plot(self.Solver3.alphas[:], color='tab:blue', label = 'Nesterov', linestyle='-')
        # ax6.plot(self.Solver4.alphas[:], color='tab:pink', label = 'Vanilla_ALM', linestyle='-')
        # ax6.plot(self.Solver5.alphas[:], color='y', label = 'NAG_ALM',linestyle='-')
        # ax6.plot(self.Solver6.alphas[:], color='tab:orange', label = 'ADAM_ALM',linestyle='-')
        ax6.tick_params(axis='y', labelcolor=color)
        ax6.legend(loc='upper right')
        ax6.set_xlabel('Iteration')  # Set the x-axis label
        ax6.grid(True)

        # color = 'tab:red'
        # ax5.set_ylabel('curvatures', color='tab:red')
        # ax5.plot(self.Solver1.curvatures[:], color='tab:red', label = 'multiple shooting', linestyle='-')
        # ax5.plot(self.ADAM.curvatures[:], color='tab:green', label = 'single shooting', linestyle='-')
        # # ax4.plot(self.ALM.alpha, color='tab:blue', label = 'ALM',linestyle='-')
        # ax5.tick_params(axis='y', labelcolor=color)
        # ax5.legend(loc='upper right')
        # ax5.set_xlabel('Iteration')  # Set the x-axis label
        # ax5.grid(True)
        # pdb.set_trace()
        print('Minimum alpha:' , min(self.Solver2.alphas[:]))

if __name__ == '__main__':
    T = 50
    tester = Tester(6, 6, T)

    tester.problem = tester.problem_arm_manipulation

    # pdb.set_trace()

    print(f'problem: {tester.problem}')

    params_ddp = {
        "maxIter": 50
    }
    
    tester.test("DDP", params_ddp)

    # tester.test("MS", params_MS)

    params = {
        "maxIter": 100
    }


    # tester.test("ALM")
    tester.test("ADAM", params)
    tester.test("MSls", params)
    

    # tester.test("ALM", params_MSls)
    # print(f'Average control from FDDP = {np.linalg.norm(np.array(tester.FDDP_hybrid.us), 2)/T}\nAverage control from MSls = {np.linalg.norm(np.array(tester.MSls.us), 2)/T}')
    # print(f'Solution difference Average (warmstart) = {np.linalg.norm(np.array(tester.FDDP_hybrid.us)-np.array(tester.MSls.us), 2)/T}')
    # print(f'Solution difference Average (initial) = {np.linalg.norm(np.array(tester.FDDP_hybrid.us)-np.array(tester.init_us), 2)/T}')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    

