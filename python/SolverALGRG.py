""""
This implementation is based on https://link.springer.com/article/10.1007/s00245-020-09718-8
"""
import pdb

import numpy as np
from numpy import linalg
import scipy.linalg as scl
import crocoddyl
from crocoddyl import SolverAbstract

LINE_WIDTH = 100

VERBOSE = False

def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error

class SolverAlGRG(SolverAbstract):
    def __init__(self, shootingProblem):
        SolverAbstract.__init__(self, shootingProblem)
        self.cost = 0.
        self.cost_try = 0.
        self.threshold = 1e-12
        self.stop = 0.
        self.x_reg = 0
        self.u_reg = 0
        self.regFactor = 10
        self.regMax = 1e9
        self.regMin = 1e-9
        self.th_step = .5
        self.th_stop = 1.e-10
        self.n_little_improvement = 0
        
        self.c1 = 0.1
        self.past_grad = 0.
        self.curr_grad = 0.
        self.change = 0.
        self.change_p = 0.

        #Zhang & Hager line search
        self.C = 0
        self.Q = 1
        self.Eta = 0.1
        self.delta = 0.25
        self.sigma = 0.5

        self.allocateData()

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod

    def calc(self):
        # compute cost and derivatives at deterministic nonlinear trajectory
        self.problem.calc(self.xs, self.us)
        cost = self.problem.calcDiff(self.xs, self.us)
        return cost

    def computeDirection(self, recalc=True):
        if recalc:
            if VERBOSE: print("Going into Calc from compute direction")
            self.calc()
        if VERBOSE: print("Going into Backward Pass from compute direction")
        self.dJdu_p = self.dJdu.copy()
        self.dLdu_p = self.dLdu.copy()
        self.backwardPass()

    def backwardPass(self):
        temp = 0.
        self.H = 0
        self.dJdx[-1, :] = self.problem.terminalData.Lx
        self.constraint_norm = 0.

        if self.problem.terminalModel.ng:
            violated = ((self.problem.terminalModel.g_ub < self.problem.terminalData.g) | (self.problem.terminalModel.g_lb > self.problem.terminalData.g)).reshape((3,1))
            self.dHdu[-1] += violated * self.problem.terminalData.Gu
            self.dHdx[-1] += violated * self.problem.terminalData.Gx
            self.h[-1] = np.minimum(np.maximum(np.zeros_like(self.h[-1]),  self.problem.terminalData.g - self.problem.terminalModel.g_ub)
                                    , self.problem.terminalData.g - self.problem.terminalModel.g_lb).reshape(3)

        # self.g[-1, :] = np.maximum(self.g[-1, :], self.problem.terminalData.g - self.problem.terminalModel.g_ub)
            temp = self.rho[-1] * (self.dHdu[-1].T @ (self.h[-1] + self.lambdas[-1] / (self.rho[-1])))
            self.constraint_norm  += np.linalg.norm(self.h[-1], 1)

        self.dLdu[-1, :] = self.dJdu[-1, :] + temp

        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            temp = 0.
            self.dJdu[t, :] = data.Lu + self.dJdx[t + 1, :] @ data.Fu
            self.dJdx[t, :] = data.Lx + self.dJdx[t + 1, :] @ data.Fx

            if model.ng:
            
                violated = ((model.g_ub < data.g) | (model.g_lb > data.g)).reshape((3,1)) 
                self.h[t] = np.minimum(np.maximum(np.zeros_like(self.h[t]),  self.problem.terminalData.g - self.problem.terminalModel.g_ub)
                                    , self.problem.terminalData.g - self.problem.terminalModel.g_lb).reshape(3)
                self.constraint_norm += np.linalg.norm(self.h[t], 1)
                
                self.dHdu[t] = violated * data.Gu + self.dHdx[t + 1] @ data.Fu
                self.dHdx[t] = violated * data.Gx + self.dHdx[t + 1] @ data.Fx
                temp = self.rho[t] * (self.dHdu[t].T @ (self.h[t] + violated.reshape((3,)) * self.lambdas[t] / (self.rho[t])))
            else:
                self.dHdu[t] = self.dHdx[t + 1] @ data.Fu
                self.dHdx[t] = self.dHdx[t + 1] @ data.Fx
                # temp = self.rho[t] * (self.dHdu[t].T @ (self.h[t] + violated.reshape((3,)) * self.lambdas[t] / (self.rho[t])))
            self.dLdu[t, :] = self.dJdu[t, :] + temp
        self.inequality.append(self.constraint_norm )
        self.kkt = linalg.norm(np.hstack((self.dLdu, self.gap[1:])), np.inf)
        self.KKTs.append(self.kkt)
        # self.Infeasibilities.append(np.linalg.norm(self.gap, np.inf))
        self.gap_norm = np.linalg.norm(self.gap, 1)
        self.gradient = linalg.norm(self.dLdu, np.inf)
        self.gradients.append(self.gradient)
    
        
    def forwardPass(self, alpha, i):

        

        self.m_p = self.m.copy() 
        self.v_p = self.v.copy() 
        self.gap_p = self.gap.copy()
        self.gap_lookahead_p = self.gap_lookahead.copy()    

        #############################Adam#############################
        # c = (1 - self.Beta2 ** (i+1))**0.5 / (1 - self.Beta1 ** (i+1))
        # c = 1
        self.m = self.Beta1 * self.m + (1 - self.Beta1) * (self.dLdu)
        # self.v = np.maximum(self.Beta2 * self.v + (1 - self.Beta2) * (self.dLdu ** 2), self.v) #AMSgrad
        self.v = self.Beta2 * self.v + (1 - self.Beta2) * (self.dLdu ** 2) # ADAM
        # self.v += self.dLdu ** 2 # AdaGrad
        if self.bias_correction:
            self.m_corrected = self.m / (1 - self.Beta1 ** (i + 2))
            self.v_corrected = self.v / (1 - self.Beta2 ** (i + 2))
        else:
            self.m_corrected = self.m
            self.v_corrected = self.v
        us = np.array(self.us)
        self.update_u = -self.m_corrected / (np.sqrt(self.v_corrected))
        
        ##########################Nesterov accelerated gradient############
        # self.tau = 0.8
        # self.m = self.us - self.alpha * self.dJdu
        # self.us_try = (1-self.tau) * self.m + self.tau * us
        # self.update_u = self.us_try - us
        
        us_try = us + (alpha * self.update_u)
        self.us_try = list(us_try)
        # us_try = us + (alpha * c * self.update_u)
        # self.us_try = list(us_try)
        self.p = self.dLdu / (np.sqrt(self.v_corrected))

        self.update_u_lookahead = -self.p
        us_lookahead = us + (alpha * self.update_u_lookahead)
        self.us_lookahead = list(us_lookahead)



        self.cost_try = 0.
        self.cost_lookahead = 0.
        self.curvature_0 = 0.
        self.K = 0.


        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.K += self.dLdu[t, :].T @ self.p[t, :]
            self.curvature_0 += self.dLdu[t, :].T @ self.update_u[t, :]

            self.update_x_lookahead[t+1] = (data.Fx @ self.update_x_lookahead[t] + data.Fu @ (self.alpha * self.update_u_lookahead[t]) + self.gap_lookahead[t+1])
            self.xs_lookahead[t+1] = self.xs[t+1] + self.update_x_lookahead[t+1]

            model.calc(data, self.xs_lookahead[t], self.us_lookahead[t])
            self.cost_lookahead += data.cost
            self.gap_lookahead[t+1] = data.xnext - self.xs_lookahead[t+1]
            
        
        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_lookahead[-1])

        

        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
        
            for i in range(model.nu):
                # pdb.set_trace()
                if self.us_try[t][i] > self.ub[i]:
                    self.update_u[t][i] = self.ub[i] - self.us_try[t][i]
                    self.us_try[t][i] = self.ub[i]
                    # pdb.set_trace()
                    

                elif self.us_try[t][i] < self.lb[i]:
                    self.update_u[t][i] = self.lb[i] - self.us_try[t][i]
                    self.us_try[t][i] = self.lb[i]
                    # pdb.set_trace()

            self.update_x[t+1] = (data.Fx @ self.update_x[t] + data.Fu @ (self.alpha * self.update_u[t]) + self.gap[t+1])
            self.xs_try[t+1] = self.xs[t+1] + self.update_x[t+1]
            model.calc(data, self.xs_try[t], self.us_try[t])
            self.gap[t+1] = data.xnext - self.xs_try[t+1]
            self.cost_try += data.cost
            self.cost_try += (self.rho[t]/2) * (self.h_try[t] +  self.lambdas[t] / (self.rho[t])).T @ (self.h_try[t] +  self.lambdas[t] / (self.rho[t]))

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])

        self.cost_try += self.problem.terminalData.cost

        violated = (self.problem.terminalModel.g_ub < self.problem.terminalData.g).reshape((3,1))
        
        self.cost_try += (self.rho[-1]/2) * (self.h_try[-1] +  self.lambdas[-1] / (self.rho[-1])).T @ (self.h_try[-1] +  self.lambdas[-1] / (self.rho[-1]))

        return self.cost_try

    def computeGap(self):

        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):

            model.calc(data, self.xs[t], self.us[t])
            self.gap[t+1] = data.xnext - self.xs[t+1]
            

    def tryStep(self, alpha, i):
        self.direction_p = self.direction
        self.cost_try = self.forwardPass(alpha, i)
        
        return self.cost - self.cost_try
    
    def getCost(self):
        xs_temp = self.problem.rollout(self.us)
        # self.setCandidate(xs_temp, self.us)
        cost = self.problem.calc(xs_temp, self.us)
        return cost

    def getValue(self):
        xs_temp = self.problem.rollout(self.us)
        value = self.problem.calc(xs_temp, self.us)
        for t in range(len(self.lambdas)):
            value += (self.rho[t]/2) * (self.h[t, :] + self.lambdas[t, :] / (self.rho[t])).T @ (self.h[t, :] + self.lambdas[t, :] / (self.rho[t]))
        return value

    def checkUpdating(self, j):
        decay = 1.0
        # print('updating') 
        self.V = 0.

        for t in range(len(self.lambdas)):    
            self.V += np.linalg.norm(np.maximum(np.array(self.h[t]), -self.lambdas[t] / self.rho[t]), 1)

        if self.V <= 1.0 * self.V_p:
            if min(self.rho)> 1.: 
                self.rho *= 1.

        else:  
            if min(self.rho)< 1e3: 
                self.rho *= 2.

        for t in range(len(self.lambdas)):
            self.lambdas[t] += self.rho[t] * (self.h[t])

        

        self.m = decay * self.m
        self.m_p = decay * self.m_p

        self.v = (decay**2) * self.v
        self.v_p = (decay**2) * self.v_p    
        
        self.V_p = self.V.copy()
        

    def getCost_try(self):
        xs_temp = self.problem.rollout(self.us_try)
        # self.setCandidate(xs_temp, self.us)
        cost = self.problem.calc(xs_temp, self.us)
        return cost

    def solve(self, init_xs=None, init_us=None, maxIter=100, isFeasible=False):
        #___________________ Initialize ___________________#
        if init_xs is None:
            init_xs = [np.zeros(m.state.nx) for m in self.models()]
        if init_us is None:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels]

        init_xs[0][:] = self.problem.x0.copy()  # Initial condition guess must be x0
        self.xs_try[0][:] = self.problem.x0.copy()
        self.xs_lookahead[0][:] = self.problem.x0.copy()

        init_xs = self.problem.rollout(init_us)

        self.setCandidate(init_xs, init_us, False)

        self.computeGap()
        # pdb.set_trace()
        if self.refresh:
            self.refresh_()
        else:
            self.warmStart_()

        self.cost = self.calc()  # self.forwardPass(1.)  # compute initial value for merit function
        self.costs.append(self.cost)
        # print("initial cost is %s" % self.cost)

        for j in range(maxIter):
            # self.m *= 0.
            # self.v *= 0.
            i = 0
            while True:
                i+=1
                self.numIter = i
                # self.guess = 16.0 
                self.alpha = .25
                recalc = True  # this will recalculate derivatives in computeDirection
                while True:  # backward pass                            
                    try:    
                        self.computeDirection(recalc=recalc)

                    except:
                        print('In', i, 'th iteration.')
                        # pdb.set_trace()
                        raise BaseException("Backward Pass Failed")
                    break

                if self.kkt < self.th_stop:
                    print('Converged')
                    return True

                # print('GAP:', np.linalg.norm(self.gap, 2))
                
                while True:  # doing line search
                    while True:  # forward pass
                        try:
                            self.tryStep(self.alpha, i)
                        except:
                            # repeat starting from a smaller alpha
                            print("Try Step Failed for alpha = %s" % self.alpha)
                            raise BaseException('FP failed')
                        break



                    # Goldstein conditions
                    # self.c = 0.25
                    # ub = self.cost + self.c * self.alpha * self.curvature_0 
                    # lb = self.cost + (1 - self.c) * self.alpha * self.curvature_0 
                    # # print(f'cost_try = {self.cost_try}, cost = {self.getCost()}')
                    # # print(f'lb = {lb}, ub = {ub}')
                    # if  lb <= self.cost_try <= ub:

                    # Zhang & Hager line search
                    # if self.cost_try <= self.C + self.delta * self.alpha * self.curvature_0:

                    
                    # Wolfe conditions
                    # print(f'cost_try = {self.cost_try}, cost = {self.getValue()}')

                    
                    c = 0.1

                    # print('alpha is %s' % self.alpha)
                    # print("cost_try is %s" % self.cost_try)

                    # print("cost is %s" % self.cost)
                    # print("cost_lookahead is %s" % self.cost_lookahead)
                    # # print('K is %s' % self.K)
                    # print('update_u_lookahead is %s' % np.linalg.norm(self.update_u_lookahead, 2))
                    # print('update_u is %s' % np.linalg.norm(self.update_u, 2))
                    # print("RHS is %s\n" % (self.cost - c * self.alpha * self.K))
                    
                    # if np.isnan(self.cost_lookahead):
                    #     pdb.set_trace()

                    # if self.cost_lookahead <= self.cost - c * self.alpha * self.K:
                    if True:

                    
                    # if self.cost_try <= self.getValue() + self.c1 * self.alpha * self.curvature_0:

                        # line search succeed -> exit
                    
                        self.lineSearch_fail.append(False)
                        self.setCandidate(self.xs_try, self.us_try, True)

                        self.dV = np.abs(self.cost - self.cost_try)
                        self.cost = self.cost_try
        
                        self.costs.append(self.getCost())
                        self.alpha_p = self.alpha
                        self.alphas.append(self.alpha)

                        self.curvatures.append(self.curvature_0)
                        self.step_norm.append(np.linalg.norm(self.update_u, ord=2))

                        self.us_p = self.us
                        self.Ks.append(self.K)
                        self.rhos.append(self.rho[0].copy())
                        self.s = self.alpha - self.alpha_p
                        self.y = self.dLdu - self.dLdu_p
                        # print(f'lb = {lb}, ub = {ub}')
                        # print(f'True cost:{self.getCost()}\n')

                        # Zhang & Hager line search
                        # self.C = (self.Eta * self.Q * self.C + self.cost_try) / self.Q
                        # self.Q = self.Eta * self.Q + 1
                        break

                    elif self.alpha <= .1:
                        # keep going anyway
                        # self.Infeasibilities.append(np.linalg.norm(self.gap, np.inf))
                        self.lineSearch_fail.append(True)
                        self.guess_accepted.append(False)
                        self.setCandidate(self.xs_try, self.us_try, True)

                        self.dV = self.cost - self.cost_try

                        self.cost = self.cost_try
                        self.costs.append(self.getCost())
                        self.alpha_p = self.alpha
                        self.fail_ls += 1
                        self.alphas.append(self.alpha)
                        # self.updates.append(np.linalg.norm(self.alpha * self.update, ord=2))
                        self.curvatures.append(self.curvature_0)
                        self.step_norm.append(np.linalg.norm(self.update_u, ord=2))
                        self.u_magnitude.append(np.linalg.norm(self.us, 2))
                        self.us_p = self.us
                        self.Ks.append(self.K)

                        self.s = self.alpha - self.alpha_p
                        self.y = self.dLdu - self.dLdu_p

                        # Zhang & Hager line search
                        # self.C = (self.Eta * self.Q * self.C + self.cost_try) / self.Q
                        # self.Q = self.Eta * self.Q + 1
                        # print(f'at {i} th iteration, Line search failed')
                        break

                    else:
                        # restore momentum terms
                        self.alpha *= 0.8
                        self.m = self.m_p.copy() 
                        self.v = self.v_p.copy() 
                        self.gap = self.gap_p.copy()
                        self.gap_lookahead = self.gap_lookahead_p.copy()


                # if self.gradient < 10 ** (-4):
                #     break

                if i >= 100:
                    break
            # print('______________________________')
            # print('Iteration:', j)
            # print('lambdas:', self.lambdas)
            # print('rho = ', self.rho)
            # print('h = ', self.h)
            # # if j == 4: pdb.set_trace()
            # print('V = ', self.V)
            # print('______________________________')
            self.checkUpdating(j)
            self.stoppingCriteria()
        return False

    def warmStart_(self):
        m = list(self.m[1:]) + [self.m[-1]]
        v = list(self.v[1:]) + [self.v[-1]]
        n = list(self.n[1:]) + [self.n[-1]]
        self.m = self.decay1 * np.array(m)
        self.v = self.decay2 * np.array(v)
        self.n = self.decay3 * np.array(n)
        self.dJdu = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdx = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.costs = []
        self.KKTs = []
        self.updates = []
        self.curvatures = []
        self.alphas = []
        self.lineSearch_fail = []
        self.guess_accepted = []

    def refresh_(self):
        self.m = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.v = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.n = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdu = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdx = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.costs = []
        self.KKTs = []
        self.updates = []
        self.curvatures = []
        self.alphas = []
        self.lineSearch_fail = []
        self.guess_accepted = []


    def stoppingCriteria(self):
        if self.dV < 1e-5:
            self.n_little_improvement += 1
            if VERBOSE: print('Little improvement.')
            return True

    def allocateData(self):
        self.Ks = []
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()]
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels]

        self.xs_lookahead = [np.zeros(m.state.nx) for m in self.models()]
        self.us_lookahead = [np.zeros(m.nu) for m in self.problem.runningModels]

        self.s = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])

        self.y = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])

        self.gap = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.gap_lookahead = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.update_u_lookahead = np.array([np.zeros(m.nu) for m in self.problem.runningModels])
        self.update_x_lookahead = np.array([np.zeros(m.state.ndx) for m in self.models()])

        self.update_u = np.array([np.zeros(m.nu) for m in self.problem.runningModels])
        self.update_x = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.Infeasibilities = []
        self.dJdu = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdx = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.dLdu = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dLdx = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.alpha_p = 0
        self.dJdu_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.direction = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.direction_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        
        # self.lambdas = np.array([np.ones([m.ng]) for m in self.models()]) * 0.
        self.lambdas = [np.zeros(m.ng) for m in self.models()]

        self.dHdu = [np.zeros([m.ng, m.nu]) for m in self.problem.runningModels]
        self.dHdx = [np.zeros([m.ng, m.state.ndx]) for m in self.models()]
        self.rho = np.ones([len(self.models()), 1]) * 10.0
        self.rhos = []
        # self.H = np.array([np.zeros([self.models()[0].ng])]).reshape(3,1)
        self.h = [np.zeros([m.ng]) for m in self.models()]
        self.g = [np.zeros([m.ng]) for m in self.models()]
        self.h_try = [np.zeros([m.ng]) for m in self.models()]
        self.h_lookahead = [np.zeros([m.ng]) for m in self.models()]
        self.V = np.zeros([len(self.models()), 1])
        self.inequality = []
        self.gradients = []
        self.numIter = 0
        self.costs = []
        self.kkt = 0.
        self.KKTs = []
        self.alpha = 1.
        self.curvature_0 = 0.
        self.alpha_p = 1.
        self.guess_accepted = []
        self.m = np.zeros_like(self.dJdu)
        self.v = np.zeros_like(self.dJdu)
        self.m_p = np.zeros_like(self.dJdu)
        self.v_p = np.zeros_like(self.dJdu)
        self.n = np.zeros_like(self.dJdu)
        self.u_magnitude = []
        self.Beta1 = .8
        self.Beta2 = .8
        self.Beta3 = .999
        self.eps = 1e-12

        self.us_p = self.us.copy()
        self.s = np.zeros_like(self.dJdu)
        self.V_p = 0
        self.kkt = 0.
        self.KKTs = []
        self.costs = []
        self.numIter = 0
        self.bias_correction = True
        self.refresh = False
        self.updates = []
        self.curvatures = []
        self.num_restart = 0
        self.fail_ls = 0
        self.decay1 = 1.
        self.decay2 = 1.
        self.decay3 = 1.
        self.lineSearch_fail = []
        self.alphas = []
        self.step_norm = []
        self.lb = np.ones([self.problem.terminalModel.nu]) * -5000.
        self.ub = np.ones([self.problem.terminalModel.nu]) * 5000.



