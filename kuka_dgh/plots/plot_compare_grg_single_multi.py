from plot_utils import SimpleDataPlotter
from mim_data_utils import DataReader
from croco_mpc_utils.pinocchio_utils import *
import numpy as np
import matplotlib.pyplot as plt 

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'kuka_dgh'
os.sys.path.insert(1, str(python_path))
from demos import launch_utils


from mim_robots.robot_loader import load_pinocchio_wrapper

pinrobot    = load_pinocchio_wrapper('iiwa')
model       = pinrobot.model
data        = model.createData()
frameId     = model.getFrameId('contact')
nq = model.nq ; nv = model.nv
# Overwrite effort limit as in DGM
model.effortLimit = np.array([100, 100, 50, 50, 20, 10, 10])




# Load config file
SIM           = True
EXP_NAME      = 'circle_grg' # <<<<<<<<<<<<< Choose experiment here (cf. launch_utils)
config        = launch_utils.load_config_file(EXP_NAME)


# Create data Plottger
s = SimpleDataPlotter()

if(SIM):
    data_path = '/tmp/'

    data_name = 'circle_grg_SIM_2024-03-18T14:31:23.780511_grg' #multi threading
    data_name3 = 'circle_grg_SIM_2024-03-18T14:33:14.930491_grg' #single threading

    


else:
    data_path = 'data/unconstrained/new/'
    data_name = 'circle_ssqp_REAL_2023-10-31T17:06:02.992743_fddp' 

r       = DataReader(data_path+data_name+'.mds')
r3     = DataReader(data_path+data_name3+'.mds')
# r1.data = {}
# r2.data = {}
# r3      = DataReader(data_path+data_name3+'.mds')
N       = r.data['absolute_time'].shape[0]
print("Total number of control cycles = ", N)
time_lin = np.linspace(0, N/config['ctrl_freq'], N)


fig, ax = plt.subplots(4, 1, sharex='col') 
ax[0].plot(r.data['KKT'], label='KKT residual GRG')
ax[0].plot(r3.data['KKT'], label='KKT residual grg_single_thread')
ax[0].plot(N*[config['solver_termination_tolerance']], label= 'KKT residual tolerance', color = 'r')

ax[1].plot(r.data['ddp_iter'], label='# solver iterations GRG')
ax[1].plot(r3.data['ddp_iter'], label='# solver iterations grg_single_thread')

ax[2].plot(r.data['t_child']*1000, label='OCP solve time GRG')
ax[2].plot(r3.data['t_child']*1000, label='OCP solve time grg_single_thread')
ax[2].plot(N*[1000./config['ctrl_freq']], label= 'dt_MPC', color='r')

ax[3].plot((r.data['timing_control'])* 1000, label='Control cycle time GRG' )
ax[3].plot((r3.data['timing_control'])* 1000, label='Control cycle time grg_single_thread' )

ax[3].plot(N*[1000./config['ctrl_freq']], label= 'dt_MPC', color='r')
for i in range(4):
    ax[i].grid()
    ax[i].legend()


s.plot_joint_pos( [r.data['joint_positions'], 
                   r3.data['joint_positions']], 
                   ['mea_grg',
                    'single_thread'], 
                   ['r', 
                    'b'],
                   ylims=[model.lowerPositionLimit, model.upperPositionLimit] )

# s.plot_joint_vel( [r.data['joint_velocities'], r.data['x_des'][:,nq:nq+nv]], # r.data['x'][:,nq:nq+nv], r.data['x1'][:,nq:nq+nv]],
#                   ['mea', 'pred'], # 'pred0', 'pred1'], 
#                   ['r', 'b'], #[0.2, 0.2, 0.2, 0.5], 'b', 'g']) 
#                   ylims=[-model.velocityLimit, +model.velocityLimit] )

# For SIM robot only
if(SIM):
    s.plot_joint_tau( [r.data['tau'], 
                       r3.data['tau_ff']],
                      ['total grg', 
                       'total grg_single_thread',
                       'ff'], 
                      ['r', 
                       'g', 
                       'y',
                       'b', 
                       [0.2, 0.2, 0.2, 0.5]],
                      ylims=[-model.effortLimit, +model.effortLimit] )
# For REAL robot only !! DEFINITIVE FORMULA !!
else:
    # Our self.tau was subtracted gravity, so we add it again
    # joint_torques_measured DOES include the gravity torque from KUKA
    # There is a sign mismatch in the axis so we use a minus sign
    s.plot_joint_tau( [-r.data['joint_cmd_torques'], 
                       r.data['joint_torques_measured'], 
                       r.data['tau'] + r.data['tau_gravity'], 
                       r.data['tau_ff'] + r.data['tau_gravity']], 
                  ['-cmd (FRI)', 
                   'Measured', 
                   'Desired (sent to robot) [+g(q)]', 
                   'tau_ff (OCP solution) [+g(q)]', 
                   'Measured - EXT'], 
                  ['k', 'r', 'b', 'g', 'y'],
                  ylims=[-model.effortLimit, +model.effortLimit],
                  linestyle=['dotted', 'solid', 'solid', 'solid', 'solid'])

p_mea = get_p_(r.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_mea3 = get_p_(r3.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))

if(EXP_NAME == 'reach_ssqp'):
    target_position = r.data['target_position'] #np.zeros((N,3))
else:
    target_position = np.zeros((N,3))
    target_position[:,0] = r.data['target_position_x'][:,0]
    target_position[:,1] = r.data['target_position_y'][:,0]
    target_position[:,2] = r.data['target_position_z'][:,0]
s.plot_ee_pos( [p_mea, 
                p_mea3,
                target_position],  
               ['mea_grg',
                'mea_grg_single_thread',
                'ref (position cost)'], 
               ['r',  
                'blue',
                'k'], 
               linestyle=['solid', 'solid', 'dotted'])




# Compute the total cost of the experiment 
state_cost_list       = []
tau_cost_list         = []
translation_cost_list = []
total_cost_list       = []
N_START = int(config['T_CIRCLE']*config['ctrl_freq'])
for index in range(N_START, N):
    state_mea = np.concatenate([r.data['joint_positions'][index,:], r.data['joint_velocities'][index,:]])
    tau_mea   = r.data['tau_ff'][index, :]
    state_ref = np.array([0., 1.0471975511965976, 0., -1.1344640137963142, 0.2,  0.7853981633974483, 0, 0.,0.,0.,0.,0.,0.,0.])
    tau_ref   = r.data['tau_gravity'][index,:]
    
    state_cost = 0.5 * config['stateRegWeight'] * (state_mea - state_ref).T @ np.diag(config['stateRegWeights'])**2 @ (state_mea - state_ref)
    state_cost_list.append(state_cost)

    tau_cost = 0.5 * config['ctrlRegGravWeight'] * (tau_mea - tau_ref).T @ np.diag(config['ctrlRegGravWeights'])**2 @ (tau_mea - tau_ref)
    tau_cost_list.append(tau_cost)

    translation_cost = 0.5 * config['frameTranslationWeight'] * (p_mea[index, :] - target_position[index, :]).T @ np.diag(config['frameTranslationWeights'])**2 @ (p_mea[index, :] - target_position[index, :])
    translation_cost_list.append(translation_cost)
    
    total_cost = state_cost + tau_cost + translation_cost
    total_cost_list.append(total_cost)
state_cost_       = np.array(state_cost_list).reshape(-1, 1)
tau_cost_         = np.array(tau_cost_list).reshape(-1, 1)
translation_cost_ = np.array(translation_cost_list).reshape(-1, 1)
total_cost_       = np.array(total_cost_list).reshape(-1, 1)



ANIMATION = True
LINEWIDTH = 6
ALPHA = 0.8
 

print("PLOTTING")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex='col', figsize=(55, 13.5))
 

ax1.grid(linewidth=1) 
ax2.grid(linewidth=1) 
ax3.grid(linewidth=1) 
ax4.grid(linewidth=1) 

ax1.set_xlim(time_lin[0], time_lin[-1])
ax2.set_xlim(time_lin[0], time_lin[-1])
ax3.set_xlim(time_lin[0], time_lin[-1])
ax4.set_xlim(time_lin[0], time_lin[-1])

# ax1.set_ylim(0., 0.7)
# ax2.set_ylim(0., 1.1)
# ax3.set_ylim(0., 1.6)
   
ax1.set_ylabel('State cost ', fontsize=20)
ax2.set_ylabel('Contrl cost ', fontsize=20)
ax3.set_ylabel('Translation cost ' , fontsize=20)
ax4.set_ylabel('Total cost' , fontsize=20)
ax4.set_xlabel('Time (s)', fontsize=20)

ax1.tick_params(axis = 'y', labelsize=38)
ax2.tick_params(axis = 'y', labelsize=38)
ax3.tick_params(axis = 'y', labelsize=38)
ax4.tick_params(axis = 'y', labelsize=38)
ax4.tick_params(axis = 'x', labelsize=38)

ax1.tick_params(labelbottom=False)  
ax2.tick_params(labelbottom=False)  
ax3.tick_params(labelbottom=False)  

time_lin2 = time_lin[N_START:N]
ax1.plot(time_lin2, state_cost_,  linewidth=LINEWIDTH, label='grg')
ax2.plot(time_lin2, tau_cost_, linewidth=LINEWIDTH, label='grg')
ax3.plot(time_lin2, translation_cost_, linewidth=LINEWIDTH, label='grg')
ax4.plot(time_lin2, total_cost_, linewidth=LINEWIDTH, label='grg')

state_cost_list       = []
tau_cost_list         = []
translation_cost_list = []
total_cost_list       = []
N_START = int(config['T_CIRCLE']*config['ctrl_freq'])
for index in range(N_START, N):
    state_mea = np.concatenate([r3.data['joint_positions'][index,:], r3.data['joint_velocities'][index,:]])
    tau_mea   = r3.data['tau_ff'][index, :]
    state_ref = np.array([0., 1.0471975511965976, 0., -1.1344640137963142, 0.2,  0.7853981633974483, 0, 0.,0.,0.,0.,0.,0.,0.])
    tau_ref   = r3.data['tau_gravity'][index,:]
    
    state_cost = 0.5 * config['stateRegWeight'] * (state_mea - state_ref).T @ np.diag(config['stateRegWeights'])**2 @ (state_mea - state_ref)
    state_cost_list.append(state_cost)

    tau_cost = 0.5 * config['ctrlRegGravWeight'] * (tau_mea - tau_ref).T @ np.diag(config['ctrlRegGravWeights'])**2 @ (tau_mea - tau_ref)
    tau_cost_list.append(tau_cost)

    translation_cost = 0.5 * config['frameTranslationWeight'] * (p_mea3[index, :] - target_position[index, :]).T @ np.diag(config['frameTranslationWeights'])**2 @ (p_mea3[index, :] - target_position[index, :])
    translation_cost_list.append(translation_cost)
    
    total_cost = state_cost + tau_cost + translation_cost
    total_cost_list.append(total_cost)
state_cost_       = np.array(state_cost_list).reshape(-1, 1)
tau_cost_         = np.array(tau_cost_list).reshape(-1, 1)
translation_cost_ = np.array(translation_cost_list).reshape(-1, 1)
total_cost_       = np.array(total_cost_list).reshape(-1, 1)

time_lin2 = time_lin[N_START:N]
ax1.plot(time_lin2, state_cost_,  linewidth=LINEWIDTH, label='grg_single_thread')
ax2.plot(time_lin2, tau_cost_, linewidth=LINEWIDTH, label='grg_single_thread')
ax3.plot(time_lin2, translation_cost_, linewidth=LINEWIDTH, label='grg_single_thread')
ax4.plot(time_lin2, total_cost_, linewidth=LINEWIDTH, label='grg_single_thread')



fig.legend()
# print("Cumulative cost (log)      = ", np.sum(r.data['cost'][N_START:N]))
# print("Cumulative cost of the MPC = ", np.sum(total_cost_))
plt.show()






