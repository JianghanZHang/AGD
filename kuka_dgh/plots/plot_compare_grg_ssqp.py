from plot_utils import SimpleDataPlotter
from mim_data_utils import DataReader
from croco_mpc_utils.pinocchio_utils import *
import numpy as np
import matplotlib.pyplot as plt 
import os 
os.sys.path.insert(1, '../')
from demos import launch_utils


from robot_properties_kuka.config import IiwaConfig

iiwa_config = IiwaConfig()
pinrobot    = iiwa_config.buildRobotWrapper()
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
    
    #T = 10 @100HZ
    # data_name = 'circle_grg_SIM_2024-03-05T17:02:19.278784_grg_CODE_SPRINT'
    # data_name1 = 'circle_grg_SIM_2024-03-01T14:09:07.930336_sqp_CODE_SPRINT'
    # data_name2 = 'circle_grg_SIM_2024-03-01T13:57:35.403756_ddp_CODE_SPRINT'
    # data_name3 = 'circle_grg_SIM_2024-03-05T17:02:48.793888_grg_h_CODE_SPRINT'

    #T = 20 @100HZ
    # data_name = 'circle_grg_SIM_2024-03-04T18:16:16.567219_grg_CODE_SPRINT'
    # data_name1 = 'circle_grg_SIM_2024-03-01T14:08:23.138874_sqp_CODE_SPRINT'
    # data_name2 = 'circle_grg_SIM_2024-03-01T14:11:48.881412_ddp_CODE_SPRINT'

    #T = 30 @100HZ
    # data_name3 = 'circle_grg_SIM_2024-03-05T16:20:45.916532_grg_h_CODE_SPRINT'
    # data_name = 'circle_grg_SIM_2024-03-04T18:17:23.242381_grg_CODE_SPRINT'
    # data_name1 = 'circle_grg_SIM_2024-03-01T14:18:24.556685_sqp_CODE_SPRINT'
    # data_name2 = 'circle_grg_SIM_2024-03-01T14:15:06.186234_ddp_CODE_SPRINT'

    #T = 40 @100HZ
    # data_name = 'circle_grg_SIM_2024-03-04T18:17:52.883502_grg_CODE_SPRINT'
    # data_name1 = 'circle_grg_SIM_2024-03-01T14:21:49.120390_sqp_CODE_SPRINT'
    # data_name2 = 'circle_grg_SIM_2024-03-01T14:23:55.574577_ddp_CODE_SPRINT'

    #T = 50 @100HZ
    # data_name = 'circle_grg_SIM_2024-03-04T18:18:29.763694_grg_CODE_SPRINT'
    # data_name1 = 'circle_grg_SIM_2024-03-04T18:19:08.448627_sqp_CODE_SPRINT'
    # data_name2 = 'circle_grg_SIM_2024-03-04T18:20:09.840563_ddp_CODE_SPRINT'
    # data_name3 = 'circle_grg_SIM_2024-03-05T17:05:42.964743_grg_h_CODE_SPRINT'

    #T = 20 @1000HZ
    data_name = 'circle_grg_SIM_2024-03-05T17:14:17.970712_grg_CODE_SPRINT'
    data_name1 = 'circle_grg_SIM_2024-03-05T17:11:19.106560_sqp_CODE_SPRINT'
    data_name2 =  'circle_grg_SIM_2024-03-05T17:10:26.681366_ddp_CODE_SPRINT'
    data_name3 = 'circle_grg_SIM_2024-03-05T17:08:45.175145_grg_h_CODE_SPRINT'


else:
    data_path = 'data/unconstrained/new/'
    data_name = 'circle_ssqp_REAL_2023-10-31T17:06:02.992743_fddp' 
    # data_name = 'circle_ssqp_REAL_2023-10-31T16:45:47.050199_sqp' 

r       = DataReader(data_path+data_name+'.mds')
r1      = DataReader(data_path+data_name1+'.mds')
r2      = DataReader(data_path+data_name2+'.mds')
r3      = DataReader(data_path+data_name3+'.mds')
N       = r.data['absolute_time'].shape[0]
print("Total number of control cycles = ", N)
time_lin = np.linspace(0, N/config['ctrl_freq'], N)


fig, ax = plt.subplots(4, 1, sharex='col') 
ax[0].plot(r.data['KKT'], label='KKT residual GRG')
ax[0].plot(r1.data['KKT'], label='KKT residual SSQP')
ax[0].plot(r2.data['KKT'], label='KKT residual DDP')
ax[0].plot(r3.data['KKT'], label='KKT residual GRG_H')
ax[0].plot(N*[config['solver_termination_tolerance']], label= 'KKT residual tolerance', color = 'r')

ax[1].plot(r.data['ddp_iter'], label='# solver iterations GRG')
ax[1].plot(r1.data['ddp_iter'], label='# solver iterations SSQP')
ax[1].plot(r2.data['ddp_iter'], label='# solver iterations DDP')
ax[1].plot(r3.data['ddp_iter'], label='# solver iterations GRG_H')

ax[2].plot(r.data['t_child']*1000, label='OCP solve time GRG')
ax[2].plot(r1.data['t_child']*1000, label='OCP solve time SSQP')
ax[2].plot(r2.data['t_child']*1000, label='OCP solve time DDP')
ax[2].plot(r3.data['t_child']*1000, label='OCP solve time GRG_H')
ax[2].plot(N*[1000./config['ctrl_freq']], label= 'dt_MPC', color='r')

ax[3].plot((r.data['timing_control'])* 1000, label='Control cycle time GRG' )
ax[3].plot((r1.data['timing_control'])* 1000, label='Control cycle time SSQP' )
ax[3].plot((r2.data['timing_control'])* 1000, label='Control cycle time DDP' )
ax[3].plot((r3.data['timing_control'])* 1000, label='Control cycle time GRG_H' )

ax[3].plot(N*[1000./config['ctrl_freq']], label= 'dt_MPC', color='r')
for i in range(4):
    ax[i].grid()
    ax[i].legend()


s.plot_joint_pos( [r.data['joint_positions'], 
                   r1.data['joint_positions'],
                   r2.data['joint_positions'],
                   r1.data['x_des'][:,:nq]], 
                   ['mea_grg',
                    'mea_ssqp',
                    'mea_ddp',  
                    'pred'], 
                   ['r', 
                    'g',
                    'y',
                    'b'],
                   ylims=[model.lowerPositionLimit, model.upperPositionLimit] )

# s.plot_joint_vel( [r.data['joint_velocities'], r.data['x_des'][:,nq:nq+nv]], # r.data['x'][:,nq:nq+nv], r.data['x1'][:,nq:nq+nv]],
#                   ['mea', 'pred'], # 'pred0', 'pred1'], 
#                   ['r', 'b'], #[0.2, 0.2, 0.2, 0.5], 'b', 'g']) 
#                   ylims=[-model.velocityLimit, +model.velocityLimit] )

# For SIM robot only
if(SIM):
    s.plot_joint_tau( [r.data['tau'], 
                       r1.data['tau'],
                       r2.data['tau'],
                       r1.data['tau_ff']],
                    #    r.data['tau_riccati'], 
                    #    r.data['tau_gravity']], 
                      ['total grg', 
                       'total ssqp',
                       'total ddp',
                       'ff'], 
                    #    'riccati', 
                    #    'gravity'], 
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
p_mea1 = get_p_(r1.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_mea2 = get_p_(r2.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_mea3 = get_p_(r3.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_des = get_p_(r.data['x_des'][:,:nq], pinrobot.model, pinrobot.model.getFrameId('contact'))

if(EXP_NAME == 'reach_ssqp'):
    target_position = r.data['target_position'] #np.zeros((N,3))
else:
    target_position = np.zeros((N,3))
    target_position[:,0] = r.data['target_position_x'][:,0]
    target_position[:,1] = r.data['target_position_y'][:,0]
    target_position[:,2] = r.data['target_position_z'][:,0]
s.plot_ee_pos( [p_mea, 
                p_mea1,
                p_mea2,
                p_mea3,
                target_position],  
               ['mea_grg',
                'mea_ssqp', 
                'mea_ddp',
                'mea_grg_h',
                'ref (position cost)'], 
               ['r',  
                'g', 
                'yellow',
                'blue',
                'k'], 
               linestyle=['solid', 'solid', 'solid', 'solid', 'dotted'])




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
    state_mea = np.concatenate([r1.data['joint_positions'][index,:], r1.data['joint_velocities'][index,:]])
    tau_mea   = r1.data['tau_ff'][index, :]
    state_ref = np.array([0., 1.0471975511965976, 0., -1.1344640137963142, 0.2,  0.7853981633974483, 0, 0.,0.,0.,0.,0.,0.,0.])
    tau_ref   = r1.data['tau_gravity'][index,:]
    
    state_cost = 0.5 * config['stateRegWeight'] * (state_mea - state_ref).T @ np.diag(config['stateRegWeights'])**2 @ (state_mea - state_ref)
    state_cost_list.append(state_cost)

    tau_cost = 0.5 * config['ctrlRegGravWeight'] * (tau_mea - tau_ref).T @ np.diag(config['ctrlRegGravWeights'])**2 @ (tau_mea - tau_ref)
    tau_cost_list.append(tau_cost)

    translation_cost = 0.5 * config['frameTranslationWeight'] * (p_mea1[index, :] - target_position[index, :]).T @ np.diag(config['frameTranslationWeights'])**2 @ (p_mea1[index, :] - target_position[index, :])
    translation_cost_list.append(translation_cost)
    
    total_cost = state_cost + tau_cost + translation_cost
    total_cost_list.append(total_cost)
state_cost_       = np.array(state_cost_list).reshape(-1, 1)
tau_cost_         = np.array(tau_cost_list).reshape(-1, 1)
translation_cost_ = np.array(translation_cost_list).reshape(-1, 1)
total_cost_       = np.array(total_cost_list).reshape(-1, 1)

time_lin2 = time_lin[N_START:N]
ax1.plot(time_lin2, state_cost_,  linewidth=LINEWIDTH, label='ssqp')
ax2.plot(time_lin2, tau_cost_, linewidth=LINEWIDTH, label='ssqp')
ax3.plot(time_lin2, translation_cost_, linewidth=LINEWIDTH, label='ssqp')
ax4.plot(time_lin2, total_cost_, linewidth=LINEWIDTH, label='ssqp')

state_cost_list       = []
tau_cost_list         = []
translation_cost_list = []
total_cost_list       = []
N_START = int(config['T_CIRCLE']*config['ctrl_freq'])
for index in range(N_START, N):
    state_mea = np.concatenate([r2.data['joint_positions'][index,:], r2.data['joint_velocities'][index,:]])
    tau_mea   = r2.data['tau_ff'][index, :]
    state_ref = np.array([0., 1.0471975511965976, 0., -1.1344640137963142, 0.2,  0.7853981633974483, 0, 0.,0.,0.,0.,0.,0.,0.])
    tau_ref   = r2.data['tau_gravity'][index,:]
    
    state_cost = 0.5 * config['stateRegWeight'] * (state_mea - state_ref).T @ np.diag(config['stateRegWeights'])**2 @ (state_mea - state_ref)
    state_cost_list.append(state_cost)

    tau_cost = 0.5 * config['ctrlRegGravWeight'] * (tau_mea - tau_ref).T @ np.diag(config['ctrlRegGravWeights'])**2 @ (tau_mea - tau_ref)
    tau_cost_list.append(tau_cost)

    translation_cost = 0.5 * config['frameTranslationWeight'] * (p_mea2[index, :] - target_position[index, :]).T @ np.diag(config['frameTranslationWeights'])**2 @ (p_mea2[index, :] - target_position[index, :])
    translation_cost_list.append(translation_cost)
    
    total_cost = state_cost + tau_cost + translation_cost
    total_cost_list.append(total_cost)
state_cost_       = np.array(state_cost_list).reshape(-1, 1)
tau_cost_         = np.array(tau_cost_list).reshape(-1, 1)
translation_cost_ = np.array(translation_cost_list).reshape(-1, 1)
total_cost_       = np.array(total_cost_list).reshape(-1, 1)

time_lin2 = time_lin[N_START:N]
ax1.plot(time_lin2, state_cost_,  linewidth=LINEWIDTH, label='ddp')
ax2.plot(time_lin2, tau_cost_, linewidth=LINEWIDTH, label='ddp')
ax3.plot(time_lin2, translation_cost_, linewidth=LINEWIDTH, label='ddp')
ax4.plot(time_lin2, total_cost_, linewidth=LINEWIDTH, label='ddp')

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
ax1.plot(time_lin2, state_cost_,  linewidth=LINEWIDTH, label='grg_h')
ax2.plot(time_lin2, tau_cost_, linewidth=LINEWIDTH, label='grg_h')
ax3.plot(time_lin2, translation_cost_, linewidth=LINEWIDTH, label='grg_h')
ax4.plot(time_lin2, total_cost_, linewidth=LINEWIDTH, label='grg_h')



fig.legend()
# print("Cumulative cost (log)      = ", np.sum(r.data['cost'][N_START:N]))
# print("Cumulative cost of the MPC = ", np.sum(total_cost_))
plt.show()






