from plot_utils import SimpleDataPlotter
from mim_data_utils import DataReader
from croco_mpc_utils.pinocchio_utils import *
import numpy as np
import matplotlib
import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'kuka_dgh'
os.sys.path.insert(1, str(python_path))
from demos import launch_utils


from mim_robots.robot_loader import load_pinocchio_wrapper

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
import matplotlib.pyplot as plt 


pinrobot    = load_pinocchio_wrapper('iiwa')
model       = pinrobot.model
data        = model.createData()
frameId     = model.getFrameId('contact')
nq = model.nq ; nv = model.nv
# Overwrite effort limit as in DGM
model.effortLimit = np.array([100, 100, 50, 50, 20, 10, 10])




# Load config file
SIM           = False
EXP_NAME      = 'circle_grg' # <<<<<<<<<<<<< Choose experiment here (cf. launch_utils)
config        = launch_utils.load_config_file(EXP_NAME)


# Create data Plottger
s = SimpleDataPlotter()

if(SIM):
    data_path = '/tmp/'

    data_name = 'long_experiment_GD_iter8' #multi threading
    data_name3 = 'long_experiment_DDP_iter3' #single threading

    


else:
    data_path = '/home/jianghan/data/grg/'
    data_name = 'long_experiment_GD_iter8' 
    data_name3 = 'long_experiment_DDP_iter3'

r       = DataReader(data_path+data_name+'.mds')
r3     = DataReader(data_path+data_name3+'.mds')

N       = r.data['absolute_time'].shape[0]
START = 2000
STOP = 17001
print("Total number of control cycles = ", N)
time_lin = np.linspace(0, N/config['ctrl_freq'], N)

p_mea = get_p_(r.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_mea3 = get_p_(r3.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))

if(EXP_NAME == 'reach_ssqp'):
    target_position = r.data['target_position'] #np.zeros((N,3))
else:
    target_position = np.zeros((N,3))
    target_position[:,0] = r.data['target_position_x'][:,0]
    target_position[:,1] = r.data['target_position_y'][:,0]
    target_position[:,2] = r.data['target_position_z'][:,0]
fig2, (ax1, ax2 ,ax3) = s.plot_ee_pos( [p_mea[START:STOP,], 
                p_mea3[START:STOP,],
                target_position[START:STOP,]],  
               ['mea_agd',
                'mea_ddp',
                'ref (position cost)'], 
               ['r',  
                'blue',
                'k'], 
               linestyle=['solid', 'solid', 'dotted'])


fig2.savefig('/home/jianghan/Devel/workspace/src/GRG/kuka_dgh/plots/plots/comparing_position.pdf', bbox_inches="tight")


# Compute the total cost of the experiment 
state_cost_list       = []
tau_cost_list         = []
translation_cost_list = []
total_cost_list       = []
for index in range(START, STOP):
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
LINEWIDTH = 4
ALPHA = 0.8
 
time_lin2 = time_lin[0:(STOP-START)]
print("PLOTTING")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex='col', figsize=(55, 13.5))

ax1.grid(linewidth=1) 
ax2.grid(linewidth=1) 
ax3.grid(linewidth=1) 
ax4.grid(linewidth=1) 

ax1.set_xlim(time_lin2[0], time_lin2[-1])
ax2.set_xlim(time_lin2[0], time_lin2[-1])
ax3.set_xlim(time_lin2[0], time_lin2[-1])
ax4.set_xlim(time_lin2[0], time_lin2[-1])

ax1.set_ylabel('State cost ', fontsize=20)
ax2.set_ylabel('Contrl cost ', fontsize=20)
ax3.set_ylabel('Translation cost ' , fontsize=20)
ax4.set_ylabel('Total cost' , fontsize=20)
ax4.set_xlabel('Time (s)', fontsize=20)

ax1.tick_params(axis = 'y', labelsize=38, labelleft=True)
ax2.tick_params(axis = 'y', labelsize=38, labelleft=True)
ax3.tick_params(axis = 'y', labelsize=38, labelleft=True)
ax4.tick_params(axis = 'y', labelsize=38, labelleft=True)
ax4.tick_params(axis = 'x', labelsize=38)

ax1.tick_params(labelbottom=False)  
ax2.tick_params(labelbottom=False)  
ax3.tick_params(labelbottom=False)  

ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

N_START=0


ax1.plot(time_lin2, state_cost_,  linewidth=LINEWIDTH, label='agd', alpha=0.8)
ax2.plot(time_lin2, tau_cost_, linewidth=LINEWIDTH, alpha=0.8)
ax3.plot(time_lin2, translation_cost_, linewidth=LINEWIDTH, alpha=0.8)
ax4.plot(time_lin2, total_cost_, linewidth=LINEWIDTH, alpha=0.8)


fig1, (ax) = plt.subplots(1, 1, figsize=(55, 13.5))

ax.grid(linewidth=1) 

ax.set_xlim(time_lin2[0], time_lin2[-1])

ax.set_ylabel('State cost ', fontsize=20)

ax.set_xlabel('Time (s)', fontsize=20)

ax.tick_params(axis = 'y', labelsize=38, labelleft=True)

ax.tick_params(axis = 'x', labelsize=38)

# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax.plot(time_lin2, total_cost_,  linewidth=LINEWIDTH, label='agd', alpha=0.8)



state_cost_list       = []
tau_cost_list         = []
translation_cost_list = []
total_cost_list       = []
# N_START = int(config['T_CIRCLE']*config['ctrl_freq'])

# import pdb; pdb.set_trace()
for index in range(START, STOP):
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

time_lin2 = time_lin[N_START:(STOP-START)]
ax1.plot(time_lin2, state_cost_,  linewidth=LINEWIDTH, label='ddp')
ax2.plot(time_lin2, tau_cost_, linewidth=LINEWIDTH, alpha=0.8)
ax3.plot(time_lin2, translation_cost_, linewidth=LINEWIDTH, alpha=0.8)
ax4.plot(time_lin2, total_cost_, linewidth=LINEWIDTH, alpha=0.8)

ax.plot(time_lin2, total_cost_,  linewidth=LINEWIDTH, label='ddp', alpha=0.8)

fig1.legend(framealpha=0.95, fontsize=26)

fig.legend(framealpha=0.95, fontsize=26)

fig1.savefig('/home/jianghan/Devel/workspace/src/GRG/kuka_dgh/plots/plots/comparing_cost.pdf', bbox_inches="tight")

plt.show()






