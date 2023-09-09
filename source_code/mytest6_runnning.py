from itertools import count
from pickle import TRUE
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R_trans
torch.set_default_dtype(torch.float32)

def rm2eula(Rm):
    r = R_trans.from_matrix(Rm)
    eulas=r.as_euler('xyz', degrees=False)
    eulas=np.flip(eulas,axis=1)
    return eulas

def qua2rm(Rqs):
    rs = R_trans.from_quat(Rqs)
    Rms = rs.as_matrix()
    return Rms

def rm2qua(Rms):
    rs = R_trans.from_matrix(Rms)
    qua = rs.as_quat()
    return qua

def viewpoint_params_to_matrix(towards, angle):
    '''
    **Input:**

    - towards: numpy array towards vector with shape (3,).

    - angle: float of in-plane rotation.

    **Output:**

    - numpy array of the rotation matrix with shape (3, 3).
    '''
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    if np.linalg.norm(axis_y) == 0:
        axis_y = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]])
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2.dot(R1)
    return matrix.astype(np.float32)



def get_model_grasps(datapath):
    ''' 
    Load grasp labels from .npz files.
    '''
    label = np.load(datapath)
    points = label['points']
    offsets = label['offsets']
    scores = label['scores']
    collision = label['collision']
    return points, offsets, scores, collision

def generate_views(N, phi=(np.sqrt(5)-1)/2, center=np.zeros(3, dtype=np.float32), R=1):
    ''' 
    View sampling on a sphere using Febonacci lattices.

    **Input:**

    - N: int, number of viewpoints.

    - phi: float, constant angle to sample views, usually 0.618.

    - center: numpy array of (3,), sphere center.

    - R: float, sphere radius.

    **Output:**

    - numpy array of (N, 3), coordinates of viewpoints.
    '''
    idxs = np.arange(N, dtype=np.float32)
    Z = (2 * idxs + 1) / N - 1
    X = np.sqrt(1 - Z**2) * np.cos(2 * idxs * np.pi * phi)
    Y = np.sqrt(1 - Z**2) * np.sin(2 * idxs * np.pi * phi)
    views = np.stack([X,Y,Z], axis=1)
    views = R * np.array(views) + center
    return views
    

dataset_root='/home/tencent_go/Music/codes/multi_feature_get/utils/npztest'
obj_idx = 5
sampled_points, offsets, scores, collision = get_model_grasps('%s/%03d_labels.npz'%(dataset_root, obj_idx))

flag = False
th=0.4
max_width=0.08
# np.random.shuffle(point_inds)
num_views = 300
views = generate_views(num_views)
point_inds = np.arange(sampled_points.shape[0])
Rs=[]
target_points=[]
depths=[]
grasp_indice=[]
new_label={}
num=0
flag=False
# ======================= get  grasp poses =======================
for point_ind in point_inds:
    if flag: break
    target_point = sampled_points[point_ind]
    offset = offsets[point_ind]
    score = scores[point_ind]
    view_inds = np.arange(300)
    for v in view_inds:
        if flag: break
        view = views[v]
        angle_inds = np.arange(12)
        for a in angle_inds:
            if flag: break
            depth_inds = np.arange(4)
            for d in depth_inds:
                if flag: break
                angle, depth, width = offset[v, a, d]
                if score[v, a, d] > th or score[v, a, d] < 0 or width > max_width:
                    continue
                R = viewpoint_params_to_matrix(-view, angle)
                axis_y_90=np.array([[0,0,1],[0,1,0],[-1,0,0]])
                R = R.dot(axis_y_90)
                t = target_point
                Rs.append(R)
                target_points.append(t)
                depths.append(depth)
                grasp_indice.append([point_ind,v,a,d])
                num += 1
                if num == 200: 
                #sampled_points.shape[0]:选择多少个点进行测试
                    flag = True
                   
         
gym = gymapi.acquire_gym()


# Add custom arguments

args = gymutil.parse_arguments(
    description="mytest",
)


# Set controller parameters
damping = 0.05

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -10)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# create table asset
table_dims = gymapi.Vec3(4, 4, 0.01)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)


#create model
asset_root = ""
asset_options = gymapi.AssetOptions()
asset_options.use_mesh_materials = True
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

# Override the bogus inertia tensors and center-of-mass properties in the YCB assets.
# These flags will force the inertial properties to be recomputed from geometry.
asset_options.override_inertia = True
asset_options.override_com = True

# use default convex decomposition params
asset_options.vhacd_enabled = True
asset_options.fix_base_link = False
model_asset = gym.load_asset(sim, asset_root, "/home/tencent_go/Music/codes/ggcnnpybullet/pybullet_grasp/myModel/objs/meshes/2c9ec139a84e56dc4f7c5cc287db4b75_1.urdf", asset_options)


# load franka asset
franka_asset_file = "/home/tencent_go/Music/codes/multi_feature_get/urdf/franka_description/robots/franka_panda_fem_simple_v5_with_arm.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# configure franka dofs
franka_dof_props = gym.get_asset_dof_properties(franka_asset) #zi you du property
franka_lower_limits = franka_dof_props["lower"] #xia xian
franka_upper_limits = franka_dof_props["upper"] #shang xian
franka_ranges = franka_upper_limits - franka_lower_limits #yundong fanwei
franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)


franka_dof_props["driveMode"][:6].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][:6].fill(100.0)
franka_dof_props["damping"][:6].fill(100.0)
    
# grippers
franka_dof_props["driveMode"][6:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][6:].fill(100.0)
franka_dof_props["damping"][6:].fill(200.0)

# default dof states and position targets
franka_num_dofs = gym.get_asset_dof_count(franka_asset)
default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
default_dof_pos[:6] = np.array([1.0,0,1.0,0,0,0])
# grippers open
default_dof_pos[6:] = franka_upper_limits[6:]

default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
# franka_hand_index = franka_link_dict["panda_hand"]


# configure env grid
num_envs = 6*6
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)


franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0, 0, 0)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)

model_pose = gymapi.Transform()

envs = []
hand_idxs = []
model_idxs = []
init_pos_list = []
init_rot_list = []
model_handles =[]
model_states=[]
masks=[]

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 1)
    table_color = gymapi.Vec3(0.2, 0.2, 0.2)
    gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)
    
    # add franka
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 1)
    
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)
    
    # get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    # init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    # init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])
    
    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)
    
    # add model 
    model_pose.p.x = table_pose.p.x 
    model_pose.p.y = table_pose.p.y 
    model_pose.p.z = table_dims.z + 0.5
    
    model_handle = gym.create_actor(env, model_asset, model_pose, "model", i, 0)
    model_handles.append(model_handle)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, model_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    model_state = gym.get_actor_rigid_body_states(env, model_handle, gymapi.STATE_POS)
    model_states.append(model_state)
    # model_states_revised['pose']['p'].fill((0,-0.3,0.51))
    # gym.set_actor_rigid_body_states(env, model_handle, model_states_revised ,gymapi.STATE_POS)
    # get global index of box in rigid body state tensor
    model_idx = gym.get_actor_rigid_body_index(env, model_handle, 0, gymapi.DOMAIN_SIM)
    model_idxs.append(model_idx)
    

    
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

   
# ======================= prepare tensors ==========================
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)

_actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
actor_root_state_tensor = gymtorch.wrap_tensor(_actor_root_state_tensor).view(num_envs, -1, 13)

dof_pos = dof_states[:, 0].view(num_envs, 8, 1)
model_poses = torch.zeros([num_envs,3],dtype=torch.float32).to(device)
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
gripper_sep = dof_pos[:, 6] + dof_pos[:, 7]
grasped, gripped, move_up, move_down=False,False,False,False
cnt=0
initialize_cnt=0
reset=True
initialize=True
actor_root_state_tensor_fixed=torch.zeros_like(actor_root_state_tensor)
first_time= True
while not gym.query_viewer_has_closed(viewer):
    if reset:
        cnt += 1
        temp_R=Rs[(cnt-1)*num_envs:cnt*num_envs]
        temp_t=target_points[(cnt-1)*num_envs:cnt*num_envs]
        temp_depth=depths[(cnt-1)*num_envs:cnt*num_envs]
        grasp_len = len(temp_R)  
        if grasp_len==0:
            break
        elif(grasp_len<num_envs):
            temp_R+=[0.7*np.ones_like(temp_R[0])]*(num_envs-grasp_len)
            temp_t+=[0.7*np.ones_like(temp_t[0])]*(num_envs-grasp_len)
            temp_depth+=[0.7*np.ones_like(temp_depth[0])]*(num_envs-grasp_len)
        reset=False
    if(grasped and (not gripped)):
        print("enter state 2")
        state=2
    if(grasped and gripped):
        print("enter state 3")
        state=3
    if(gripped and move_up):
        print("enter state 4")
        state=4
    if(gripped and move_down):
        print("enter state 5")
        state=5
    if((not gripped) and move_down):
        print("enter state 6")
        state=6
       
    gym.simulate(sim)
    gym.fetch_results(sim, True) 
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    
    model_poses_pre = model_poses
    model_poses = rb_states[model_idxs, :3]
    model_rots = rb_states[model_idxs, 3:7]

    hand_poses = rb_states[hand_idxs, :3]
    hand_rots = rb_states[hand_idxs, 3:7]
    
    model_poses_revised=-0.15*torch.ones_like(model_poses[:,1])
    actor_root_state_tensor_revised = actor_root_state_tensor.clone()
    actor_root_state_tensor_revised[:,2,1]=model_poses_revised
    
    model_dist = torch.norm(model_poses_pre-model_poses)
              
    if initialize:
        if (model_dist>1e-4) & first_time:    
            model_poses_fixed = model_poses.clone()
            model_rots_fixed = model_rots.clone()
            actor_root_state_tensor_fixed = actor_root_state_tensor.clone()  
            first_time=False
        else:
            initial_poses=model_poses_fixed
            initial_rots=model_rots_fixed

            model_rms = qua2rm(np.asarray(initial_rots.cpu())).astype(np.float32)
            target_rots=torch.bmm(torch.from_numpy(model_rms).to(device),torch.from_numpy(np.asarray(temp_R).astype(np.float32)).to(device))
            eula_angles=rm2eula(target_rots.cpu().numpy())
            target_poses=torch.bmm(torch.from_numpy(model_rms).to(device),torch.from_numpy(np.asarray(temp_t).astype(np.float32)).unsqueeze(-1).to(device))+initial_poses.unsqueeze(-1)
            offsets=torch.Tensor([[0.0, 0.0, -0.4]] * num_envs).to(device)
            grasp_pos=torch.bmm(target_rots,offsets.unsqueeze(-1))+target_poses
            to_target = grasp_pos.view(-1,3) - hand_poses
            target_dist = torch.norm(to_target, dim=-1)
            if torch.any(target_dist<0.001):
                gym.set_actor_root_state_tensor(sim,gymtorch.unwrap_tensor(actor_root_state_tensor_revised))
                if torch.all(target_dist<0.001):
                    initialize = False
                    gym.set_actor_root_state_tensor(sim,gymtorch.unwrap_tensor(actor_root_state_tensor_fixed))
                    state=1
            pos_action[:,:3] = grasp_pos.view(-1,3)
            pos_action[:,3:6] = torch.from_numpy(eula_angles.astype(np.float32)).to(device)
            pos_action[:,6:] = (torch.Tensor([[0.04, 0.04]] * num_envs).to(device))
    else:
        if(state==1):
            print("state==1")
            offsets=torch.Tensor([[0.0, 0.0, 0.0]] * num_envs).to(device)
            offsets[:,2]=torch.from_numpy((0.108-np.asarray(temp_depth).astype(np.float32))*-1.0).to(device)
            grasp_pos=torch.bmm(target_rots,offsets.unsqueeze(-1))+target_poses
            to_target = grasp_pos.view(-1,3) - hand_poses
            target_dist = torch.norm(to_target, dim=-1)
            print(target_dist)
            if torch.any(target_dist<0.001):
                grasped=True
            pos_action[:,:3] = grasp_pos.view(-1,3)
            pos_action[:,3:6] = torch.from_numpy(eula_angles.astype(np.float32)).to(device)
            pos_action[:,6:] = (torch.Tensor([[0.04, 0.04]] * num_envs).to(device)) 
        if(state==2):
            print("state==2")
            pos_action[:,6:] = (torch.Tensor([[0.00, 0.00]] * num_envs).to(device))
            pre_gripper_seq=gripper_sep
            gripper_sep = dof_pos[:, 6] + dof_pos[:, 7]
            print(abs(pre_gripper_seq-gripper_sep))
            if torch.all(abs(pre_gripper_seq-gripper_sep)<1e-6):
                gripped=True 
        if(state==3):
            print("state==3")  
            grasp_pos=torch.Tensor([[0.0,0.0,1.0,0.0,0.0,3.14]] * num_envs).to(device)
            pos_action[:,:6]=grasp_pos
            to_target = grasp_pos[:,:3] - hand_poses
            target_dist = torch.norm(to_target, dim=-1)
            print(target_dist)
            if torch.any(target_dist<1e-4):
                mask=(model_poses[:grasp_len,2]<0.7).tolist()
                masks.extend(mask)
                grasped=False
                move_up=True                      
            
        if(state==4): 
            print("state==4")
            grasp_pos=torch.Tensor([[0.0,0.0,0.75,0.0,0.0,3.14]] * num_envs).to(device)
            pos_action[:,:6]=grasp_pos
            to_target = grasp_pos[:,:3] - hand_poses
            target_dist = torch.norm(to_target, dim=-1)
            if torch.any(target_dist<0.0001):    
                move_down=True
                move_up=False    
            
        if(state==5):
            print("state==5")
            pos_action[:,6:] = (torch.Tensor([[0.04, 0.04]] * num_envs).to(device))
            pre_gripper_seq=gripper_sep
            gripper_sep = dof_pos[:, 6] + dof_pos[:, 7]
            if torch.any(abs(pre_gripper_seq-gripper_sep)<1e-5):
                gripped=False
        
        if(state==6):
             print("state==6")
             gym.set_actor_root_state_tensor(sim,gymtorch.unwrap_tensor(actor_root_state_tensor_fixed))
             reset=True
             grasped,gripped,move_up,move_down=False,False,False,False
             initialize=True
             first_time=True
                             
    
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)   



print(masks)

scores[np.asarray(grasp_indice)[masks]]=-1.0
new_label['scores']= scores
new_label['points']=sampled_points
new_label['offsets']=offsets
new_label['collision']=collision





        