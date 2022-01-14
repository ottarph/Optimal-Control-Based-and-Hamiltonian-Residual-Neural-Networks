import numpy as np
import torch
from torch.utils.data import Dataset
    

class MassSpringVectorFieldDataset(Dataset):
    
    def __init__(self, qq, pp, dqq, dpp):

        self.x = torch.tensor(np.column_stack((qq, pp))).float()
        self.y = torch.tensor(np.column_stack((dqq, dpp))).float()
        
        return
    
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        xx = self.x[idx,:]
        yy = self.y[idx,:]

        
        return xx, yy
        


def mass_spring_vector_field_data_gen(N_points, H_min, H_max, sigma, m=1, k=1, seed=None):

    if seed is not None:
        np.random.seed(seed=seed)

    r = np.sqrt(np.random.uniform(H_min, H_max, N_points))
    theta = np.random.uniform(0.0, 2*np.pi, N_points)

    q = np.sqrt(2/k) * r * np.cos(theta)
    p = np.sqrt(2*m) * r * np.sin(theta)

    dq = p / m
    dp = -k * q

    dq += np.random.normal(loc=0.0, scale=sigma, size=dq.shape)
    dp += np.random.normal(loc=0.0, scale=sigma, size=dp.shape)

    return q, p, dq, dp


def getMassSpringVectorFieldDataset(N_points, H_min, H_max, sigma, m=1, k=1, seed=None):

    qq, pp, dqq, dpp = mass_spring_vector_field_data_gen(N_points, H_min, H_max,
                                                                sigma, m=m, k=k, seed=seed)

    dataset = MassSpringVectorFieldDataset(qq, pp, dqq, dpp)

    return dataset
    

class MassSpringTrajectoryDataset(Dataset):
    
    def __init__(self, q_trajectories, p_trajectories, times):
        """
            First axis is trajectory, second axis is time, third axis is coordinate.
        """
        
        z = torch.stack((torch.tensor(q_trajectories).float(),
                         torch.tensor(p_trajectories).float()), axis=2)
        
        self.x = z[:,0,:]
        self.y = z[:,1:,:]

        self.pred_times = times[1:] - times[0]
        self.trajectory_length = times.shape[0]
        
        return
    
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        xx = self.x[idx,:]
        yy = self.y[idx,:,:]

        
        return xx, yy
    


def mass_spring_trajectory_data_gen(N_traj, traj_length, delta_t, H_min, H_max, 
                                                sigma, m=1, k=1, seed=None):
    """
        First axis is trajectory, second axis is time.
    """
    
    q0, p0, _, _ = mass_spring_vector_field_data_gen(N_traj, H_min, H_max, 
                                                    sigma, m=m, k=k, seed=seed)


    t = np.linspace(0.0, (traj_length-1)*delta_t, traj_length)

    w = np.sqrt(k/m)
    q_ex = lambda t, q0, p0: q0 * np.cos(w*t) + p0/np.sqrt(k*m) * np.sin(w*t)
    p_ex = lambda t, q0, p0: -q0*np.sqrt(m*k) * np.sin(w*t) + p0 * np.cos(w*t)

    q_trajectories = q_ex(t[None,:], q0[:,None], p0[:,None])
    p_trajectories = p_ex(t[None,:], q0[:,None], p0[:,None])

    q_trajectories += np.random.normal(scale=sigma, size=q_trajectories.shape)
    p_trajectories += np.random.normal(scale=sigma, size=p_trajectories.shape)

    return q_trajectories, p_trajectories, t


def getMassSpringTrajectoryDataset(N_traj, traj_length, delta_t, H_min, H_max, 
                                                sigma, m=1, k=1, seed=None):

    q_traj, p_traj, t_out = mass_spring_trajectory_data_gen(N_traj, traj_length, delta_t, 
                                                    H_min, H_max, sigma, m=m, k=k, seed=seed)

    dataSet = MassSpringTrajectoryDataset(q_traj, p_traj, t_out)

    return dataSet

