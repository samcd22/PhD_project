import numpy as np
import pandas as pd
import scipy.stats as stats
from inference_toolbox.domain import Domain
import os
current_directory = os.getcwd()
if current_directory != '/project/':
    os.chdir('/project/')

def generate_dummy_data(sigma, log = False, I_y = 0.1, I_z = 0.1, Q = 3e13, H=10, r = 100, resolution = 20, theta = np.pi/8):

    domain = Domain('cone_from_source_z_limited', resolution = resolution)
    domain.add_domain_param('r', r)
    domain.add_domain_param('theta', theta)
    domain.add_domain_param('source', [0,0,H])

    points = domain.create_domain()

    def C_func(x, y, z, I_y, I_z, Q, H, sigma, log):
        # mu =  np.log(Q/(np.pi*I_y*I_z*x**2)) + -y**2/(2*I_y**2*x**2) + np.log(np.exp(-(z-H)**2/(2*I_z**2*x**2))+np.exp(-(z+H)**2/(2*I_z**2*x**2)))
        C = []
        C_ind = []
        if log:
            mu =  np.log10(Q/(np.pi*I_y*I_z*x**2)*np.exp(-y**2/(2*I_y**2*x**2))*(np.exp(-(z-H)**2/(2*I_z**2*x**2))+np.exp(-(z+H)**2/(2*I_z**2*x**2))))
            for i in range(len(mu)):
                C.append(mu[i] + sigma*np.random.normal())
                C_ind.append(i)
        else:
            mu =  Q/(np.pi*I_y*I_z*x**2)*np.exp(-y**2/(2*I_y**2*x**2))*(np.exp(-(z-H)**2/(2*I_z**2*x**2))+np.exp(-(z+H)**2/(2*I_z**2*x**2)))
            for i in range(len(mu)):
                beta = mu[i]/sigma**2
                a = mu[i]**2/sigma**2
                if a == 0:
                    P = stats.gamma.rvs(1e-12,scale=1/beta)
                else:
                    P = stats.gamma.rvs(a,scale=1/beta)
                if P != 0:
                    C.append(P)
                    C_ind.append(i)

        
            C = np.array(C)
            C_ind = np.array(C_ind)

        return C, C_ind


    C, C_ind = C_func(points[:,0], points[:,1], points[:,2], I_y, I_z, Q, H, sigma, log)

    X = points[C_ind,0]
    Y = points[C_ind,1]
    Z = points[C_ind,2]

    data = pd.DataFrame({'x': X, 'y': Y, 'z': Z, 'Concentration': C})

    if log:
        path = './data/log_dummy_data.csv'
    else:
        path = './data/dummy_data.csv'

    data.to_csv(path)

    return data
