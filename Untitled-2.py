# %%
import numpy as np
import pandas as pd
import scipy.stats as stats
from inference_toolbox.domain import Domain

I_y = 0.1
I_z = 0.1
Q = 3e13

H=10

sigma = 1e6

domain = Domain('cone_from_source_z_limited')
domain.add_domain_param('r', 1000)
domain.add_domain_param('theta', np.pi/8)
domain.add_domain_param('source', [0,0,10])

points = domain.create_domain(resolution = 10)

def C_func(x, y, z, I_y, I_z, Q, H, sigma):
    # mu =  np.log(Q/(np.pi*I_y*I_z*x**2)) + -y**2/(2*I_y**2*x**2) + np.log(np.exp(-(z-H)**2/(2*I_z**2*x**2))+np.exp(-(z+H)**2/(2*I_z**2*x**2)))

    mu =  Q/(np.pi*I_y*I_z*x**2)*np.exp(-y**2/(2*I_y**2*x**2))*(np.exp(-(z-H)**2/(2*I_z**2*x**2))+np.exp(-(z+H)**2/(2*I_z**2*x**2)))

    C = []
    C_ind = []
    for i in range(len(mu)):
        beta = mu[i]/sigma**2
        a = mu[i]**2/sigma**2
        if a == 0:
            P = stats.gamma.rvs(1e-12,scale=1/beta)
        else:
            P = stats.gamma.rvs(a,scale=1/beta)

        if P==0:
            o=1
        if P != 0:
            C.append(P)
            C_ind.append(i)

        
    C = np.array(C)
    C = np.array(C_ind)
    return C, C_ind

C, C_ind = C_func(points[:,0], points[:,1], points[:,2], I_y, I_z, Q, H, sigma)

X = points[C_ind,0]
Y = points[C_ind,1]
Z = points[C_ind,2]

data = pd.DataFrame(data = [X,Y,Z,C]).T
data.columns = ['x', 'y', 'z', 'Concentration']

print(data[data['Concentration']<0])

data.to_csv('./data/dummy_data.csv')



