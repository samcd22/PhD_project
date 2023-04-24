import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import bisect

class BoxGridder:
    
    def __init__(self, data):
        self.data = data
        
    def myround(self, x, base):
        return base * np.floor(x/base)

    def get_grid(self, grid_size, target = False):
        if target:
            min_x = self.myround(np.min(self.data.Distance),grid_size[0])
            min_y = self.myround(np.min(self.data.Peak_Dist),grid_size[1])
            min_z = self.myround(np.min(self.data.Height),grid_size[2])
            max_x = self.myround(np.max(self.data.Distance),grid_size[0])+ grid_size[0]
            max_y = self.myround(np.max(self.data.Peak_Dist),grid_size[1])+ grid_size[1]
            max_z = self.myround(np.max(self.data.Height),grid_size[2])+ grid_size[2]
        else:
            min_x = self.myround(np.min(self.data.x),grid_size[0])
            min_y = self.myround(np.min(self.data.y),grid_size[1])
            min_z = self.myround(np.min(self.data.z),grid_size[2])
            max_x = self.myround(np.max(self.data.x),grid_size[0])+grid_size[0]
            max_y = self.myround(np.max(self.data.y),grid_size[1])+grid_size[1]
            max_z = self.myround(np.max(self.data.z),grid_size[2])+grid_size[2]

        grid_x = np.arange(min_x,max_x+grid_size[0],grid_size[0])
        grid_y = np.arange(min_y,max_y+grid_size[1],grid_size[1])
        grid_z = np.arange(min_z,max_z+grid_size[2],grid_size[2])

        grid = (grid_x,grid_y,grid_z)

        return grid


    def get_bounds(self, haystack, needle):
        idx = bisect.bisect(haystack, needle)
        if 0 < idx < len(haystack):
            return haystack[idx-1], haystack[idx]
        else:
            raise ValueError(f"{needle} is out of bounds of {haystack}")

    def get_indices_and_centroids(self, grid):
        indices = []
        centroids = []
        for i in range(len(grid[0])-1):
            for j in range(len(grid[1])-1):
                for k in range(len(grid[2])-1):
                    indices.append([i,j,k])
                    centroids.append([np.mean([grid[0][i],grid[0][i+1]]),np.mean([grid[1][j],grid[1][j+1]]),np.mean([grid[2][k],grid[2][k+1]])])
        return indices, np.array(centroids)      

    def get_grid_index(self, grid, point):
        x_lower, x_upper = self.get_bounds(grid[0], point[0])
        y_lower, y_upper = self.get_bounds(grid[1], point[1])
        z_lower, z_upper = self.get_bounds(grid[2], point[2])

        box_bounds = np.array([[x_lower,x_upper],[y_lower,y_upper],[z_lower,z_upper]])

        return box_bounds

    def get_averages(self, grid_size, target = False, conc_column_name = 'Norm_Conc'):
        grid = self.get_grid(grid_size, target)

        indices, centroids = self.get_indices_and_centroids(grid)

        sums = np.zeros((len(grid[0])-1,len(grid[1])-1,len(grid[2])-1))
        counts = np.zeros((len(grid[0])-1,len(grid[1])-1,len(grid[2])-1))

        for i in range(self.data.shape[0]):
            if target:
                point = [self.data.loc[i,'Distance'],self.data.loc[i,'Peak_Dist'],self.data.loc[i,'Height']]
            else:
                point = [self.data.loc[i,'x'],self.data.loc[i,'y'],self.data.loc[i,'z']]

            conc = self.data.loc[i,conc_column_name]

            box_bounds = self.get_grid_index(grid, point)
            centroid = np.mean(box_bounds,axis=1)

            idx = np.where(np.all(centroid == centroids, axis=1))[0][0]

            sums[indices[idx][0],indices[idx][1],indices[idx][2]] += conc
            counts[indices[idx][0],indices[idx][1],indices[idx][2]] += 1

        averages = sums/counts

        averages_data = []
        centroid_x = []
        centroid_y = []
        centroid_z = []

        for i in range(len(indices)):
            averages_data.append(averages[indices[i][0],indices[i][1],indices[i][2]])
            centroid_x.append(centroids[i][0])
            centroid_y.append(centroids[i][1])
            centroid_z.append(centroids[i][2])

        averaged_df = pd.DataFrame({'x':centroid_x,'y':centroid_y,'z':centroid_z,'Concentration':averages_data})
        averaged_df = averaged_df.dropna()
        return averaged_df

    def visualise_averaged_slices(self, averaged_df):
        for z in averaged_df.z.unique():
            data_slice = averaged_df[averaged_df.z == z]
            plt.scatter(data_slice.x,data_slice.y,s=data_slice.Concentration/1e8)
            plt.xlabel('Distance Downwind')
            plt.ylabel('Distance Crosswind')
            plt.title('Slice of plume data at a height of ' + str(z) + 'm')
            plt.show()

    def visualise_average_data(self, averaged_df):
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111, projection = '3d')
        p = ax.scatter(averaged_df.x, averaged_df.y, averaged_df.z, c=np.log10(averaged_df.Concentration), s = np.log10(averaged_df.Concentration), cmap='jet')
        ax.set_xlabel('Distance Downwind')
        ax.set_ylabel('Distance Crosswind')
        ax.set_title('Average log concentration across all experiments')
        fig.colorbar(p)
        plt.show()