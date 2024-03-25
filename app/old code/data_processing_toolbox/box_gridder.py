import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import bisect
import os

class BoxGridder:
    
    # Takes the data as an input to initialise the box gridder
    def __init__(self, data, grid_size, target = False, data_path = 'data/normalised_gridded_data'):
        self.data = data
        self.data_path = data_path
        self.grid_size = grid_size
        self.target = target

        if target:
            target_string = 'target'
        else:
            target_string = 'real'

        self.data_dir = data_path + '/gridded_' + ('_').join(str(x) for x in grid_size) + '_' + target_string
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        
    # Function which rounds down to the next base number
    def myround(self, x, base):
        return base * np.floor(x/base)

    # Function which generates the grid boundaries
    def get_grid(self):
        if self.target:
            min_x = self.myround(np.min(self.data.Distance),self.grid_size[0])
            min_y = self.myround(np.min(self.data.Peak_Dist),self.grid_size[1])
            min_z = self.myround(np.min(self.data.Height),self.grid_size[2])
            max_x = self.myround(np.max(self.data.Distance),self.grid_size[0])+ self.grid_size[0]
            max_y = self.myround(np.max(self.data.Peak_Dist),self.grid_size[1])+ self.grid_size[1]
            max_z = self.myround(np.max(self.data.Height),self.grid_size[2])+ self.grid_size[2]
        else:
            min_x = self.myround(np.min(self.data.x),self.grid_size[0])
            min_y = self.myround(np.min(self.data.y),self.grid_size[1])
            min_z = self.myround(np.min(self.data.z),self.grid_size[2])
            max_x = self.myround(np.max(self.data.x),self.grid_size[0])+self.grid_size[0]
            max_y = self.myround(np.max(self.data.y),self.grid_size[1])+self.grid_size[1]
            max_z = self.myround(np.max(self.data.z),self.grid_size[2])+self.grid_size[2]

        grid_x = np.arange(min_x,max_x+self.grid_size[0],self.grid_size[0])
        grid_y = np.arange(min_y,max_y+self.grid_size[1],self.grid_size[1])
        grid_z = np.arange(min_z,max_z+self.grid_size[2],self.grid_size[2])

        grid = (grid_x,grid_y,grid_z)

        return grid

    # Function which checks which boundary points in the grid the point of interest lies between i.e. which grid square the point is in
    def get_bounds(self, haystack, needle):
        idx = bisect.bisect(haystack, needle)
        if 0 < idx < len(haystack):
            return haystack[idx-1], haystack[idx]
        else:
            raise ValueError(f"{needle} is out of bounds of {haystack}")

    # Function which generates the centroid points of the grid, along with their associated index
    def get_indices_and_centroids(self, grid):
        indices = []
        centroids = []
        for i in range(len(grid[0])-1):
            for j in range(len(grid[1])-1):
                for k in range(len(grid[2])-1):
                    indices.append([i,j,k])
                    centroids.append([np.mean([grid[0][i],grid[0][i+1]]),np.mean([grid[1][j],grid[1][j+1]]),np.mean([grid[2][k],grid[2][k+1]])])
        return indices, np.array(centroids)      

    # Function which takes the point and the grid and attaches the 3D box bounds of that point in the grid
    def get_box_bounds_of_point(self, grid, point):
        x_lower, x_upper = self.get_bounds(grid[0], point[0])
        y_lower, y_upper = self.get_bounds(grid[1], point[1])
        z_lower, z_upper = self.get_bounds(grid[2], point[2])

        box_bounds = np.array([[x_lower,x_upper],[y_lower,y_upper],[z_lower,z_upper]])

        return box_bounds

    def get_averages(self, conc_column_name = 'Norm_Conc'):
        data_name = self.data_dir + '/results.csv'
        if os.path.exists(data_name):
            averaged_df = pd.read_csv(data_name)
        else:
            print('Generating grid averages...')
            grid = self.get_grid()

            indices, centroids = self.get_indices_and_centroids(grid)

            sums = np.zeros((len(grid[0])-1,len(grid[1])-1,len(grid[2])-1))
            counts = np.zeros((len(grid[0])-1,len(grid[1])-1,len(grid[2])-1))
            self.samples = np.empty((len(grid[0])-1,len(grid[1])-1,len(grid[2])-1), object)
            self.samples.fill(())

            for i in range(self.data.shape[0]):
                if self.target:
                    point = [self.data.loc[i,'Distance'],self.data.loc[i,'Peak_Dist'],self.data.loc[i,'Height']]
                else:
                    point = [self.data.loc[i,'x'],self.data.loc[i,'y'],self.data.loc[i,'z']]

                conc = self.data.loc[i,conc_column_name]

                box_bounds = self.get_box_bounds_of_point(grid, point)
                centroid = np.mean(box_bounds,axis=1)

                idx = np.where(np.all(centroid == centroids, axis=1))[0][0]

                a = indices[idx][0]
                b = indices[idx][1]
                c = indices[idx][2]
                
                d = self.samples[indices[idx][0],indices[idx][1],indices[idx][2]]

                sums[indices[idx][0],indices[idx][1],indices[idx][2]] += conc
                counts[indices[idx][0],indices[idx][1],indices[idx][2]] += 1
                self.samples[indices[idx][0],indices[idx][1],indices[idx][2]]+=(conc,)

            averages = sums/counts

            averages_data = []
            counts_data = []
            samples_data = []
            centroid_x = []
            centroid_y = []
            centroid_z = []

            for i in range(len(indices)):
                averages_data.append(averages[indices[i][0],indices[i][1],indices[i][2]])
                counts_data.append(counts[indices[i][0],indices[i][1],indices[i][2]])
                # print(self.samples[indices[i][0],indices[i][1],indices[i][2]])
                samples_data.append(list(self.samples[indices[i][0],indices[i][1],indices[i][2]]))
                centroid_x.append(centroids[i][0])
                centroid_y.append(centroids[i][1])
                centroid_z.append(centroids[i][2])

            averaged_df = pd.DataFrame({'x':centroid_x,'y':centroid_y,'z':centroid_z,'Concentration':averages_data, 'Counts': counts_data, 'Samples':samples_data})
            averaged_df = averaged_df.dropna()

            averaged_df.to_csv(data_name)

        return averaged_df

    def visualise_averaged_slices(self, averaged_df):
        for z in averaged_df.z.unique():
            data_slice = averaged_df[averaged_df.z == z]
            plt.scatter(data_slice.x,data_slice.y,s=data_slice.Concentration/1e8)
            plt.xlabel('Distance Downwind')
            plt.ylabel('Distance Crosswind')
            plt.title('Slice of plume data at a height of ' + str(z) + 'm')
            plt.tight_layout()
            plt.show()

    def visualise_average_data(self, averaged_df, type = 'Concentration'):
        file_name = self.data_dir + '/' + type.lower() + '_grid_plot.png'
        if not os.path.exists(file_name):
            fig = plt.figure(figsize = (10,10))

            if type == 'Concentration':
                size_output = np.log10(averaged_df.Concentration)
                colour_output = np.log10(averaged_df.Concentration)
                title = 'Average log concentration across all experiments'

            elif type == 'Counts':
                size_output = 10*np.log10(averaged_df.Counts)
                colour_output = averaged_df.Counts
                title = 'Number of data points within each grid square'

            else:
                raise Exception('Type not found')

            ax = fig.add_subplot(111, projection = '3d')
            p = ax.scatter(averaged_df.x, averaged_df.y, averaged_df.z, c=colour_output, s = 20, cmap='jet', vmin = np.percentile(colour_output,5), vmax = np.percentile(colour_output,95))

            ax.set_xlabel('Distance Downwind')
            ax.set_ylabel('Distance Crosswind')

            ax.set_title(title)
            fig.colorbar(p)
            plt.tight_layout()
            fig.savefig(file_name)
            plt.close()

    def get_sample_histograms(self, averaged_df, n_hists = 5):
        samples = averaged_df.Samples.sample(n_hists, random_state=1)
        samples_dir = self.data_dir + '/sample_grid_squares'
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)

        for i in samples.index:
            sample = samples[i]
            if type(sample) == str:
                string_elements = sample[1:-1].split(", ")

                # Convert the string elements to floats and create a new tuple
                sample = [float(element) for element in string_elements]

            plot_path = samples_dir + '/sample_grid_square_' + str(i) + '.png'

            x = averaged_df.x[i]
            y = averaged_df.y[i]
            z = averaged_df.z[i]

            title = str([x,y,z])
            plt.hist(np.log10(sample))
            plt.xlabel('Concentration')
            plt.ylabel('Count')
            plt.title(title)
            plt.savefig(plot_path)
            plt.close()