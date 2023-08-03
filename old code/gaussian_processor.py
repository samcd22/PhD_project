import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import f1_score, mean_squared_error
import os
import imageio


class GaussianProcessor:
    def __init__(self, data, data_norm = 'none', kernel = 'metern_white'):
        self.data_normaliser = data_norm
        self.data = data
        self.kernel_dict = {'matern_white': 'Matern and White Noise', 'rbf': 'Radial Basis Function', 'matern':'Matern'}
        self.normaliser_dict = {'none':'', 'sqrt': 'square route of the'}
        self.kernel = kernel

        path = 'results/gaussian_process/'
        folder_name = self.kernel + '_' + self.data_normaliser
                        
        if not os.path.exists(path + folder_name):
            os.makedirs(path + folder_name)
        else:
            for f in os.listdir(path + folder_name):
                os.remove(os.path.join(path + folder_name, f))

        self.params = None
        self.model = False
        
    def train_gp(self, training_data, max_z = 125, num_epochs = 20, conc_col_name = 'Concentration'):
        data_for_GP = training_data[training_data['z']<=max_z]

        # x, y, and z are the independent variables (scalars)
        x = data_for_GP.x.values
        y = data_for_GP.y.values
        z = data_for_GP.z.values

        # "Concentration" is the dependent variable
        concentration = data_for_GP[conc_col_name].values

        # Stack the independent variables into a 2D array
        X = np.column_stack((x, y, z))

        # Create the kernel
        if self.kernel == 'matern_white':
            kernel = Matern(length_scale=1, nu=0.5) + WhiteKernel(noise_level=1.0)
        elif self.kernel == 'rbf':
            kernel = RBF(length_scale=1)
        elif self.kernel == 'matern':
            kernel = Matern(length_scale=1, nu=0.5)        

        # Create the Gaussian Process Regression model
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=num_epochs, normalize_y=True)

        if self.data_normaliser == 'none':
            norm_conc = concentration
        elif self.data_normaliser == 'sqrt':
            norm_conc = np.sqrt(concentration)

        # Fit the model with the data
        self.model.fit(X, norm_conc)
        
        self.params = self.model.kernel_.get_params()
        print(self.params)
        
        return self.model

    def predict_from_gp(self, grid, threeD = False, save = False, log_results = False):
        # Define a 3D grid of points
        if self.model:
            x_grid = grid[0]
            y_grid = grid[1]
            z_grid = grid[2]
            X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)
            grid_points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

            # Predict using the trained Gaussian Process Regression model
            concentration_pred, sigma_pred = self.model.predict(grid_points, return_std=True)
            
            if self.data_normaliser == 'none':
                norm_conc = concentration_pred
                norm_sigma = sigma_pred
            elif self.data_normaliser == 'sqrt':
                norm_conc = concentration_pred**2
                norm_sigma = sigma_pred**2

            if threeD:
                self.threeD_plots(X, Y, Z, norm_conc, norm_sigma, q=10, save = save, log_results=log_results)
            else:
                self.slice_plots(x_grid, y_grid, z_grid, norm_conc, norm_sigma)
        else:
            print("No trained model available!")

    def threeD_plots(self, X, Y, Z, concentration_pred, sigma, q=10, save = False, log_results = False):
        #  labs = pd.qcut(np.log10(concentration_pred,q))

        if log_results:
            lower_conc = np.log10(concentration_pred - 2*sigma)
            lower_conc = np.nan_to_num(lower_conc)

            mean_conc = np.log10(concentration_pred)
            mean_conc = np.nan_to_num(mean_conc)

            upper_conc = np.log10(concentration_pred + 2*sigma)
            upper_conc = np.nan_to_num(upper_conc)

        else:
            lower_conc = concentration_pred - 2*sigma
            lower_conc[lower_conc<0] = 0

            mean_conc = concentration_pred
            mean_conc[mean_conc<0] = 0

            upper_conc = concentration_pred + 2*sigma
            upper_conc[upper_conc<0] = 0


        mean_bin_nums = pd.qcut(mean_conc,q, labels = False, duplicates='drop')
        
        lower_bin_nums = pd.qcut(lower_conc,q, labels = False, duplicates='drop')

        upper_bin_nums = pd.qcut(upper_conc,q, labels = False)

        min_val = np.min(lower_conc)
        max_val = np.max(upper_conc)

        mean_conc_and_bins = pd.DataFrame([X.reshape(mean_bin_nums.shape), Y.reshape(mean_bin_nums.shape), Z.reshape(mean_bin_nums.shape), mean_conc, mean_bin_nums]).T
        mean_conc_and_bins.columns=['x', 'y', 'z', 'conc', 'bin']

        lower_conc_and_bins = pd.DataFrame([X.reshape(lower_bin_nums.shape), Y.reshape(lower_bin_nums.shape), Z.reshape(lower_bin_nums.shape), lower_conc, lower_bin_nums]).T
        lower_conc_and_bins.columns=['x', 'y', 'z', 'conc', 'bin']

        upper_conc_and_bins = pd.DataFrame([X.reshape(upper_bin_nums.shape), Y.reshape(upper_bin_nums.shape), Z.reshape(upper_bin_nums.shape), upper_conc, upper_bin_nums]).T
        upper_conc_and_bins.columns=['x', 'y', 'z', 'conc', 'bin']

        min_x = np.min(X)
        min_y = np.min(Y)
        min_z = np.min(Z)
        max_x = np.max(X)
        max_y = np.max(Y)
        max_z = np.max(Z)

        print(np.unique(mean_bin_nums))

        prefix_dict = {True: 'Log m', False: 'M'}

        for bin_num in np.unique(mean_bin_nums):

            mean_bin_data = mean_conc_and_bins[mean_conc_and_bins['bin'] == bin_num]
            lower_bin_data = lower_conc_and_bins[lower_conc_and_bins['bin'] == bin_num - (np.unique(mean_bin_nums).size - np.unique(lower_bin_nums).size)]
            upper_bin_data = upper_conc_and_bins[upper_conc_and_bins['bin'] == bin_num]


            fig = plt.figure(figsize = (20,6))
            fig.subplots_adjust(hspace=0.4, wspace=0.4)

            lower_output = lower_bin_data.conc
            mean_output = mean_bin_data.conc
            upper_output = upper_bin_data.conc

            ax1 = fig.add_subplot(131, projection = '3d')
            plot_1 = ax1.scatter(lower_bin_data.x, lower_bin_data.y, lower_bin_data.z, c = lower_output, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
            ax1.set_title(prefix_dict[log_results] + 'ean - 2σ concentration')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_xlim(min_x, max_x)
            ax1.set_ylim(min_y, max_y)
            ax1.set_zlim(min_z, max_z)
            plt.tight_layout()
            plt.colorbar(plot_1)

            ax2 = fig.add_subplot(132, projection = '3d')
            plot_2 = ax2.scatter(mean_bin_data.x, mean_bin_data.y, mean_bin_data.z, c = mean_output, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
            ax2.set_title(prefix_dict[log_results] + 'ean concentration')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('z')
            ax2.set_xlim(min_x, max_x)
            ax2.set_ylim(min_y, max_y)
            ax2.set_zlim(min_z, max_z)
            plt.tight_layout()
            plt.colorbar(plot_2)

            ax3 = fig.add_subplot(133, projection = '3d')
            plot_3 = ax3.scatter(upper_bin_data.x, upper_bin_data.y, upper_bin_data.z, c = upper_output, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
            ax3.set_title(prefix_dict[log_results] + 'ean + 2σ concentration')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_zlabel('z')
            ax3.set_xlim(min_x, max_x)
            ax3.set_ylim(min_y, max_y)
            ax3.set_zlim(min_z, max_z)
            plt.tight_layout()
            plt.colorbar(plot_3)

            plt.suptitle('Gaussian process using the ' + self.normaliser_dict[self.data_normaliser] + ' concentration data and the ' + self.kernel_dict[self.kernel] + ' kernel, plot ' + str(int(bin_num+1)) + ' of ' + str(q))
            if save:
                path = 'results/gaussian_process/'
                folder_name = self.kernel + '_' + self.data_normaliser
                file_name =  'plot_' + str(int(bin_num+1)) + '_of_' + str(q) + '.png'
                fig.savefig(path + folder_name + '/' + file_name)
            else:
                plt.show()

        # p = ax.scatter(conc_and_bins.x, conc_and_bins.y, conc_and_bins.z, c = conc_and_bins.conc, alpha = 0.4)
        # fig.colorbar(p)
    
    def train_test_split(self, training_ratio = 0.8):
        msk = np.random.rand(np.shape(self.data)[0]) < training_ratio

        train = self.data[msk]
        test = self.data[~msk]

        return train, test

    def test(self, test_set, conc_col_name = 'Concentration'):
        grid_points = np.column_stack((test_set.x, test_set.y, test_set.z))
        concentration_pred = self.model.predict(grid_points)

        if self.data_normaliser == 'sqrt':
            norm_conc = concentration_pred**2
        elif self.data_normaliser == 'none':
            norm_conc = concentration_pred
        # f_score = f1_score(test_set[conc_col_name], concentration_pred) 
        MSE = np.sqrt(mean_squared_error(test_set[conc_col_name], norm_conc))
        # print('F score = ' + str(f_score))
        print('MSE = ' + str(MSE))

    def animate(self, frame_dur = 0.5):
        path = 'results/gaussian_process/'
        folder_name = self.kernel + '_' + self.data_normaliser

        files = os.listdir('results/gaussian_process/' + folder_name)

        images = []
        for i in range(len(files)):
            images.append(imageio.imread(path + folder_name + '/plot_' + str(i+1) + '_of_' + str(len(files)) + '.png'))
        gif_name = folder_name + '.gif'
        gif_path = 'results/gaussian_process/gifs/' + gif_name
        if os.path.exists(gif_path):
            os.remove(gif_path)

        imageio.mimsave(gif_path, images, duration = frame_dur)

    def slice_plots(self, x_grid, y_grid, z_grid, concentration_pred, sigma):
        
        X, _, _ = np.meshgrid(x_grid, y_grid, z_grid)
        # Reshape the predicted concentration values and standard deviation to match the shape of the grid
        concentration_pred = concentration_pred.reshape(X.shape)
        sigma = sigma.reshape(X.shape)

        # Plot the predicted concentration values using pcolormesh
        for i, z_val in enumerate(z_grid):
            fig = plt.figure(figsize = (10,10))
            fig.subplots_adjust(hspace=0.4, wspace=0.4)

            min_val = np.min(np.log10(concentration_pred-2*sigma))
            if np.isnan(min_val):
                min_val = 8 

            max_val = np.max(np.log10(concentration_pred+2*sigma))

            ax1 = fig.add_subplot(2,2,1)
            plot_1 = ax1.pcolormesh(x_grid, y_grid, np.log10(concentration_pred[:, :, i] - 2*sigma[:, :, i]), cmap='jet', shading='auto', vmin=min_val, vmax=max_val)
            ax1.set_title('Log mean - 2σ concentration')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            plt.tight_layout()
            plt.colorbar(plot_1)

            ax2 = fig.add_subplot(2,2,2)
            plot_2 = ax2.pcolormesh(x_grid, y_grid, np.log10(concentration_pred[:, :, i]), cmap='jet', shading='auto', vmin=min_val, vmax=max_val)
            ax2.set_title('Log mean concentration')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            plt.tight_layout()
            plt.colorbar(plot_2)

            ax3 = fig.add_subplot(2,2,3)
            plot_3 = ax3.pcolormesh(x_grid, y_grid, np.log10(concentration_pred[:, :, i] + 2*sigma[:, :, i]), cmap='jet', shading='auto', vmin=min_val, vmax=max_val)
            ax3.set_title('Log mean + 2σ concentration')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            plt.tight_layout()
            plt.colorbar(plot_3)

            ax4 = fig.add_subplot(2,2,4)
            plot_4 = ax4.pcolormesh(x_grid, y_grid, np.log10(sigma[:, :, i]), cmap='jet', shading='auto')
            ax4.set_title('Log σ concentration')
            ax4.set_xlabel('x')
            ax4.set_ylabel('y')
            plt.tight_layout()
            plt.colorbar(plot_4)
            
            plt.suptitle('z = {:.2f}m'.format(z_val))

            plt.show()

