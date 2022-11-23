import numpy as np
from matplotlib import pyplot as plt

class GPM_analysis:
    def __init__(self,predictions,data):
        self.predictions = predictions
        self.data = data
    
    def RMSE_func(self):
        precalc2 = (self.predictions - self.data[:,0])**2

        RMSE = np.sqrt(sum(precalc2)/self.data.shape[0])
        print('RMSE = ' + str(RMSE))
        data_range = np.max(self.data[:,0]) - np.min(self.data[:,0])
        print('Range = ' + str(data_range))
        return RMSE

    def peak_distance(self):
        ind = np.argmax(self.data[:,0])
        pd = np.abs(self.predictions[ind]-self.data[ind,0])
        print('Distance from peak value: ' + str(pd))
        return pd

    def compare_transects(self):
        unique_transects = np.unique(self.data[:,4])
        for transect in unique_transects[:1]:
            peak_dist = []
            test_transect_data = []
            predict_transect_data = []
            for i in range(self.data.shape[0]):
                if self.data[i,4] == transect:
                    peak_dist.append(self.data[i,5])
                    test_transect_data.append(self.data[i,0])
                    predict_transect_data.append(self.predictions[i])

            indices = np.argsort(peak_dist)
            peak_dist = np.array(peak_dist)[indices]
            test_transect_data = np.array(test_transect_data)[indices]
            predict_transect_data = np.array(predict_transect_data)[indices]

            plt.plot(peak_dist,test_transect_data)
            plt.plot(peak_dist,predict_transect_data)
            plt.show()
