import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from inference_toolbox.parameter import Parameter
from inference_toolbox.model import Model
from inference_toolbox.likelihood import Likelihood
from inference_toolbox.sampler import Sampler
from inference_toolbox.visualiser import Visualiser

from data_processing.get_data import get_data

class Optimiser:
    def __init__(self, hyperparameter_space, data_params, default_params = {}):
        self.default_params = default_params
        self.hyperparameter_space = hyperparameter_space
        self.data_params = data_params


    # Define the RMSE cost function
    def calculate_rmse(self, hyperparams):
        data = get_data(self.data_params['data_type'], self.data_params)
        training_data, testing_data = train_test_split(data, test_size=0.2)
        params = pd.Series({},dtype='float64')

        # Parameter Assignment
        I_y = Parameter(name = 'I_y', prior_select="gamma")
        I_y.add_prior_param("mu",hyperparams['I_y_mu'])
        I_y.add_prior_param("sigma",hyperparams['I_y_sigma'])
        params['I_y'] = I_y

        I_z = Parameter(name = 'I_z', prior_select="gamma")
        I_z.add_prior_param("mu",hyperparams['I_z_mu'])
        I_z.add_prior_param("sigma",hyperparams['I_z_sigma'])
        params['I_z'] = I_z

        Q = Parameter(name = 'Q', prior_select="gamma")
        Q.add_prior_param("mu",hyperparams['Q_mu'])
        Q.add_prior_param("sigma",hyperparams['Q_sigma'])
        params['Q'] = Q

        sigma = Parameter(name = 'sigma', prior_select="gamma")
        sigma.add_prior_param("mu",hyperparams['sigma_mu'])
        sigma.add_prior_param("sigma",hyperparams['sigma_sigma'])
        params['sigma'] = sigma

        model = Model('log_gpm_alt_norm')
        model.add_model_param("H",10)

        # Likelihood function assigmnent
        likelihood = Likelihood("gaussian")

        num_samples = 10000

        data_path = 'results/test'

        # Initialize the sampler
        sampler = Sampler(params, 
                        model, 
                        likelihood, 
                        training_data, 
                        num_samples, 
                        show_sample_info = True, 
                        n_chains=1, 
                        thinning_rate=1,
                        data_path = data_path)

        # Sample the parameters
        params_samples, chain_samples, fields = sampler.sample_all()
        visualiser = Visualiser(testing_data, 
                        params_samples, 
                        model, 
                        sampler.hyperparams, 
                        chain_samples=chain_samples,
                        fields = fields, 
                        previous_instance = sampler.instance, 
                        data_path = data_path,
                        suppress_prints=True)
        
        return visualiser.RMSE

    def run(self, num_episodes, batch_size, max_timesteps):
        # Initialize the DQN agent
        state_size = len(self.hyperparameter_space)
        action_size = 1  # Number of hyperparameters to select in each action
        agent = DQNAgent(state_size, action_size)

        # Hyperparameter optimization using DQN

        for episode in range(num_episodes):
            state = np.random.rand(1, state_size)
            total_reward = 0
            for t in range(max_timesteps):
                action = agent.act(state)
                hyperparameters = [hyperparameter_space[k][action] for k in hyperparameter_space]
                cost = self.calculate_rmse(dict(zip(hyperparameter_space.keys(), hyperparameters)))
                reward = -cost  # Negative RMSE as the reward

                next_state = np.random.rand(1, state_size)  # Update the next state with the new values

                agent.remember(state, action, reward, next_state, False)  # Store the experience in agent's memory
                state = next_state  # Update the state with the next_state

                total_reward += reward

                # Perform the replay step to train the agent
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            # Reduce exploration rate over time
            agent.epsilon *= agent.epsilon_decay

            print("Episode: {}/{}, Total Reward: {:.2f}".format(episode+1, num_episodes, total_reward))
            print(hyperparameters)

        
# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995  # Epsilon decay rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.learning_rate = 0.001  # Learning rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


hyperparameter_space = {
    'I_y_mu': np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]),
    'I_y_sigma': np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]),
    'I_z_mu': np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]),
    'I_z_sigma': np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]),
    'Q_mu': np.array([3e9, 3e10, 3e11, 3e12, 3e13, 3e14, 3e15, 3e16, 3e17, 3e18]),
    'Q_sigma': np.array([1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18]),
    'sigma_mu': np.array([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]),
    'sigma_sigma': np.array([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
}

data_params = {
    'data_type': 'gridded',
    'output_header': 'Concentration',
    'log':True,
    'grid_size': [200,200,50],
    'target': False,
    'data_path':'data'
}              

optimiser = Optimiser(hyperparameter_space, data_params)

optimiser.run(10, 1, 1)