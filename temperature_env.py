import numpy as np

# define temperature environment
class TemperatureEnv():
    def __init__(self, function, x_dim = None, budget = 100, max_batch_size = 10):
        # takes a function class which takes a temperature path as first input, and possibly second argument x
        self.function = function

        # check if we are taking x-arguments
        if x_dim == None:
            self.x_dim = None
        else:
            assert isinstance(x_dim, int), 'x-dimension should be an integer'
            self.x_dim = None
        # set optim budget and batch size
        self.budget = budget
        self.max_batch_size = max_batch_size
        self.t_dim = self.function.t_dim

        # initialise other variables
        self.initialise_optim()
    
    def initialise_optim(self):
        # initialise query / observation lists
        if self.x_dim is not None:
            self.X = []
        self.Y = []

        # initialise optimisation time and batch size
        self.t = 0
        self.batch_size = 0

        # initialise eval batch
        self.temperature_list = []
        if self.x_dim is not None:
            self.batch = []
        
        # draw new function
        self.function.draw_new_function()
    
    def step(self, T_i, x = None):
        '''
        Advances the process forward. Takes as input a new temperature and possibly a new x-observation.
        '''
        # initialise new query and observation variable
        obs = None
        # add action to batch
        self.temperature_list.append(T_i)
        if self.x_dim is not None:
            self.batch.append(x)
        self.batch_size = self.batch_size + 1

        if self.batch_size == self.max_batch_size:
            # obtain temperature path
            temp_path = self.temperature_list[:-1]
            # update temperature list
            self.temperature_list = self.temperature_list[1:]
            query = (temp_path)
            # same for x-query
            if self.x_dim is not None:
                x_obs = self.batch[0]
                self.batch = self.batch[1:]
                query = (temp_path, x_obs)
            # obtain observation
            obs = self.function.query(*query)
        
        return obs

class NormalDropletFunctionEnv():
    def __init__(self, function, budget = 100, max_batch_size = 10):
        # takes a function class which takes a temperature path as first input, and possibly second argument x
        self.function = function
        # check if we are taking x-arguments
        self.x_dim = self.function.x_dim
        # set optim budget and batch size
        self.budget = budget
        self.max_batch_size = max_batch_size
        self.t_dim = function.t_dim

        # initialise other variables
        self.initialise_optim()
    
    def initialise_optim(self):
        # initialise query / observation lists
        self.X = []
        self.Y = []

        # initialise optimisation time and batch size
        self.t = 0
        self.batch_size = 0

        # initialise eval batch
        self.temperature_list = []
        if self.x_dim is not None:
            self.batch = []
        
        # draw new function
        self.function.draw_new_function()
    
    def step(self, T_i, x = None):
        '''
        Advances the process forward. Takes as input a new temperature and possibly a new x-observation.
        '''
        # initialise new query and observation variable
        obs, query_return = None, None
        # add action to batch
        self.temperature_list.append(T_i)
        if self.x_dim is not None:
            self.batch.append(x)
        self.batch_size = self.batch_size + 1

        if self.batch_size == self.max_batch_size:
            # obtain query and observation
            temp_query = self.temperature_list[0].reshape(1, -1)
            # update temperature list
            self.temperature_list = self.temperature_list[1:]
            self.batch_size = self.batch_size - 1
            query = [temp_query]
            query_return = temp_query
            # same for x-query
            if self.x_dim is not None:
                x_query = self.batch[0].reshape(1, -1)
                self.batch = self.batch[1:]
                query_return = np.concatenate((temp_query, x_query), axis = 1)
                query = [query_return]
            # obtain observation
            obs = self.function.query_function(*query)

            self.X.append(query_return)
            self.Y.append(obs)

        self.t += 1
        
        return query_return, obs
    
    def finished_with_optim(self):
        for i, t in enumerate(self.temperature_list):
            query_t = t
            query = [t]
            query_out = t
            if self.x_dim is not None:
                query_x = self.batch[i]
                query = np.concatenate((query_t.reshape(1, -1), query_x.reshape(1, -1)), axis = 1).reshape(1, -1)
                query_out = np.concatenate((query_t.reshape(1, -1), query_x.reshape(-1, 1)), axis = 1)
            obs = self.function.query_function(*query)

            self.X.append(query_out)
            self.Y.append(obs)
        
        return self.X, self.Y