import numpy as np

class NormalDropletFunctionEnv():
    def __init__(self, function, budget = 100, max_batch_size = 10):
        '''
        Environment inspired by micro-reactors. A single query is submitted at every time-step, and the environment evaluates up to
        max_batch_size queries at any time. This means there is a max_batch_size iteration delay between asking for a query, and obtaining
        an evaluation. To define the environment we require an objective function.

        Input:
        function - Function to optimise (see function class on functions.py)
        budget - integer, budget of optimization
        max_batch_size - integer, value of t_delay (called batch size because it is the 'batch size' of the micro-reactor)
        '''
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
        Advances the optimization process forward. Takes as input a new temperature and possibly a new x-values
        Recall: temperature are all variables that incur input cost, x-values are variables we can change freely
        '''
        # initialise new query and observation variable
        obs, query_return = None, None
        # add action / query to batch of evaluations
        self.temperature_list.append(T_i)
        if self.x_dim is not None:
            self.batch.append(x)
        # add one to how many queries are being evaluated
        self.batch_size = self.batch_size + 1

        # once reactor is full, we can begin to output queries, this is equivalent to t >= t_delay, before this we would not have
        # any observations
        if self.batch_size == self.max_batch_size:
            # obtain query and observation
            temp_query = self.temperature_list[0].reshape(1, -1)
            # update temperature list by removing queries that are finished evaluating
            self.temperature_list = self.temperature_list[1:]
            # reduce batch size
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
            # keep track of data
            self.X.append(query_return)
            self.Y.append(obs)
        # increase time-step
        self.t += 1
        
        return query_return, obs
    
    def finished_with_optim(self):
        '''
        This function returns all evaluations once the optimization procedure is finished
        '''
        # add all queries not finished being evaluated to X and Y
        for i, t in enumerate(self.temperature_list):
            # different format of queries required due to inefficient coding
            query_t = t
            query = [t]
            query_out = t
            # add x-variables if requiried
            if self.x_dim is not None:
                query_x = self.batch[i]
                query = np.concatenate((query_t.reshape(1, -1), query_x.reshape(1, -1)), axis = 1).reshape(1, -1)
                query_out = np.concatenate((query_t.reshape(1, -1), query_x.reshape(1, -1)), axis = 1)
            # get observations
            obs = self.function.query_function(*query)
            # append to X and Y list
            self.X.append(query_out)
            self.Y.append(obs)
        
        return self.X, self.Y


class MultiObjectiveNormalDropletFunctionEnv():
    def __init__(self, function, budget = 100, max_batch_size = 10):
        '''
        Environment inspired by micro-reactors. A single query is submitted at every time-step, and the environment evaluates up to
        max_batch_size queries at any time. This means there is a max_batch_size iteration delay between asking for a query, and obtaining
        an evaluation. To define the environment we require an objective function.

        Input:
        function - Function to optimise (see function class on functions.py)
        budget - integer, budget of optimization
        max_batch_size - integer, value of t_delay (called batch size because it is the 'batch size' of the micro-reactor)
        '''
        # takes a function class which takes a temperature path as first input, and possibly second argument x
        self.function = function
        self.num_of_objectives = self.function.num_of_objectives
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
        self.Y = [[] for _ in range(self.num_of_objectives)]

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
        Advances the optimization process forward. Takes as input a new temperature and possibly a new x-values
        Recall: temperature are all variables that incur input cost, x-values are variables we can change freely
        '''
        # initialise new query and observation variable
        obs, query_return = None, None
        # add action / query to batch of evaluations
        self.temperature_list.append(T_i)
        if self.x_dim is not None:
            self.batch.append(x)
        # add one to how many queries are being evaluated
        self.batch_size = self.batch_size + 1

        # once reactor is full, we can begin to output queries, this is equivalent to t >= t_delay, before this we would not have
        # any observations
        if self.batch_size == self.max_batch_size:
            # obtain query and observation
            temp_query = self.temperature_list[0].reshape(1, -1)
            # update temperature list by removing queries that are finished evaluating
            self.temperature_list = self.temperature_list[1:]
            # reduce batch size
            self.batch_size = self.batch_size - 1
            query = [temp_query]
            query_return = temp_query
            # same for x-query
            if self.x_dim is not None:
                x_query = self.batch[0].reshape(1, -1)
                self.batch = self.batch[1:]
                query_return = np.concatenate((temp_query, x_query), axis = 1)
                query = [query_return]
            # obtain observations
            obs = self.function.query_function(*query)
            # keep track of data
            self.X.append(query_return)
            for obj in range(self.num_of_objectives):
                self.Y[obj].append(obs[obj])
        # increase time-step
        self.t += 1
        return query_return, obs
    
    def finished_with_optim(self):
        '''
        This function returns all evaluations once the optimization procedure is finished
        '''
        # add all queries not finished being evaluated to X and Y
        for i, t in enumerate(self.temperature_list):
            # different format of queries required due to inefficient coding
            query_t = t
            query = [t]
            query_out = t
            # add x-variables if requiried
            if self.x_dim is not None:
                query_x = self.batch[i]
                query = np.concatenate((query_t.reshape(1, -1), query_x.reshape(1, -1)), axis = 1).reshape(1, -1)
                query_out = np.concatenate((query_t.reshape(1, -1), query_x.reshape(1, -1)), axis = 1)
            # get observations
            obs = self.function.query_function(*query)
            # append to X and Y list
            self.X.append(query_out)
            for obj in range(self.num_of_objectives):
                self.Y[obj].append(obs)
        
        return self.X, self.Y