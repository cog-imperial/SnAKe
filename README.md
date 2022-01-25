# SnAKe
Repository that includes the code for the paper: "SnAKe: Bayesian Optimization with Pathwise Exploration"

The code allows for reproducibility of the results and figures shown in the paper. To reproduce any experimental run, use the corresponding experiment script, these are: 

- experiment.py (synchronous, synthetic benchmark)
- experiment_async.py (asynchronous, synthetic benchmark)
- experiment_snar.py (synchronous, SnAr benchmark)
- experiment_snar_async.py (asynchronous, SnAr benchmark)

For the figures you can use:

- resampling_vs_pd_figure.py: Figure 2 and 9
- create_graph.py: Figure 3
- experiment_pt.py: Figure 7 and 8: 

As a guide for the rest of the files:

- adaptive_thompson_scheduling.py : Contains the main implementation of SnAKe, and the Random + TSP baseline.
- bayes_op.py : Contains the implementation of classical Bayesian Optimization methods.
- cost_functions.py : Defines the function used to calculate the cost in the SnAr benchmark.
- functions.py : Defines all benchmark functions used in the paper.
- gp_utils.py : Defines the GP class which is used by all methods in the paper.
- sampling.py : Implementation of sampling method.
- temperature_env.py : Defines the environment class that is used in all optimizations.