# SnAKe
Repository that includes the code for the paper: ["SnAKe: Bayesian Optimization with Pathwise Exploration"](https://arxiv.org/abs/2202.00060). The paper has been accepted into NeurIPS 2022. Until the conference, please cite the the preprint:

- Folch, Jose Pablo, Shiqiang Zhang, Robert M. Lee, Behrang Shafei, David Walz, Calvin Tsay, Mark van der Wilk, and Ruth Misener. "SnAKe: Bayesian Optimization with Pathwise Exploration." arXiv preprint arXiv:2202.00060 (2022).

The BibTeX reference is:

    @article{folch2022snake,
      title={SnAKe: Bayesian Optimization with Pathwise Exploration},
      author={Folch, Jose Pablo and Zhang, Shiqiang and Lee, Robert M and Shafei, Behrang and Walz, David and Tsay, Calvin and van der Wilk, Mark and Misener, Ruth},
      journal={arXiv preprint arXiv:2202.00060},
      year={2022}}

The code allows for reproducibility of the results and figures shown in the paper. To reproduce any experimental run, use the corresponding experiment script, these are: 

- experiment.py : synchronous, synthetic benchmark
- experiment_async.py : asynchronous, synthetic benchmark
- experiment_snar.py : synchronous, SnAr benchmark
- experiment_snar_async.py : asynchronous, SnAr benchmark
- ypacarai_lake.py : Ypacarai experiments

For the figures you can use:

- resampling_vs_pd_figure.py : Figure 1 and 10
- create_graph.py : Figure 2
- experiment_pt.py : Figure 8 and 9 
- ypacarai_lake.py : Figure 4 and 7

The rest of the files correspond to:

- snake.py : Contains the main implementation of SnAKe, and the Random + TSP baseline.
- bayes_op.py : Contains the implementation of classical Bayesian Optimization methods.
- cost_functions.py : Defines the function used to calculate the cost in the SnAr benchmark.
- functions.py : Defines all benchmark functions used in the paper.
- gp_utils.py : Defines the GP class which is used by all methods in the paper.
- sampling.py : Implementation of sampling method.
- temperature_env.py : Defines the environment class that is used in all optimizations.

# Contributors

[Jose Pablo Folch](https://jpfolch.github.io). Funded by EPSRC through the Modern Statistics and Statistical Machine Learning (StatML) CDT (grant no. EP/S023151/1) and by BASF SE, Ludwigshafen am Rhein.
