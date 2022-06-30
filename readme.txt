Implements algorithms for strategyproof assignment via partitioning. See the corresponding full paper here: https://arxiv.org/abs/2201.10631. Requires Gurobi for LP solving.

- main.py : runs the main algorithms for one-to-one authorship
- main_general_authorship.py : runs the algorithm for general authorship
- evaluate_results.py : plots saved results

The file "data/iclr2018.npz" is sourced from https://github.com/xycforgithub/StrategyProof_Conference_Review.
The file "data/iclr2018.json" is sources from https://github.com/Chillee/OpenReviewExplorer.
