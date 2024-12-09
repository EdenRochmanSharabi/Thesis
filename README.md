# Thesis Code: Optimal Planning Techniques in Multi-Agent Learning

This repository contains the code developed for the thesis project on optimizing resource allocation in multi-agent scenarios and analyzing the performance of various solvers. The code models a machine time allocation problem among multiple departments, each seeking to maximize its own utility given a shared capacity constraint. The implemented methods range from gradient-based optimization to evolutionary game-theoretic algorithms and search-based techniques.

## Overview

The primary objective of the code is to:
1. Formulate a resource allocation game in which multiple agents (departments) compete for a finite resource (machine hours).
2. Implement and test different computational solvers—ranging from classical optimization techniques to modern evolutionary and simulation-based algorithms.
3. Compare these solvers in terms of:
   - Total utility maximization
   - Ability to reach a Nash equilibrium
   - Computational efficiency

**Implemented Solvers:**
- **Sequential Least Squares Programming (SLSQP):**  
  Uses a gradient-based optimization method to find an optimal allocation under given constraints.  
- **Best Response Dynamics (BRD):**  
  Iteratively updates each department's allocation assuming the others remain fixed.  
- **Replicator Dynamics (RD):**  
  Models allocation strategies as evolving proportions, adjusting over time based on relative performance.  
- **Fictitious Play (FP):**  
  Adjusts strategies based on the historical average actions of other departments.  
- **Monte Carlo Tree Search (MCTS):**  
  Uses simulation-based search to explore the allocation space and iteratively refine solutions.  
- **Incremental Allocator Algorithm (IAA):**  
  Incrementally increases allocations based on probabilistic activation and local utility improvements.  
- **Stochastic Extremum Seeking Algorithm (SESA):**  
  Introduces sinusoidal, stochastic perturbations to allocations, seeking equilibria under uncertainty.

## Repository Structure

- **Department Class**:  
  Models an individual department with a defined utility function and an allocation of machine time.
  
- **MachineTimeGame Class**:  
  Represents the allocation game, containing a list of departments, total capacity, and penalty parameters. Provides methods to get/set allocations, compute total utility, and track penalty-based utility adjustments.
  
- **Solver Classes**:  
  Each solver is implemented as its own class inheriting from an abstract base `Solver` class. Each provides a `solve(game)` method that returns the final allocation:
  
  - `OptimizationSolver`: Uses SLSQP to find the allocation maximizing total utility.
  - `BestResponseSolver`: Iteratively solves best response problems until convergence.
  - `ReplicatorDynamicsSolver`: Implements evolutionary updates for allocations.
  - `FictitiousPlaySolver`: Uses historical averages to find consistent responses.
  - `MCTSSolver`: Employs a tree search to simulate and backtrack utilities.
  - `AllocationAlgorithmSolver`: Incrementally improves allocations via random activation.
  - `StochasticNashEquilibriumSolver`: Uses a stochastic extremum-seeking approach.
  
- **Utility and Plotting Functions**:  
  Functions for plotting results, tabulating allocations and utilities, and generating CSV files with recorded data.

- **Run Scripts**:  
  - `detailed_run()`: Executes all solvers for a fixed set of departments and capacity, printing and plotting results.
  - `multiple_run()`: Runs multiple experiments over a range of capacities, saving aggregated results to CSV.
  - `mcts_run()`: Specialized run focusing on MCTS performance across multiple capacities.
  
- **Paper Figures**:  
  Code can generate figures used in the thesis for comparing solver performance, average utilities, and computational costs.

## Installation

**Prerequisites:**
- Python 3.8 or higher
- `numpy`, `numba`, `scipy`, `abc`, `dataclasses`, `logging`
- `tabulate`, `random`, `copy`, `matplotlib`, `multiprocessing`, `pandas`, `sympy`

Install the required packages:
```bash
pip install numpy numba scipy tabulate matplotlib pandas sympy

