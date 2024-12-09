import os
import numpy as np
from numba import jit
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import math
from tabulate import tabulate
import random
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from multiprocessing import Pool
import pandas as pd
import sympy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Department:
    """
    Represents a department in the factory.
    """
    name: str
    coefficient: float  # The 'a_i' in the utility function
    allocation: float = 0.0  # Initial machine hours allocation

    @staticmethod
    @jit(nopython=True)
    def _utility_function(x_i, coeff, total_alloc, alpha, capacity):
        """
        Computes the utility of the department with JIT optimization. JIT stands for Just in testing Eden, don't play god with package you dont fully control.
        """
        penalty = alpha * x_i * (total_alloc - capacity) / max(total_alloc, 1e-8) if total_alloc > capacity else 0
        return coeff * np.log(x_i + 1) - penalty

    def utility(self, x_i, total_alloc, alpha, capacity):
        """
        Computes the utility of the department using the static JIT-compiled function.
        """
        return Department._utility_function(x_i, self.coefficient, total_alloc, alpha, capacity)

    # def utility(self, x_i, total_allocation, alpha):
    #     """
    #     Computes the utility of the department given its allocation.
    #     """
    #     if total_allocation > 100:
    #         penalty_share = x_i / total_allocation if total_allocation > 0 else 0
    #         penalty_term = alpha * (total_allocation - 100) * penalty_share
    #     else:
    #         penalty_term = 0
    #     return self.coefficient * np.log(x_i + 1) - penalty_term

    # def utility(self, x_i, total_allocation, alpha, capacity):
    #     """
    #     Computes the utility of the department given its allocation, this is the one from the experiments.
    #     """
    #     epsilon = 1e-8
    #     if total_allocation > capacity:
    #         # Calculate the individual penalty based on the department's share of the total allocation
    #         penalty_term = alpha * x_i * (total_allocation - capacity) / (total_allocation + epsilon)
    #     else:
    #         penalty_term = 0
    #     return self.coefficient * np.log(x_i + 1) - penalty_term

    # def utility(self, x_i, total_allocation, alpha, capacity, noise_level=0.2):
    #     """
    #     Computes the utility of the department given its allocation,
    #     adding randomness to the penalty or reward term.
    #     """
    #     if total_allocation > capacity:
    #         penalty_term = alpha * x_i * (total_allocation - capacity) / total_allocation
    #     else:
    #         penalty_term = 0
    #
    #     random_penalty_factor = np.random.uniform(1 - noise_level, 1 + noise_level)
    #     random_reward_factor = np.random.uniform(1 - noise_level, 1 + noise_level)
    #
    #     random_penalty = penalty_term * random_penalty_factor
    #     random_reward = self.coefficient * np.log(x_i + 1) * random_reward_factor
    #
    #     return random_reward - random_penalty
class MachineTimeGame:
    """
    Represents the machine time allocation game among departments.
    """
    def __init__(self, departments, alpha=10, capacity=87):
        self.departments = departments  # I store the list of Department instances
        self.alpha = alpha  # Penalty parameter
        self.capacity = capacity  # Total machine time capacity or whatever you actually want to allocate

    def total_allocation(self, allocations):
        """
        Computes the total machine time allocated.
        """
        return sum(allocations)

    def get_allocations(self):
        """
        Retrieves the current allocations for all departments.
        """
        return [dept.allocation for dept in self.departments]

    def set_allocations(self, allocations):
        """
        Sets the allocations for all departments.
        """
        for dept, alloc in zip(self.departments, allocations):
            dept.allocation = alloc

    def compute_utilities(self):
        """
        Computes the utilities for all departments based on current allocations.
        """
        total_alloc = self.total_allocation(self.get_allocations())
        utilities = []
        for dept in self.departments:
            util = dept.utility(
                dept.allocation,
                total_alloc,
                alpha=self.alpha,
                capacity=self.capacity
            )
            utilities.append(util)
        return utilities
class Solver(ABC):
    """
    Abstract base class for solvers.
    """

    @abstractmethod
    def solve(self, game: MachineTimeGame):
        """
        Solves the game and returns the optimal allocations.
        """
        pass

    @staticmethod
    def compute_best_response(dept, x0, others_alloc, alpha, bounds, capacity):
        """
        Compute the best response for a department given the current allocations.
        """

        def best_response_function(x):
            x_i = x[0]
            total_alloc = others_alloc + x_i
            if total_alloc > capacity:
                return 1e6  # Penalize allocations that exceed the limit
            return -dept.utility(x_i, total_alloc, alpha, capacity=capacity)

        constraints = {'type': 'ineq', 'fun': lambda x: capacity - (others_alloc + x[0])}

        res = minimize(
            best_response_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )

        return float(res.x[0])
class OptimizationSolver(Solver):
    """
    Solver that uses optimization techniques to find the optimal allocations.
    """
    def __init__(self, initial_guess=None):
        self.initial_guess = initial_guess
        self.utility_history = []
        self.iterations = 0  # Initialize iterations
        self.success = False  # Track if optimization was successful. This solver may fail, especialemente when you have random payoffs

    def solve(self, game: MachineTimeGame):
        num_departments = len(game.departments)
        if self.initial_guess is None:
            x0 = np.full(num_departments, game.capacity / num_departments) # Distribute resources equally as a starting point if no guess is provided
        else:
            x0 = self.initial_guess

        bounds = [(0, None) for _ in range(num_departments)]
        constraints = {
            'type': 'eq',
            'fun': lambda x: sum(x) - game.capacity
        }

        def objective(allocations):
            """
            Compute the negative total utility for a given set of allocations. Steven taught me that -min f(x) = max - f(x)
            """
            total_alloc = sum(allocations)
            total_utility = 0
            for k, dept in enumerate(game.departments):
                x_i = allocations[k]
                if x_i < 0:
                    return np.inf
                util = dept.utility(
                    x_i,
                    total_alloc,
                    alpha=game.alpha,
                    capacity=game.capacity
                )
                total_utility += util
            return -total_utility

        def callback(xk):
            """
            Log utility values and increment iteration count during optimization.
            """
            total_alloc = sum(xk)
            total_utility = 0
            for k, dept in enumerate(game.departments):
                util = dept.utility(
                    xk[k],
                    total_alloc,
                    alpha=game.alpha,
                    capacity=game.capacity
                )
                total_utility += util
            self.utility_history.append(total_utility)
            self.iterations += 1

        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'disp': False},
                callback=callback
            )

            if result.success:
                game.set_allocations(result.x)
                self.success = True
                return result.x
            else:
                # Optimization failed
                self.success = False
                total_alloc = sum(x0)
                total_utility = 0
                for k, dept in enumerate(game.departments):
                    util = dept.utility(
                        x0[k],
                        total_alloc,
                        alpha=game.alpha,
                        capacity=game.capacity
                    )
                    total_utility += util
                self.utility_history.append(total_utility)
                self.iterations = 1
                game.set_allocations(np.zeros(num_departments))
                return None
        except Exception as e:

            self.success = False
            self.iterations = 0
            game.set_allocations(np.zeros(num_departments))
            return None
class BestResponseSolver(Solver):
    def __init__(self, max_iterations=5000, tolerance=1e-3):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.utility_history = []
        self.iterations = 0

    def solve(self, game):
        num_departments = len(game.departments)
        allocations = np.full(num_departments, game.capacity / num_departments) # Start with an equal allocation for all departments

        for iteration in range(1, self.max_iterations + 1):
            self.iterations = iteration
            old_allocations = allocations.copy()
            # the algo perse
            for i, dept in enumerate(game.departments):
                others_sum = sum(old_allocations) - old_allocations[i]
                upper_bound = max(0, game.capacity - others_sum)
                bounds = [(0, upper_bound)]
                x0 = np.array([allocations[i]])

                allocations[i] = Solver.compute_best_response(
                    dept, x0, others_sum, game.alpha, bounds, game.capacity
                )

            game.set_allocations(allocations)
            total_utility = sum(game.compute_utilities())
            self.utility_history.append(total_utility)
            # un checkeo rapidito
            if is_nash_equilibrium(game, allocations, tolerance=self.tolerance):
                break

        game.set_allocations(allocations)
        return allocations
class ReplicatorDynamicsSolver(Solver):
    def __init__(self, time_steps=10000, delta_t=1e-3, tolerance=1e-6):
        self.time_steps = time_steps
        self.delta_t = delta_t
        self.tolerance = tolerance
        self.utility_history = []
        self.iterations = 0

    def solve(self, game: MachineTimeGame):
        num_departments = len(game.departments)
        allocations = np.full(num_departments, game.capacity / num_departments) # Lo mismo que en BR

        for t in range(self.time_steps):
            self.iterations = t + 1
            old_allocations = allocations.copy()
            total_alloc = sum(allocations)
            utilities = []
            # Compute utility for each department
            for i, dept in enumerate(game.departments):
                util = dept.utility(allocations[i], total_alloc, game.alpha, game.capacity)
                utilities.append(util)

            total_utility = sum(utilities)
            # self.utility_history.append(total_utility)

            average_utility = total_utility / num_departments if num_departments > 0 else 1 # Compute the average utility

            for j in range(num_departments):
                allocations[j] += self.delta_t * allocations[j] * (utilities[j] - average_utility)

            allocations = np.maximum(allocations, 0)

            total_alloc = sum(allocations) # Normalize
            if total_alloc > 0:
                allocations = (allocations / total_alloc) * game.capacity
            else:
                allocations = np.full(num_departments, game.capacity / num_departments)

            if t % 100 == 0 or t == self.time_steps - 1:
                self.utility_history.append(total_utility)

            if np.linalg.norm(allocations - old_allocations) < self.tolerance:
                break

        game.set_allocations(allocations)
        return allocations
class FictitiousPlaySolver(Solver):
    def __init__(self, max_iterations=1000, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.utility_history = []
        self.iterations = 0

    def solve(self, game):
        num_departments = len(game.departments)
        allocations = np.full(num_departments, game.capacity / num_departments)
        historical_allocations = np.zeros((self.max_iterations + 1, num_departments))
        historical_allocations[0] = allocations.copy()

        for iteration in range(1, self.max_iterations + 1):
            self.iterations = iteration  # Record current iteration
            old_allocations = allocations.copy()
            avg_allocations = np.mean(historical_allocations[:iteration], axis=0)
            # Compute the BR per dept
            for i, dept in enumerate(game.departments):
                others_avg_alloc = sum(avg_allocations) - avg_allocations[i]
                others_avg_alloc = min(others_avg_alloc, game.capacity)
                bounds = [(0, None)]
                x0 = np.array([allocations[i]])

                allocations[i] = Solver.compute_best_response(
                    dept, x0, others_avg_alloc, game.alpha, bounds, game.capacity
                )

            historical_allocations[iteration] = allocations.copy()  # Update historical allocations with the current allocations

            game.set_allocations(allocations) # Update the game's allocations
            total_utility = sum(game.compute_utilities()) # Compute and optionally store the total utility
            # self.utility_history.append(total_utility)

            if iteration % 100 == 0 or iteration == self.max_iterations:
                self.utility_history.append(total_utility)

            if np.linalg.norm(allocations - old_allocations) < self.tolerance:
                break

        total_alloc = sum(allocations)
        if total_alloc > 0:
            allocations = (allocations / total_alloc) * game.capacity
        else:
            allocations = np.full(num_departments, game.capacity / num_departments)

        game.set_allocations(allocations)
        return allocations
class MCTSSolver(Solver):
    def __init__(self, iterations=1000, exploration_constant=math.sqrt(2), num_buckets=10, total_hours=100, spread=20):
        self.iterations = 0
        self.max_iterations = iterations
        self.exploration_constant = exploration_constant
        self.num_buckets = num_buckets # Number of discrete allocation buckets for possible actions
        self.total_hours = total_hours
        self.spread = spread # Spread to define variability around proportional allocations
        self.utility_history = []

    def solve(self, game: MachineTimeGame):
        total_hours = game.capacity
        self.total_hours = total_hours
        # Compute initial allocation proportional to department coefficients
        coefficients = [dept.coefficient for dept in game.departments]
        total_coeff = sum(coefficients)
        initial_state = [coeff / total_coeff * total_hours for coeff in coefficients]

        root = MCTSNode(
            state=initial_state,
            parent=None,
            game=game,
            department_index=0,
            num_buckets=self.num_buckets,
            total_hours=self.total_hours,
            spread=self.spread
        )

        best_utility = float('-inf')
        best_allocation = None

        for iteration in range(self.max_iterations):
            self.iterations = iteration + 1
            node = root

            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_constant)

            # Expansion
            if not node.is_terminal():
                node = node.expand()

            # Simulation
            reward = node.simulation()

            # Update the best solution if found
            if reward > best_utility:
                best_utility = reward
                best_allocation = node.get_normalized_state()

            # Record the best utility so far
            # self.utility_history.append(best_utility)

            # Backpropagation
            node.backpropagate(reward)

            # if iteration % 100 == 0 or iteration == self.max_iterations - 1:
            #     self.utility_history.append(best_utility)
            self.utility_history.append(best_utility)
        if best_allocation is None and root.children:
            best_child = max(root.children, key=lambda n: n.visits)
            best_allocation = best_child.get_normalized_state()

        game.set_allocations(best_allocation)
        return best_allocation
class MCTSNode:
    def __init__(self, state, parent, game, department_index, num_buckets, total_hours=100, spread=20):
        self.state = state
        self.parent = parent
        self.game = game
        self.department_index = department_index
        self.num_buckets = num_buckets
        self.total_hours = total_hours
        self.spread = spread
        self.children = []
        self.visits = 0
        self.total_reward = 0
        self.untried_actions = self._get_possible_actions()

    def _get_possible_actions(self):
        """Generate smarter possible actions based on coefficient proportions."""
        if self.department_index >= len(self.state):
            return []

        dept_coefficient = self.game.departments[self.department_index].coefficient
        total_coefficient = sum(d.coefficient for d in self.game.departments)
        target_proportion = dept_coefficient / total_coefficient

        # Generate actions around the target proportion
        base_allocation = target_proportion * self.total_hours
        spread = self.spread
        min_alloc = max(0, base_allocation - spread)
        max_alloc = min(self.total_hours, base_allocation + spread)

        # Create discrete steps within this range
        step_size = (max_alloc - min_alloc) / self.num_buckets if self.num_buckets > 0 else max_alloc - min_alloc
        actions = [min_alloc + i * step_size for i in range(self.num_buckets + 1)]
        return actions

    def is_terminal(self):
        return self.department_index >= len(self.state) - 1

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand(self):
        action = self.untried_actions.pop()
        new_state = self.state.copy()
        new_state[self.department_index] = action

        child = MCTSNode(
            state=new_state,
            parent=self,
            game=self.game,
            department_index=self.department_index + 1,
            num_buckets=self.num_buckets,
            total_hours=self.total_hours,
            spread=self.spread
        )
        self.children.append(child)
        return child

    def simulation(self):
        if self.is_terminal():
            # Normalize the final state and evaluate
            normalized_state = self.get_normalized_state()
            self.game.set_allocations(normalized_state)
            return sum(self.game.compute_utilities())

        # For non-terminal nodes, complete the allocation proportionally
        current_state = self.state.copy()
        remaining_departments = len(current_state) - self.department_index - 1

        if remaining_departments > 0:
            # Calculate remaining hours
            used_hours = sum(current_state[:self.department_index + 1])
            remaining_hours = self.total_hours - used_hours
            if remaining_hours < 0:
                # Set allocations for remaining departments to zero
                for idx in range(self.department_index + 1, len(current_state)):
                    current_state[idx] = 0
            else:
                # Get coefficients for remaining departments
                remaining_coeffs = [self.game.departments[i].coefficient
                                    for i in range(self.department_index + 1, len(current_state))]
                total_remaining_coeff = sum(remaining_coeffs)

                # Allocate proportionally
                for i, coeff in enumerate(remaining_coeffs):
                    idx = self.department_index + 1 + i
                    if i == len(remaining_coeffs) - 1:
                        # Last department gets whatever is left
                        current_state[idx] = remaining_hours
                    else:
                        # Proportional allocation
                        alloc = (coeff / total_remaining_coeff) * remaining_hours
                        current_state[idx] = alloc
                        remaining_hours -= alloc

        # Evaluate the complete allocation
        normalized_state = self._normalize_allocation(current_state)
        self.game.set_allocations(normalized_state)
        return sum(self.game.compute_utilities())

    def backpropagate(self, reward):
        node = self
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def best_child(self, exploration_constant):
        if not self.children:
            return None

        # UCB1 formula with normalized rewards
        max_reward = max(child.total_reward for child in self.children)
        min_reward = min(child.total_reward for child in self.children)
        reward_range = max_reward - min_reward if max_reward > min_reward else 1

        def ucb1(child):
            exploitation = (child.total_reward - min_reward) / reward_range
            exploration = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration

        return max(self.children, key=ucb1)

    def get_normalized_state(self):
        """Return the state with allocations normalized to sum to total_hours."""
        return self._normalize_allocation(self.state)

    def _normalize_allocation(self, allocation):
        # Ensure allocations are non-negative
        allocation = [max(0, x) for x in allocation]
        total = sum(allocation)
        if total == 0:
            return [self.total_hours / len(allocation)] * len(allocation)
        return [x * (self.total_hours / total) for x in allocation]
class AllocationAlgorithmSolver(Solver):
    """
    Solver that implements the iterative allocation algorithm.
    """
    def __init__(self, delta=0.1, activation_prob=0.5, max_iterations=10000):
        self.delta = delta
        self.activation_prob = activation_prob
        self.max_iterations = max_iterations
        self.utility_history = []
        self.iterations = 0  # Initialize iterations

    def solve(self, game: MachineTimeGame):
        self.game = game  # Store game reference to access capacity
        num_departments = len(game.departments)
        allocations = np.zeros(num_departments)  # Initial allocations: W(0) = 0
        total_alloc = sum(allocations)  # Total allocation W
        t = 0  # Time step t

        while total_alloc < game.capacity and t < self.max_iterations:
            old_allocations = allocations.copy()
            total_alloc = sum(allocations)  # Update W(t)

            for i, dept in enumerate(game.departments):
                if np.random.rand() < self.activation_prob:  # Activation probability ν_x^act
                    # Possible new allocation by increasing delta
                    if total_alloc + self.delta <= game.capacity:
                        new_allocations = allocations.copy()
                        new_allocations[i] += self.delta  # Increment allocation by δ for agent x
                        new_total_alloc = total_alloc + self.delta  # Update W' = W + δ

                        # Compute utility difference for department i
                        current_utility = dept.utility(
                            allocations[i],  # Current allocation W_x
                            total_alloc,  # Current total allocation W
                            game.alpha,  # Penalty parameter α
                            game.capacity
                        )
                        new_utility = dept.utility(
                            new_allocations[i],  # Potential new allocation W'_x
                            new_total_alloc,  # New total allocation W'
                            game.alpha,
                            game.capacity
                        )
                        delta_utility = new_utility - current_utility  # ΔUtility

                        # If utility increases, accept the new allocation
                        if delta_utility > 0:
                            allocations = new_allocations  # Update allocation state W(t+1) = W'
                            total_alloc = new_total_alloc  # Update total allocation W to W'
            t += 1  # Increment time step t = t + 1

            # Record total utility
            game.set_allocations(allocations)
            total_utility = sum(game.compute_utilities())
            # self.utility_history.append(total_utility)  # Record utility at each iteration

            if t % 100 == 0 or t == self.max_iterations - 1:
                self.utility_history.append(total_utility)

            # Check for convergence
            if np.allclose(allocations, old_allocations):  # Convergence check
                break

        # Normalize allocations to sum to capacity
        allocations = self._normalize_allocations(allocations)  # Normalize W to satisfy constraint
        game.set_allocations(allocations)
        return allocations

    def _normalize_allocations(self, allocations):
        total = sum(allocations)
        if total == 0:
            return np.full(len(allocations), self.game.capacity / len(allocations))
        return allocations * (self.game.capacity / total)
class StochasticNashEquilibriumSolver(Solver):
    """
    Solver that uses a stochastic extremum-seeking approach to find Nash Equilibrium.
    """
    def __init__(self, max_iterations=10000, dt=0.01, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.dt = dt
        self.tolerance = tolerance
        self.utility_history = []
        self.iterations = 0

    def solve(self, game: MachineTimeGame):
        num_departments = len(game.departments)
        capacity = game.capacity

        hat_u = np.full(num_departments, capacity / num_departments) # Average allocation estimates
        a_i = np.full(num_departments, 0.1) # Amplitude of perturbations
        k_i = np.full(num_departments, 0.1) # Learning rate for updates
        eta_i = np.zeros(num_departments) # Phase variables for sinusoidal perturbations
        theta = 1.0 # Decay rate for phase variables
        sigma = 0.1 # Noise level in perturbations

        for iteration in range(1, self.max_iterations + 1):
            self.iterations = iteration  # Record the current iteration
            old_hat_u = hat_u.copy()  # Save the previous estimate for convergence check

            # Update the phase variables with stochastic perturbation
            eta_i += -theta * eta_i * self.dt + sigma * np.sqrt(self.dt) * np.random.randn(num_departments)
            f_eta = np.sin(eta_i)  # Compute sinusoidal perturbations
            u_i = hat_u + a_i * f_eta  # Apply perturbations to the allocation estimates

            # Ensure allocations are within valid bounds
            u_i = np.clip(u_i, 0, capacity)
            total_u = sum(u_i)
            if total_u > capacity:
                # Normalize allocations to respect total capacity
                u_i = (u_i / total_u) * capacity

            total_alloc = sum(u_i)
            utilities = []
            for i, dept in enumerate(game.departments):
                util = dept.utility(u_i[i], total_alloc, game.alpha, game.capacity)
                utilities.append(util)

            J_i = np.array(utilities)  # Utility gradients for each dept
            hat_u += self.dt * k_i * a_i * f_eta * J_i # Update the allocation estimates based on feedback
            total_utility = sum(utilities)
            # self.utility_history.append(total_utility)

            if iteration % 100 == 0 or iteration == self.max_iterations:
                self.utility_history.append(total_utility)

            if np.linalg.norm(hat_u - old_hat_u) < self.tolerance:
                break

        final_allocations = np.clip(hat_u, 0, capacity)
        total_alloc = sum(final_allocations)
        if total_alloc > capacity:
            final_allocations = (final_allocations / total_alloc) * capacity

        game.set_allocations(final_allocations)
        return final_allocations

def plot_allocations(solver_games, department_labels):
    """Plots allocation data for each solver."""
    solver_allocations = {
        solver_label: [dept.allocation for dept in game_instance.departments]
        for solver_label, game_instance in zip(
            ['Optimization', 'Best Response', 'Replicator Dynamics',
             'Fictitious Play', 'MCTS', 'Allocation Algorithm'], solver_games)
    }

    plt.figure(figsize=(12, 6))
    for solver_label, allocation_values in solver_allocations.items():
        plt.plot(department_labels, allocation_values, label=solver_label, marker='o', linestyle='-', markersize=4)

    plt.title("Machine Time Allocations by Solver")
    plt.xlabel("Departments")
    plt.ylabel("Allocation (hours)")
    plt.legend()
    plt.xticks(rotation=90)  # Rotate department labels for readability
    plt.tight_layout()
    plt.show()
def plot_utilities(solver_games, department_labels):
    """Plots utility data for each solver."""
    solver_utilities = {
        solver_label: game_instance.compute_utilities()
        for solver_label, game_instance in zip(
            ['Optimization', 'Best Response', 'Replicator Dynamics',
             'Fictitious Play', 'MCTS', 'Allocation Algorithm'], solver_games)
    }

    plt.figure(figsize=(12, 6))
    for solver_label, utility_values in solver_utilities.items():
        plt.plot(department_labels, utility_values, label=solver_label, marker='o', linestyle='-', markersize=4)

    plt.title("Utilities by Solver")
    plt.xlabel("Departments")
    plt.ylabel("Utility")
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
def plot_utility_convergence(solvers, solver_names, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    for solver, name in zip(solvers, solver_names):
        if hasattr(solver, 'utility_history') and solver.utility_history:
            print(f"Generating plot for {name}...")
            plt.figure(figsize=(8, 6))
            iterations = range(1, len(solver.utility_history) + 1)
            plt.plot(iterations, solver.utility_history, label=name)
            plt.xlabel('Iteration')
            plt.ylabel('Total Utility')
            plt.title(f'Utility vs Iteration for {name}')
            plt.legend()
            plt.grid(True)


            formatter = ScalarFormatter(useOffset=False, useMathText=False)
            formatter.set_scientific(False)
            plt.gca().xaxis.set_major_formatter(formatter)
            plt.gca().yaxis.set_major_formatter(formatter)

            plt.tight_layout()
            plot_filename = os.path.join(save_directory, f'{name}_utility_convergence.png')
            plt.savefig(plot_filename)
            plt.close()
            print(f"Plot saved to {plot_filename}")
        else:
            print(f"No utility history for {name}, skipping plot.")

def generate_allocation_table(games_local, department_local):
    allocation_data = []
    for i, dept_name in enumerate(department_local):
        row = [dept_name] + [game.departments[i].allocation for game in games_local]
        allocation_data.append(row)

    headers = ['Department', 'Optimization', 'Best Response', 'Replicator Dynamics',
               'Fictitious Play', 'MCTS', 'Allocation Algorithm', 'Stochastic Nash Equilibrium']
    print("\nMachine Time Allocations (hours):")
    print(tabulate(allocation_data, headers=headers, floatfmt='.2f', tablefmt='grid'))
def generate_utility_table(games_local, department_local):
    utilities = [game.compute_utilities() for game in games_local]
    utility_data = []

    for i, dept_name in enumerate(department_local):
        row = [dept_name] + [utilities[j][i] for j in range(len(games_local))]
        utility_data.append(row)

    utility_data.append([
        'Total'
    ] + [sum(utility) for utility in utilities])

    headers = ['Department', 'Optimization', 'Best Response', 'Replicator Dynamics',
               'Fictitious Play', 'MCTS', 'Allocation Algorithm', 'Stochastic Nash Equilibrium']
    print("\nUtilities by Department:")
    print(tabulate(utility_data, headers=headers, floatfmt='.2f', tablefmt='grid'))
def generate_comparison_table(games_local, department_local):
    comparison_data = []
    opt_allocations = [dept.allocation for dept in games_local[0].departments]

    for i, dept_name in enumerate(department_local):
        row = [
            dept_name
        ] + [opt_allocations[i] - game.departments[i].allocation for game in games_local[1:]]
        comparison_data.append(row)

    diff_headers = ['Department', 'Opt - BR', 'Opt - RD', 'Opt - FP', 'Opt - MCTS', 'Opt - AllocAlg', 'Opt - StochasticNE']
    print("\nAllocation Differences from Optimization (hours):")
    print(tabulate(comparison_data, headers=diff_headers, floatfmt='+.2f', tablefmt='grid'))
def generate_utility_difference_table(games_local):
    opt_utility = sum(games_local[0].compute_utilities())
    utility_diffs = [
        opt_utility - sum(game.compute_utilities()) for game in games_local[1:]
    ]

    utility_diff_data = [
        ['Total Utility Difference'] + utility_diffs
    ]

    diff_headers = ['Total Utility Difference', 'Opt - BR', 'Opt - RD', 'Opt - FP', 'Opt - MCTS', 'Opt - AllocAlg', 'Opt - StochasticNE']
    print("\nUtility Differences from Optimization:")
    print(tabulate(utility_diff_data, headers=diff_headers, floatfmt='+.2f', tablefmt='grid'))
def create_departments(num_departments=10, lowerbound=1, upperbound=10):
    """Creates and returns a list of Department instances with unique names and random coefficients."""
    departments = []
    names = []
    for i in range(1, num_departments + 1):
        name = f'Dept_{i}'
        coefficient = random.randint(lowerbound, upperbound)
        departments.append(Department(name=name, coefficient=coefficient))
        names.append(name)
    return departments, names
def parallelization_solver(args):
    solver, game = args
    solver.solve(game)
    return solver, game
def generate_prime_capacities(start, end):
    """Generates a list of prime numbers within a given range."""
    primes = []
    for num in range(start, end + 1):
        if num > 1:
            is_prime = True
            for i in range(2, int(math.isqrt(num)) + 1):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)
    return primes
def is_nash_equilibrium(game: MachineTimeGame, allocations, tolerance=1e-2):
    """
    Checks if the given allocations constitute a Nash Equilibrium.
    Prints the differences between current allocations and best responses for each department if not in equilibrium.
    """
    total_alloc = sum(allocations)
    for i, dept in enumerate(game.departments):
        others_alloc = total_alloc - allocations[i]
        bounds = [(0, None)]
        x0 = np.array([allocations[i]])

        best_response_alloc = Solver.compute_best_response(
            dept=dept,
            x0=x0,
            others_alloc=others_alloc,
            alpha=game.alpha,
            bounds=bounds,
            capacity=game.capacity
        )

        # Calculate the difference between the current allocation and the best response
        diff = abs(best_response_alloc - allocations[i])

        # If not within tolerance, print details and return False
        if diff > tolerance:
            # print(f"Department {i}: Allocation = {allocations[i]}, Best Response = {best_response_alloc}, Difference = {diff}")
            return False

    return True
def detailed_run():
    # Initialize base departments and names
    base_departments, department_names = create_departments(num_departments=9, lowerbound=0, upperbound=5000)
    capacity = 1888
    # Deep copies for each solver instance
    departments_opt = copy.deepcopy(base_departments)
    departments_br = copy.deepcopy(base_departments)
    departments_rd = copy.deepcopy(base_departments)
    departments_fp = copy.deepcopy(base_departments)
    departments_mcts = copy.deepcopy(base_departments)
    departments_alloc = copy.deepcopy(base_departments)
    departments_stochastic = copy.deepcopy(base_departments)  # New for Stochastic Solver

    game_opt = MachineTimeGame(departments=departments_opt, alpha=10, capacity=capacity)
    game_br = MachineTimeGame(departments=departments_br, alpha=10, capacity=capacity)
    game_rd = MachineTimeGame(departments=departments_rd, alpha=10, capacity=capacity)
    game_fp = MachineTimeGame(departments=departments_fp, alpha=10, capacity=capacity)
    game_mcts = MachineTimeGame(departments=departments_mcts, alpha=10, capacity=capacity)
    game_alloc = MachineTimeGame(departments=departments_alloc, alpha=10, capacity=capacity)
    game_stochastic = MachineTimeGame(departments=departments_stochastic, alpha=10, capacity=capacity)  # New

    solver_opt = OptimizationSolver()
    solver_br = BestResponseSolver(max_iterations=5000, tolerance=1e-5)
    solver_rd = ReplicatorDynamicsSolver(time_steps=8500, delta_t=1e-8, tolerance=1e-5)
    solver_fp = FictitiousPlaySolver(max_iterations=5000, tolerance=1e-5)
    solver_mcts = MCTSSolver(iterations=10000, exploration_constant=math.sqrt(2), num_buckets=10,
                             total_hours=game_mcts.capacity)
    solver_alloc = AllocationAlgorithmSolver(delta=0.00001, activation_prob=0.01, max_iterations=1000000)
    solver_stochastic = StochasticNashEquilibriumSolver(max_iterations=1000000, dt=0.001, tolerance=1e-6)

    solvers_and_games = [
        (solver_opt, game_opt),
        (solver_br, game_br),
        (solver_rd, game_rd),
        (solver_fp, game_fp),
        (solver_mcts, game_mcts),
        (solver_alloc, game_alloc),
        (solver_stochastic, game_stochastic)
    ]

    solvers = [solver_opt, solver_br, solver_rd, solver_fp, solver_mcts, solver_alloc, solver_stochastic]
    games = [game_opt, game_br, game_rd, game_fp, game_mcts, game_alloc, game_stochastic]
    solver_names = ['Optimization', 'Best Response', 'Replicator Dynamics', 'Fictitious Play', 'MCTS',
                    'Allocation Algorithm', 'Stochastic Nash Equilibrium']

    # Run solvers sequentially to access their utility histories
    for solver, game in zip(solvers, [game_opt, game_br, game_rd, game_fp, game_mcts, game_alloc, game_stochastic]):
        solver.solve(game)
    # Update tables to include StochasticNashEquilibriumSolver
    generate_allocation_table(games, department_names)
    generate_utility_table(games, department_names)
    generate_comparison_table(games, department_names)
    generate_utility_difference_table(games)

    total_utilities = {'Capacity': capacity}
    nash_equilibrium_results = {'Capacity': capacity}
    results = []
    nash_results = []
    solver_names = ['Optimization', 'Best Response', 'Replicator Dynamics',
                    'Fictitious Play', 'MCTS', 'Allocation Algorithm', 'Stochastic Nash Equilibrium']  # Update list

    for solver_name, game in zip(solver_names, games):
        total_utility = sum(game.compute_utilities())
        total_utilities[solver_name] = total_utility
        allocations = [dept.allocation for dept in game.departments]
        is_nash = is_nash_equilibrium(game, allocations, tolerance=1e-2)
        nash_equilibrium_results[solver_name] = is_nash

    print(nash_equilibrium_results)

    # Define the directory to save plots
    save_directory = '/Users/edenrochman/Library/Mobile Documents/com~apple~CloudDocs/Maastricht_DKE/Year 4/Thesis/factory/pythonProject1/plots'

    # Plot utility convergence and save the plots
    plot_utility_convergence(solvers, solver_names, save_directory)
def multiple_run():
    lower_limit = 100
    upper_limit = 1100
    capacities = [n for n in range(lower_limit, upper_limit + 1) if
                  not all(n % i != 0 for i in range(2, int(n ** 0.5) + 1)) and n > 1]
    iterations = sympy.primepi(upper_limit) - sympy.primepi(lower_limit - 1)
    last_prime = sympy.prevprime(upper_limit + 1)
    print("Number of iterations: ", iterations)
    results = []
    nash_results = []
    iteration_data = []
    utility_history_data = []

    solver_names = ['Optimization', 'Best Response', 'Replicator Dynamics',
                    'Fictitious Play', 'MCTS', 'Allocation Algorithm', 'Stochastic Nash Equilibrium']

    random.seed(42)

    for capacity in capacities:
        print("Current capacity", capacity, " out of ", last_prime)
        base_departments, department_names = create_departments(num_departments=100, lowerbound=0, upperbound=5000)

        # Deep copies for each solver instance
        departments_opt = copy.deepcopy(base_departments)
        departments_br = copy.deepcopy(base_departments)
        departments_rd = copy.deepcopy(base_departments)
        departments_fp = copy.deepcopy(base_departments)
        departments_mcts = copy.deepcopy(base_departments)
        departments_alloc = copy.deepcopy(base_departments)
        departments_stochastic = copy.deepcopy(base_departments)

        # Create game instances for each solver
        game_opt = MachineTimeGame(departments=departments_opt, alpha=10, capacity=capacity)
        game_br = MachineTimeGame(departments=departments_br, alpha=10, capacity=capacity)
        game_rd = MachineTimeGame(departments=departments_rd, alpha=10, capacity=capacity)
        game_fp = MachineTimeGame(departments=departments_fp, alpha=10, capacity=capacity)
        game_mcts = MachineTimeGame(departments=departments_mcts, alpha=10, capacity=capacity)
        game_alloc = MachineTimeGame(departments=departments_alloc, alpha=10, capacity=capacity)
        game_stochastic = MachineTimeGame(departments=departments_stochastic, alpha=10, capacity=capacity)

        # Calculate adaptive spread and num_buckets
        num_departments = len(game_mcts.departments)
        average_allocation = capacity / num_departments
        spread_factor = 0.5  # You can adjust this value, play around
        steps_per_average_alloc = 10  # You can adjust this value, play around

        spread = int(spread_factor * average_allocation)
        desired_step_size = average_allocation / steps_per_average_alloc
        total_range = 2 * spread  # Since spread is ± around the base allocation

        num_buckets = max(int(total_range / desired_step_size), 1)

        # Create solver instances
        solver_opt = OptimizationSolver()
        solver_br = BestResponseSolver(max_iterations=5000, tolerance=1e-8)
        solver_rd = ReplicatorDynamicsSolver(time_steps=10000, delta_t=1e-8, tolerance=1e-5)
        solver_fp = FictitiousPlaySolver(max_iterations=5000, tolerance=1e-5)
        solver_mcts = MCTSSolver(iterations=10000, exploration_constant=math.sqrt(2), num_buckets=num_buckets,
                                 total_hours=game_mcts.capacity, spread=spread)
        solver_alloc = AllocationAlgorithmSolver(delta=1, activation_prob=0.8, max_iterations=1000000)
        solver_stochastic = StochasticNashEquilibriumSolver(max_iterations=100000, dt=0.001, tolerance=1e-6)

        # Pair solvers with their corresponding game instances
        solvers_and_games = [
            (solver_opt, game_opt),
            (solver_br, game_br),
            (solver_rd, game_rd),
            (solver_fp, game_fp),
            (solver_mcts, game_mcts),
            (solver_alloc, game_alloc),
            (solver_stochastic, game_stochastic)
        ]

        # Parallel computation for each solver
        with Pool(processes=len(solvers_and_games)) as pool:
            solver_game_pairs = pool.map(parallelization_solver, solvers_and_games)

        solvers, games = zip(*solver_game_pairs)

        # Collect results for each capacity
        total_utilities = {'Capacity': capacity}
        nash_equilibrium_results = {'Capacity': capacity}

        for solver_name, solver, game in zip(solver_names, solvers, games):
            # Check if solver was successful
            if getattr(solver, 'success', True):
                total_utility = sum(game.compute_utilities())
                iterations = solver.iterations
                # Get allocations
                allocations = [dept.allocation for dept in game.departments]
                # Test if the allocation is a Nash Equilibrium
                is_nash = is_nash_equilibrium(game, allocations, tolerance=1e-2)
            else:
                print(f"Solver {solver_name} failed at capacity {capacity}")
                total_utility = None
                iterations = None
                is_nash = False  # Or set to None if you prefer

            total_utilities[solver_name] = total_utility
            nash_equilibrium_results[solver_name] = is_nash

            # Collect iteration data
            iteration_data.append({
                'Capacity': capacity,
                'Solver': solver_name,
                'Total Utility': total_utility,
                'Iterations': iterations
            })
            if hasattr(solver, 'utility_history') and solver.utility_history:
                for iteration_idx, utility in enumerate(solver.utility_history):
                    utility_history_data.append({
                        'Capacity': capacity,
                        'Solver': solver_name,
                        'Iteration': iteration_idx + 1,
                        'Utility': utility
                    })

        results.append(total_utilities)
        nash_results.append(nash_equilibrium_results)

        # Optional: Print results for each capacity
        print(f"Capacity: {capacity}")
        solver_utilities = total_utilities.copy()
        solver_utilities.pop('Capacity', None)
        sorted_solvers = sorted(
            [(name, util) for name, util in solver_utilities.items() if util is not None],
            key=lambda item: item[1], reverse=True)
        for solver_name, utility in sorted_solvers:
            print(f"  {solver_name}: Total Utility = {utility:.2f}")
        print()

    # Save total utilities to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('fixed_mcts_total_utilities_by_capacity.csv', index=False)

    # Save Nash equilibrium results to CSV
    df_nash_results = pd.DataFrame(nash_results)
    df_nash_results.to_csv('fixed_mcts_nash_equilibrium_results.csv', index=False)

    # Save iteration data to CSV
    df_iteration_data = pd.DataFrame(iteration_data)
    df_iteration_data.to_csv('fixed_mcts_iterations_utilities_capacities.csv', index=False)

    # Save utility history data to CSV
    df_utility_history = pd.DataFrame(utility_history_data)
    df_utility_history.to_csv('fixed_mcts_utility_history.csv', index=False)
def mcts_run():
    lower_limit = 100
    upper_limit = 1100  # Adjust the upper limit if needed
    capacities = [n for n in range(lower_limit, upper_limit + 1) if
                  not all(n % i != 0 for i in range(2, int(n ** 0.5) + 1)) and n > 1]
    iterations = sympy.primepi(upper_limit) - sympy.primepi(lower_limit - 1)
    last_prime = sympy.prevprime(upper_limit + 1)
    print("Number of iterations: ", iterations)
    results = []
    nash_results = []
    iteration_data = []
    utility_history_data = []

    solver_name = 'MCTS'

    random.seed(42)

    for capacity in capacities:
        print("Current capacity", capacity, " out of ", last_prime)
        base_departments, department_names = create_departments(num_departments=100, lowerbound=0, upperbound=5000)

        # Create game instance
        departments_mcts = copy.deepcopy(base_departments)
        game_mcts = MachineTimeGame(departments=departments_mcts, alpha=10, capacity=capacity)

        # Calculate adaptive spread and num_buckets
        num_departments = len(game_mcts.departments)
        average_allocation = capacity / num_departments
        spread_factor = 0.5  # You can adjust this value
        steps_per_average_alloc = 10  # You can adjust this value

        spread = int(spread_factor * average_allocation)
        desired_step_size = average_allocation / steps_per_average_alloc
        total_range = 2 * spread  # Since spread is ± around the base allocation

        num_buckets = max(int(total_range / desired_step_size), 1)

        # Create MCTS solver instance
        solver_mcts = MCTSSolver(iterations=10000, exploration_constant=math.sqrt(2), num_buckets=num_buckets,
                                 total_hours=game_mcts.capacity, spread=spread)

        # Run the solver
        solver_mcts.solve(game_mcts)

        # Collect results
        total_utility = sum(game_mcts.compute_utilities())
        iterations_count = solver_mcts.iterations
        # Get allocations
        allocations = [dept.allocation for dept in game_mcts.departments]
        # Test if the allocation is a Nash Equilibrium
        is_nash = is_nash_equilibrium(game_mcts, allocations, tolerance=1e-2)

        # Collect data
        results.append({'Capacity': capacity, 'MCTS': total_utility})
        nash_results.append({'Capacity': capacity, 'MCTS': is_nash})
        iteration_data.append({'Capacity': capacity, 'Solver': 'MCTS', 'Total Utility': total_utility, 'Iterations': iterations_count})

        if hasattr(solver_mcts, 'utility_history') and solver_mcts.utility_history:
            for iteration_idx, utility in enumerate(solver_mcts.utility_history):
                utility_history_data.append({
                    'Capacity': capacity,
                    'Solver': 'MCTS',
                    'Iteration': iteration_idx + 1,
                    'Utility': utility
                })

    # Save total utilities to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('mcts_total_utilities_by_capacity.csv', index=False)

    # Save Nash equilibrium results to CSV
    df_nash_results = pd.DataFrame(nash_results)
    df_nash_results.to_csv('mcts_nash_equilibrium_results.csv', index=False)

    # Save iteration data to CSV
    df_iteration_data = pd.DataFrame(iteration_data)
    df_iteration_data.to_csv('mcts_iterations_utilities_capacities.csv', index=False)

    # Save utility history data to CSV
    df_utility_history = pd.DataFrame(utility_history_data)
    df_utility_history.to_csv('mcts_utility_history.csv', index=False)

if __name__ == "__main__":
    choice = input("Presiona 1 para correlo una vez, 2 para serializado, 3 para testear el mcts: ")
    if choice == "1":
        detailed_run()
    elif choice == "2":
        multiple_run()
    elif choice == "3":
        mcts_run()




