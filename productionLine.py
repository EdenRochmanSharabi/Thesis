import numpy as np
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import math
from tabulate import tabulate
import random
import copy
import matplotlib.pyplot as plt
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
    #     Computes the utility of the department given its allocation, optimized for best response dynamics.
    #     """
    #     if total_allocation > capacity:
    #         # Calculate the individual penalty based on the department's share of the total allocation
    #         penalty_term = alpha * x_i * (total_allocation - capacity) / total_allocation
    #     else:
    #         penalty_term = 0
    #     return self.coefficient * np.log(x_i + 1) - penalty_term
    def utility(self, x_i, total_allocation, alpha, capacity, noise_level=0.2):
        """
        Computes the utility of the department given its allocation,
        adding randomness to the penalty or reward term.
        """
        if total_allocation > capacity:
            penalty_term = alpha * x_i * (total_allocation - capacity) / total_allocation
        else:
            penalty_term = 0

        random_penalty_factor = np.random.uniform(1 - noise_level, 1 + noise_level)
        random_reward_factor = np.random.uniform(1 - noise_level, 1 + noise_level)

        random_penalty = penalty_term * random_penalty_factor
        random_reward = self.coefficient * np.log(x_i + 1) * random_reward_factor

        return random_reward - random_penalty
class MachineTimeGame:
    """
    Represents the machine time allocation game among departments.
    """
    def __init__(self, departments, alpha=10, capacity=87):
        self.departments = departments  # List of Department instances
        self.alpha = alpha  # Penalty parameter
        self.capacity = capacity  # Total machine time capacity

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

        return res.x[0]
class OptimizationSolver(Solver):
    """
    Solver that uses optimization techniques to find the optimal allocations.
    """
    def __init__(self, initial_guess=None):
        self.initial_guess = initial_guess

    def solve(self, game: MachineTimeGame):
        """
        Solves the game using constrained optimization.
        """
        num_departments = len(game.departments)
        if self.initial_guess is None:
            x0 = np.full(num_departments, game.capacity / num_departments)
        else:
            x0 = self.initial_guess

        bounds = [(0, None) for _ in range(num_departments)]

        constraints = {
            'type': 'eq',
            'fun': lambda x: sum(x) - game.capacity  # Ensure total allocation equals capacity
        }

        def objective(allocations):
            total_alloc = sum(allocations)
            total_utility = 0
            for k, dept in enumerate(game.departments):
                util = dept.utility(
                    allocations[k],
                    total_alloc,
                    alpha=game.alpha,
                    capacity=game.capacity
                )
                total_utility += util
            return -total_utility

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )

        try:
            if result.success:
                game.set_allocations(result.x)
                return result.x
            else:
                raise ValueError("Optimization failed.")
        except ValueError as e:
            print(f"Error: {e}")
class BestResponseSolver(Solver):
    def __init__(self, max_iterations=1000, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(self, game):
        num_departments = len(game.departments)
        allocations = np.full(num_departments, game.capacity / num_departments)

        for iteration in range(self.max_iterations):
            old_allocations = allocations.copy()

            for i, dept in enumerate(game.departments):
                others_sum = sum(old_allocations) - old_allocations[i]
                bounds = [(0, game.capacity - others_sum)]
                x0 = np.array([allocations[i]])

                allocations[i] = Solver.compute_best_response(
                    dept, x0, others_sum, game.alpha, bounds, game.capacity
                )

            if np.linalg.norm(allocations - old_allocations) < self.tolerance:
                game.set_allocations(allocations)
                return allocations

        logger.warning("Maximum iterations reached without convergence")
        game.set_allocations(allocations)
        return allocations
class ReplicatorDynamicsSolver(Solver):
    """
    Solver that uses replicator dynamics to find an equilibrium.
    """
    def __init__(self, time_steps=10000, delta_t=1e-3, tolerance=1e-6):
        self.time_steps = time_steps
        self.delta_t = delta_t
        self.tolerance = tolerance

    def solve(self, game: MachineTimeGame):
        num_departments = len(game.departments)
        allocations = np.full(num_departments, game.capacity / num_departments)

        for t in range(self.time_steps):
            old_allocations = allocations.copy()
            total_alloc = sum(allocations)
            utilities = []

            for i, dept in enumerate(game.departments):
                util = dept.utility(allocations[i], total_alloc, game.alpha, game.capacity)
                utilities.append(util)

            average_utility = sum(utilities) / num_departments if num_departments > 0 else 1

            for j in range(num_departments):
                allocations[j] += self.delta_t * allocations[j] * (utilities[j] - average_utility)

            allocations = np.maximum(allocations, 0)

            total_alloc = sum(allocations)
            if total_alloc > 0:
                allocations = (allocations / total_alloc) * game.capacity
            else:
                allocations = np.full(num_departments, game.capacity / num_departments)

            if np.linalg.norm(allocations - old_allocations) < self.tolerance:
                break

        game.set_allocations(allocations)
        return allocations
class FictitiousPlaySolver(Solver):
    def __init__(self, max_iterations=1000, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(self, game):
        num_departments = len(game.departments)
        allocations = np.full(num_departments, game.capacity / num_departments)
        historical_allocations = np.zeros((self.max_iterations + 1, num_departments))
        historical_allocations[0] = allocations.copy()

        for iteration in range(1, self.max_iterations + 1):
            old_allocations = allocations.copy()
            avg_allocations = np.mean(historical_allocations[:iteration], axis=0)

            for i, dept in enumerate(game.departments):
                others_avg_alloc = sum(avg_allocations) - avg_allocations[i]
                others_avg_alloc = min(others_avg_alloc, game.capacity)
                bounds = [(0, None)]
                x0 = np.array([allocations[i]])

                allocations[i] = Solver.compute_best_response(
                    dept, x0, others_avg_alloc, game.alpha, bounds, game.capacity
                )

            historical_allocations[iteration] = allocations.copy()

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
    """
    Improved MCTS solver that uses smart rollout and better state evaluation.
    """

    def __init__(self, iterations=1000, exploration_constant=math.sqrt(2), num_buckets=10, total_hours=100, spread=20):
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.num_buckets = num_buckets
        self.total_hours = total_hours
        self.spread = spread

    def solve(self, game: MachineTimeGame):
        total_hours = game.capacity
        self.total_hours = total_hours

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

        for _ in range(self.iterations):
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

            # Backpropagation
            node.backpropagate(reward)

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
        spread = self.spread  # Allow ±spread variation around the target
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
        """Improved rollout using coefficient-based allocation."""
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

def generate_allocation_table(games_local, department_local):
    allocation_data = []
    for i, dept_name in enumerate(department_local):
        row = [dept_name] + [game.departments[i].allocation for game in games_local]
        allocation_data.append(row)

    headers = ['Department', 'Optimization', 'Best Response', 'Replicator Dynamics',
               'Fictitious Play', 'MCTS', 'Allocation Algorithm']
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
               'Fictitious Play', 'MCTS', 'Allocation Algorithm']
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

    diff_headers = ['Department', 'Opt - BR', 'Opt - RD', 'Opt - FP', 'Opt - MCTS', 'Opt - AllocAlg']
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

    diff_headers = ['Total Utility Difference', 'Opt - BR', 'Opt - RD', 'Opt - FP', 'Opt - MCTS', 'Opt - AllocAlg']
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
    plt.xticks(rotation=90)  # Rotate department labels for readability
    plt.tight_layout()
    plt.show()
def parallelization_solver(args):
    solver, game = args
    solver.solve(game)
    return game
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
if __name__ == "__main__":
    # Initialize base departments and names
    # base_departments, department_names = create_departments(num_departments=9, lowerbound=0, upperbound=5000)
    # capacity = 2389
    # # Deep copies for each solver instance
    # departments_opt = copy.deepcopy(base_departments)
    # departments_br = copy.deepcopy(base_departments)
    # departments_rd = copy.deepcopy(base_departments)
    # departments_fp = copy.deepcopy(base_departments)
    # departments_mcts = copy.deepcopy(base_departments)
    # departments_alloc = copy.deepcopy(base_departments)
    #
    # game_opt = MachineTimeGame(departments=departments_opt, alpha=10, capacity=capacity)
    # game_br = MachineTimeGame(departments=departments_br, alpha=10, capacity=capacity)
    # game_rd = MachineTimeGame(departments=departments_rd, alpha=10, capacity=capacity)
    # game_fp = MachineTimeGame(departments=departments_fp, alpha=10, capacity=capacity)
    # game_mcts = MachineTimeGame(departments=departments_mcts, alpha=10, capacity=capacity)
    # game_alloc = MachineTimeGame(departments=departments_alloc, alpha=10, capacity=capacity)
    #
    # solver_opt = OptimizationSolver()
    # solver_br = BestResponseSolver(max_iterations=5000, tolerance=1e-5)
    # solver_rd = ReplicatorDynamicsSolver(time_steps=10000, delta_t=1e-8, tolerance=1e-5)
    # solver_fp = FictitiousPlaySolver(max_iterations=5000, tolerance=1e-5)
    # solver_mcts = MCTSSolver(iterations=10, exploration_constant=math.sqrt(2), num_buckets=10, total_hours=game_mcts.capacity)
    # solver_alloc = AllocationAlgorithmSolver(delta=0.00001, activation_prob=0.01, max_iterations=1000000)
    #
    # solvers_and_games = [
    #     (solver_opt, game_opt),
    #     (solver_br, game_br),
    #     (solver_rd, game_rd),
    #     (solver_fp, game_fp),
    #     (solver_mcts, game_mcts),
    #     (solver_alloc, game_alloc)
    # ]
    #
    # with Pool(processes=len(solvers_and_games)) as pool:
    #     games = pool.map(parallelization_solver, solvers_and_games)
    #
    # # # Plot allocations and utilities
    # # plot_allocations(games, department_names)
    # # plot_utilities(games, department_names)
    #
    # # Generate tables using the department_names list from create_departments
    # generate_allocation_table(games, department_names)
    # generate_utility_table(games, department_names)
    # generate_comparison_table(games, department_names)
    # generate_utility_difference_table(games)
    # New main code that runs simulations over different prime capacities
    # Capacities as prime numbers
    # Generate a larger list of prime capacities
    lower_limit = 100
    upper_limit = 1000000
    capacities = generate_prime_capacities(lower_limit, upper_limit)
    print(sympy.primepi(upper_limit) - sympy.primepi(lower_limit - 1))
    results = []

    solver_names = ['Optimization', 'Best Response', 'Replicator Dynamics',
                    'Fictitious Play', 'MCTS', 'Allocation Algorithm']

    random.seed(42)

    for capacity in capacities:
        print(capacity)
        base_departments, department_names = create_departments(num_departments=20, lowerbound=0, upperbound=5000)

        departments_opt = copy.deepcopy(base_departments)
        departments_br = copy.deepcopy(base_departments)
        departments_rd = copy.deepcopy(base_departments)
        departments_fp = copy.deepcopy(base_departments)
        departments_mcts = copy.deepcopy(base_departments)
        departments_alloc = copy.deepcopy(base_departments)

        game_opt = MachineTimeGame(departments=departments_opt, alpha=10, capacity=capacity)
        game_br = MachineTimeGame(departments=departments_br, alpha=10, capacity=capacity)
        game_rd = MachineTimeGame(departments=departments_rd, alpha=10, capacity=capacity)
        game_fp = MachineTimeGame(departments=departments_fp, alpha=10, capacity=capacity)
        game_mcts = MachineTimeGame(departments=departments_mcts, alpha=10, capacity=capacity)
        game_alloc = MachineTimeGame(departments=departments_alloc, alpha=10, capacity=capacity)

        solver_opt = OptimizationSolver()
        solver_br = BestResponseSolver(max_iterations=5000, tolerance=1e-5)
        solver_rd = ReplicatorDynamicsSolver(time_steps=10000, delta_t=1e-8, tolerance=1e-5)
        solver_fp = FictitiousPlaySolver(max_iterations=5000, tolerance=1e-5)
        solver_mcts = MCTSSolver(iterations=1000, exploration_constant=math.sqrt(2), num_buckets=10,
                                 total_hours=game_mcts.capacity, spread=100)
        solver_alloc = AllocationAlgorithmSolver(delta=1, activation_prob=0.8, max_iterations=1000000)
        # solver_alloc = AllocationAlgorithmSolver(delta=1, activation_prob=0.8, max_iterations=1000000) this works much much better
        solvers_and_games = [
            (solver_opt, game_opt),
            (solver_br, game_br),
            (solver_rd, game_rd),
            (solver_fp, game_fp),
            (solver_mcts, game_mcts),
            (solver_alloc, game_alloc)
        ]

        with Pool(processes=len(solvers_and_games)) as pool:
            games = pool.map(parallelization_solver, solvers_and_games)

        total_utilities = {'Capacity': capacity}
        for solver_name, game in zip(solver_names, games):
            total_utility = sum(game.compute_utilities())
            total_utilities[solver_name] = total_utility

        results.append(total_utilities)

    df_results = pd.DataFrame(results)

    df_results.to_csv('Mighty_test_total_utilities_by_capacity.csv', index=False)

    for index, result in df_results.iterrows():
        capacity = result['Capacity']
        print(f"Capacity: {capacity}")
        solver_utilities = result.drop(labels=['Capacity']).to_dict()
        sorted_solvers = sorted(solver_utilities.items(), key=lambda item: item[1], reverse=True)
        for solver_name, utility in sorted_solvers:
            print(f"  {solver_name}: Total Utility = {utility:.2f}")
        print()

