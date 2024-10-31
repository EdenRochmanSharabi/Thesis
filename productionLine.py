import numpy as np
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import math
from tabulate import tabulate

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

    def utility(self, x_i, total_allocation, alpha):
        """
        Computes the utility of the department given its allocation.
        """
        if total_allocation > 100:
            penalty_share = x_i / total_allocation if total_allocation > 0 else 0
            penalty_term = alpha * (total_allocation - 100) * penalty_share
        else:
            penalty_term = 0
        return self.coefficient * np.log(x_i + 1) - penalty_term
class MachineTimeGame:
    """
    Represents the machine time allocation game among departments.
    """
    def __init__(self, departments, alpha=10):
        self.departments = departments  # List of Department instances
        self.alpha = alpha  # Penalty parameter

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
                alpha=self.alpha
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
            # Default initial guess: equal allocation
            x0 = np.full(num_departments, 100 / num_departments)
        else:
            x0 = self.initial_guess

        # Bounds: allocations must be non-negative
        bounds = [(0, None) for _ in range(num_departments)]

        # Constraints: total allocation must equal 100 hours
        constraints = {
            'type': 'eq',
            'fun': lambda x: 100 - sum(x)
        }

        # Objective function: negative total utility (since we will minimize)
        def objective(allocations):
            total_alloc = sum(allocations)
            total_utility = 0
            for k, dept in enumerate(game.departments):
                util = dept.utility(
                    allocations[k],
                    total_alloc,
                    alpha=game.alpha
                )
                total_utility += util
            return -total_utility  # Negative for minimization

        # Minimize the negative total utility
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )

        if result.success:
            # Update game with the optimal allocations
            game.set_allocations(result.x)
            return result.x
        else:
            raise ValueError("Optimization failed.")
class BestResponseSolver(Solver):
    """
    Solver that uses best-response dynamics to find the Nash Equilibrium.
    """
    def __init__(self, max_iterations=1000, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(self, game: MachineTimeGame):
        num_departments = len(game.departments)
        allocations = np.full(num_departments, 100 / num_departments)

        for iteration in range(self.max_iterations):
            old_allocations = allocations.copy()

            for i, dept in enumerate(game.departments):
                others_sum = sum(old_allocations) - old_allocations[i]

                def best_response_function(x):
                    x_i = x[0]
                    total_alloc = others_sum + x_i
                    if total_alloc > 100:
                        # Penalize allocations that exceed the total limit
                        return 1e6
                    else:
                        return -dept.utility(x_i, total_alloc, game.alpha)

                # Bounds for x_i
                # bounds = [(0, None)] # This gives insane numbers
                bounds = [(0, 100 - others_sum)] # Playing with x_i so it cannot exceed the remaining available time

                # Constraints to ensure total allocation does not exceed 100 hours
                constraints = {'type': 'ineq', 'fun': lambda x: 100 - (others_sum + x[0])}

                # Initial guess for the allocation
                x0 = np.array([allocations[i]])

                # Optimize
                res = minimize(
                    best_response_function,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'ftol': 1e-9, 'disp': False}
                )

                # Update allocation for department i
                allocations[i] = res.x[0]

            # Check convergence
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
        allocations = np.full(num_departments, 100 / num_departments)

        for t in range(self.time_steps):
            old_allocations = allocations.copy()
            total_alloc = sum(allocations)
            utilities = []

            # Compute utilities for current allocations
            for i, dept in enumerate(game.departments):
                util = dept.utility(allocations[i], total_alloc, game.alpha)
                utilities.append(util)

            # Compute average utility
            average_utility = sum(utilities) / num_departments if num_departments > 0 else 1

            # Update allocations using replicator dynamics
            for j in range(num_departments):
                allocations[j] += self.delta_t * allocations[j] * (utilities[j] - average_utility)

            # Ensure allocations are non-negative
            allocations = np.maximum(allocations, 0)

            # Normalize allocations to sum to 100
            total_alloc = sum(allocations)
            if total_alloc > 0:
                allocations = (allocations / total_alloc) * 100
            else:
                allocations = np.full(num_departments, 100 / num_departments)

            # Check for convergence
            if np.linalg.norm(allocations - old_allocations) < self.tolerance:
                break

        # Update game with the final allocations
        game.set_allocations(allocations)
        return allocations
class FictitiousPlaySolver(Solver):
    """
    Solver that uses fictitious play to find an equilibrium.
    """
    def __init__(self, max_iterations=1000, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(self, game: MachineTimeGame):
        num_departments = len(game.departments)
        # Initialize allocations and historical averages
        allocations = np.full(num_departments, 100 / num_departments)
        historical_allocations = np.zeros((self.max_iterations + 1, num_departments))
        historical_allocations[0] = allocations.copy()

        for iteration in range(1, self.max_iterations + 1):
            old_allocations = allocations.copy()
            # Update historical averages
            avg_allocations = np.mean(historical_allocations[:iteration], axis=0)

            for i, dept in enumerate(game.departments):
                others_avg_alloc = sum(avg_allocations) - avg_allocations[i]
                others_avg_alloc = min(others_avg_alloc, 100)  # Ensure total does not exceed 100

                def best_response_function(x):
                    x_i = x[0]
                    total_alloc = others_avg_alloc + x_i
                    if total_alloc > 100:
                        return 1e6  # Penalize allocations that exceed the limit
                    return -dept.utility(x_i, total_alloc, game.alpha)

                # Bounds for x_i
                bounds = [(0, None)]
                # Constraints to ensure total allocation does not exceed 100 hours
                constraints = {'type': 'ineq', 'fun': lambda x: 100 - (others_avg_alloc + x[0])}
                # Initial guess for the allocation
                x0 = np.array([allocations[i]])

                # Optimize
                res = minimize(
                    best_response_function,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'ftol': 1e-9, 'disp': False}
                )

                # Update allocation for department i
                allocations[i] = res.x[0]

            # Record allocations
            historical_allocations[iteration] = allocations.copy()

            # Check convergence
            if np.linalg.norm(allocations - old_allocations) < self.tolerance:
                break

        # Normalize final allocations to sum to 100
        total_alloc = sum(allocations)
        if total_alloc > 0:
            allocations = (allocations / total_alloc) * 100
        else:
            allocations = np.full(num_departments, 100 / num_departments)

        # Update game with the final allocations
        game.set_allocations(allocations)
        return allocations
class MCTSSolver(Solver):
    """
    Improved MCTS solver that uses smart rollout and better state evaluation.
    """

    def __init__(self, iterations=1000, exploration_constant=1.414, num_buckets=10):
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.num_buckets = num_buckets  # Number of discrete allocation levels

    def solve(self, game: MachineTimeGame):
        num_departments = len(game.departments)
        total_hours = 100

        # Initialize the root node with proportional allocation based on coefficients
        coefficients = [dept.coefficient for dept in game.departments]
        total_coeff = sum(coefficients)
        initial_state = [coeff / total_coeff * 100 for coeff in coefficients]

        root = MCTSNode(
            state=initial_state,
            parent=None,
            game=game,
            department_index=0,
            num_buckets=self.num_buckets
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
            reward = node.smart_rollout()

            # Update best solution if found
            if reward > best_utility:
                best_utility = reward
                best_allocation = node.get_normalized_state()

            # Backpropagation
            node.backpropagate(reward)

        # If no better solution found, use the most visited child's allocation
        if best_allocation is None:
            best_child = max(root.children, key=lambda n: n.visits)
            best_allocation = best_child.get_normalized_state()

        game.set_allocations(best_allocation)
        return best_allocation
class MCTSNode:
    def __init__(self, state, parent, game, department_index, num_buckets):
        self.state = state
        self.parent = parent
        self.game = game
        self.department_index = department_index
        self.num_buckets = num_buckets
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
        base_allocation = target_proportion * 100
        spread = 20  # Allow ±20% variation around the target
        min_alloc = max(0, base_allocation - spread)
        max_alloc = min(100, base_allocation + spread)

        # Create discrete steps within this range
        step_size = (max_alloc - min_alloc) / self.num_buckets
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
            num_buckets=self.num_buckets
        )
        self.children.append(child)
        return child

    def smart_rollout(self):
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
            remaining_hours = 100 - used_hours

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
        """Return the state with allocations normalized to sum to 100."""
        return self._normalize_allocation(self.state)

    @staticmethod
    def _normalize_allocation(allocation):
        """Helper method to normalize allocations to sum to 100."""
        total = sum(allocation)
        if total == 0:
            return [100 / len(allocation)] * len(allocation)
        return [x * (100 / total) for x in allocation]
class AllocationAlgorithmSolver(Solver):
    """
    Solver that implements the iterative allocation algorithm.
    """
    def __init__(self, delta=0.1, activation_prob=0.5, max_iterations=10000):
        self.delta = delta
        self.activation_prob = activation_prob
        self.max_iterations = max_iterations

    def solve(self, game: MachineTimeGame):
        num_departments = len(game.departments)
        allocations = np.zeros(num_departments)  # Initial allocations: W(0) = 0
        total_alloc = sum(allocations)  # Total allocation W
        t = 0  # Time step t

        while total_alloc < 100 and t < self.max_iterations:
            old_allocations = allocations.copy()
            total_alloc = sum(allocations)  # Update W(t)

            for i, dept in enumerate(game.departments):
                if np.random.rand() < self.activation_prob:  # Activation probability ν_x^act
                    # Possible new allocation by increasing delta
                    if total_alloc + self.delta <= 100:
                        new_allocations = allocations.copy()
                        new_allocations[i] += self.delta  # Increment allocation by δ for agent x
                        new_total_alloc = total_alloc + self.delta  # Update W' = W + δ

                        # Compute utility difference for department i
                        current_utility = dept.utility(
                            allocations[i],  # Current allocation W_x
                            total_alloc,  # Current total allocation W
                            game.alpha  # Penalty parameter α
                        )
                        new_utility = dept.utility(
                            new_allocations[i],  # Potential new allocation W'_x
                            new_total_alloc,  # New total allocation W'
                            game.alpha
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

        # Normalize allocations to sum to 100
        allocations = self._normalize_allocations(allocations)  # Normalize W to satisfy constraint
        game.set_allocations(allocations)
        return allocations

    @staticmethod
    def _normalize_allocations(allocations):
        total = sum(allocations)
        if total == 0:
            return np.full(len(allocations), 100 / len(allocations))
        return allocations * (100 / total)

def create_departments():
    """Creates and returns a fresh list of Department instances."""
    return [
        Department(name='A', coefficient=0.5),
        Department(name='B', coefficient=1),
        Department(name='C', coefficient=0.3),
        Department(name='D', coefficient=10),
        Department(name='E', coefficient=15),
        Department(name='F', coefficient=20),
        Department(name='G', coefficient=0.1),
        Department(name='H', coefficient=5),
        Department(name='I', coefficient=8),
        Department(name='J', coefficient=12),
        Department(name='K', coefficient=0.2),
        Department(name='L', coefficient=25),
        Department(name='M', coefficient=3),
        Department(name='N', coefficient=1.5),
        Department(name='O', coefficient=0.7),
        Department(name='P', coefficient=0.05),
    ]


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
if __name__ == "__main__":
    # Fresh game of each solver
    game_opt = MachineTimeGame(departments=create_departments(), alpha=10)  # Optimization
    game_br = MachineTimeGame(departments=create_departments(), alpha=10)  # Best Response
    game_rd = MachineTimeGame(departments=create_departments(), alpha=10)  # Replicator Dynamics
    game_fp = MachineTimeGame(departments=create_departments(), alpha=10)  # Fictitious Play
    game_mcts = MachineTimeGame(departments=create_departments(), alpha=10)  # MCTS
    game_alloc = MachineTimeGame(departments=create_departments(), alpha=10)  # Allocation Algorithm

    # Initialize the solvers
    solver_opt = OptimizationSolver()
    solver_br = BestResponseSolver(max_iterations=5000, tolerance=1e-5)
    solver_rd = ReplicatorDynamicsSolver(time_steps=10000, delta_t=1e-8, tolerance=1e-5)
    solver_fp = FictitiousPlaySolver(max_iterations=5000, tolerance=1e-5)
    solver_mcts = MCTSSolver(iterations=10, exploration_constant=math.sqrt(2), num_buckets=10)
    solver_alloc = AllocationAlgorithmSolver(delta=0.00001, activation_prob=0.01, max_iterations=1000000)

    department_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']


    # Solve each game with the corresponding solver
    solver_opt.solve(game_opt)
    solver_br.solve(game_br)
    solver_rd.solve(game_rd)
    solver_fp.solve(game_fp)
    solver_mcts.solve(game_mcts)
    solver_alloc.solve(game_alloc)

    # List of games for easy access
    games = [game_opt, game_br, game_rd, game_fp, game_mcts, game_alloc]

    # Generate tables
    generate_allocation_table(games, department_names)
    generate_utility_table(games, department_names)
    generate_comparison_table(games, department_names)
    generate_utility_difference_table(games)
