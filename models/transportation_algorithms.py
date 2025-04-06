import numpy as np
import pandas as pd
from geopy.distance import geodesic

class TransportationProblem:
    """Base class for transportation problem algorithms"""
    
    def __init__(self, student_locations, bus_capacities, college_location):
        """
        Initialize the transportation problem
        
        Args:
            student_locations: List of (lat, lng) tuples for student locations
            bus_capacities: List of integers representing capacity of each bus
            college_location: (lat, lng) tuple for college location
        """
        self.student_locations = student_locations
        self.bus_capacities = bus_capacities
        self.college_location = college_location
        self.num_students = len(student_locations)
        self.num_buses = len(bus_capacities)
        
        # Calculate cost matrix (distance from each student to each bus stop)
        self.cost_matrix = self._calculate_cost_matrix()
        
    def _calculate_cost_matrix(self):
        """Calculate the cost (distance) matrix between students and potential bus stops"""
        # For initial implementation, we'll use a simple approach to determine potential bus stops
        # If fewer students than buses, each student gets their own bus
        if self.num_students <= self.num_buses:
            bus_stops = self.student_locations
        else:
            # Simple clustering: divide students into groups based on proximity to college
            # Sort students by distance to college
            distances_to_college = []
            for loc in self.student_locations:
                distances_to_college.append(geodesic(loc, self.college_location).kilometers)
            
            # Sort student indices by distance to college
            sorted_indices = np.argsort(distances_to_college)
            
            # Divide students into groups (one for each bus)
            students_per_bus = self.num_students // self.num_buses
            remainder = self.num_students % self.num_buses
            
            # Create bus stops as the average location of each group
            bus_stops = []
            start_idx = 0
            
            for i in range(self.num_buses):
                # Add one extra student to the first 'remainder' buses
                group_size = students_per_bus + (1 if i < remainder else 0)
                end_idx = start_idx + group_size
                
                if end_idx > start_idx:
                    # Get the indices of students in this group
                    group_indices = sorted_indices[start_idx:end_idx]
                    
                    # Calculate the average location (centroid) for this group
                    group_locs = [self.student_locations[idx] for idx in group_indices]
                    avg_lat = sum(loc[0] for loc in group_locs) / len(group_locs)
                    avg_lng = sum(loc[1] for loc in group_locs) / len(group_locs)
                    
                    bus_stops.append((avg_lat, avg_lng))
                else:
                    # If no students in this group, use college location
                    bus_stops.append(self.college_location)
                
                start_idx = end_idx
        
        # Calculate distance from each student to each potential bus stop
        cost_matrix = np.zeros((self.num_students, self.num_buses))
        for i, student_loc in enumerate(self.student_locations):
            for j, bus_stop in enumerate(bus_stops):
                cost_matrix[i, j] = geodesic(student_loc, bus_stop).kilometers
                
        return cost_matrix
    
    def solve(self):
        """Solve the transportation problem (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement this method")


class VogelsApproximationMethod(TransportationProblem):
    """Implementation of Vogel's Approximation Method for transportation problems"""
    
    def solve(self):
        """Solve using Vogel's Approximation Method"""
        # Create a copy of the cost matrix to work with
        cost_matrix = self.cost_matrix.copy()
        
        # Create supply (students) and demand (bus capacities) arrays
        supply = np.ones(self.num_students)  # Each student counts as 1
        demand = np.array(self.bus_capacities)
        
        # Initialize allocation matrix
        allocation = np.zeros((self.num_students, self.num_buses))
        
        # Continue until all allocations are made
        while np.sum(supply) > 0 and np.sum(demand) > 0:
            # Calculate row and column penalties
            row_penalties = self._calculate_penalties(cost_matrix, axis=1)
            col_penalties = self._calculate_penalties(cost_matrix, axis=0)
            
            # Find maximum penalty
            max_row_penalty = np.max(row_penalties) if len(row_penalties) > 0 else -1
            max_col_penalty = np.max(col_penalties) if len(col_penalties) > 0 else -1
            
            if max_row_penalty >= max_col_penalty:
                # Select row with maximum penalty
                row = np.argmax(row_penalties)
                # Find minimum cost cell in that row
                valid_cols = np.where(demand > 0)[0]
                if len(valid_cols) == 0:
                    break
                col = valid_cols[np.argmin(cost_matrix[row, valid_cols])]
            else:
                # Select column with maximum penalty
                col = np.argmax(col_penalties)
                # Find minimum cost cell in that column
                valid_rows = np.where(supply > 0)[0]
                if len(valid_rows) == 0:
                    break
                row = valid_rows[np.argmin(cost_matrix[valid_rows, col])]
            
            # Allocate as much as possible
            allocated = min(supply[row], demand[col])
            allocation[row, col] = allocated
            
            # Update supply and demand
            supply[row] -= allocated
            demand[col] -= allocated
            
            # Mark the cell as used by setting cost to infinity
            if supply[row] == 0:
                cost_matrix[row, :] = np.inf
            if demand[col] == 0:
                cost_matrix[:, col] = np.inf
        
        # Convert allocation matrix to bus assignments
        bus_assignments = {}
        for j in range(self.num_buses):
            bus_assignments[j] = [i for i in range(self.num_students) if allocation[i, j] > 0]
            
        return bus_assignments
    
    def _calculate_penalties(self, cost_matrix, axis):
        """
        Calculate penalties for each row or column
        Penalty is the difference between the two lowest costs
        """
        penalties = []
        
        if axis == 1:  # Row penalties
            for i in range(cost_matrix.shape[0]):
                row = cost_matrix[i, :]
                finite_values = row[np.isfinite(row)]
                if len(finite_values) >= 2:
                    sorted_values = np.sort(finite_values)
                    penalties.append(sorted_values[1] - sorted_values[0])
                elif len(finite_values) == 1:
                    penalties.append(finite_values[0])
                else:
                    penalties.append(0)
        else:  # Column penalties
            for j in range(cost_matrix.shape[1]):
                col = cost_matrix[:, j]
                finite_values = col[np.isfinite(col)]
                if len(finite_values) >= 2:
                    sorted_values = np.sort(finite_values)
                    penalties.append(sorted_values[1] - sorted_values[0])
                elif len(finite_values) == 1:
                    penalties.append(finite_values[0])
                else:
                    penalties.append(0)
                    
        return np.array(penalties)


class LeastCostMethod(TransportationProblem):
    """Implementation of Least Cost Method for transportation problems"""
    
    def solve(self):
        """Solve using Least Cost Method"""
        # Create a copy of the cost matrix to work with
        cost_matrix = self.cost_matrix.copy()
        
        # Create supply (students) and demand (bus capacities) arrays
        supply = np.ones(self.num_students)  # Each student counts as 1
        demand = np.array(self.bus_capacities)
        
        # Initialize allocation matrix
        allocation = np.zeros((self.num_students, self.num_buses))
        
        # Continue until all allocations are made
        while np.sum(supply) > 0 and np.sum(demand) > 0:
            # Find the cell with minimum cost
            valid_mask = (cost_matrix != np.inf)
            if not np.any(valid_mask):
                break
                
            min_cost_idx = np.argmin(cost_matrix[valid_mask])
            min_cost_flat_idx = np.arange(cost_matrix.size)[valid_mask.flatten()][min_cost_idx]
            row, col = np.unravel_index(min_cost_flat_idx, cost_matrix.shape)
            
            # Allocate as much as possible
            allocated = min(supply[row], demand[col])
            allocation[row, col] = allocated
            
            # Update supply and demand
            supply[row] -= allocated
            demand[col] -= allocated
            
            # Mark the cell as used by setting cost to infinity
            if supply[row] == 0:
                cost_matrix[row, :] = np.inf
            if demand[col] == 0:
                cost_matrix[:, col] = np.inf
        
        # Convert allocation matrix to bus assignments
        bus_assignments = {}
        for j in range(self.num_buses):
            bus_assignments[j] = [i for i in range(self.num_students) if allocation[i, j] > 0]
            
        return bus_assignments


class MODI(TransportationProblem):
    """Implementation of Modified Distribution Method (MODI) for transportation problems"""
    
    def solve(self):
        """Solve using MODI method"""
        # First, get an initial feasible solution using Least Cost Method
        lcm = LeastCostMethod(self.student_locations, self.bus_capacities, self.college_location)
        initial_assignments = lcm.solve()
        
        # Convert initial assignments to allocation matrix
        allocation = np.zeros((self.num_students, self.num_buses))
        for bus_id, student_indices in initial_assignments.items():
            for student_idx in student_indices:
                allocation[student_idx, bus_id] = 1
        
        # Apply MODI optimization to improve the initial solution
        improved = True
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = self._improve_solution(allocation)
            iteration += 1
        
        # Convert allocation matrix to bus assignments
        bus_assignments = {}
        for j in range(self.num_buses):
            bus_assignments[j] = [i for i in range(self.num_students) if allocation[i, j] > 0]
            
        return bus_assignments
    
    def _improve_solution(self, allocation):
        """Try to improve the current solution using MODI method"""
        # Calculate u and v values
        u = np.zeros(self.num_students)
        v = np.zeros(self.num_buses)
        
        # Set u[0] = 0 as a starting point
        u[0] = 0
        
        # Find basic variables (allocated cells)
        basic_vars = []
        for i in range(self.num_students):
            for j in range(self.num_buses):
                if allocation[i, j] > 0:
                    basic_vars.append((i, j))
        
        # Calculate u and v values
        for i, j in basic_vars:
            if i == 0:
                v[j] = self.cost_matrix[i, j]
            else:
                u[i] = self.cost_matrix[i, j] - v[j]
        
        # Calculate opportunity costs for non-basic variables
        opportunity_costs = np.zeros_like(self.cost_matrix)
        for i in range(self.num_students):
            for j in range(self.num_buses):
                if allocation[i, j] == 0:
                    opportunity_costs[i, j] = self.cost_matrix[i, j] - u[i] - v[j]
        
        # Find the cell with the most negative opportunity cost
        min_cost = np.min(opportunity_costs)
        if min_cost >= 0:
            return False  # No improvement possible
            
        entering_var = np.unravel_index(np.argmin(opportunity_costs), opportunity_costs.shape)
        
        # Find a closed loop starting from the entering variable
        loop = self._find_loop(entering_var, basic_vars)
        
        # Find the minimum allocation in the negative positions of the loop
        min_allocation = float('inf')
        for idx, (i, j) in enumerate(loop):
            if idx % 2 == 1:  # Negative position
                if allocation[i, j] < min_allocation:
                    min_allocation = allocation[i, j]
        
        # Update the allocations along the loop
        for idx, (i, j) in enumerate(loop):
            if idx % 2 == 0:  # Positive position
                allocation[i, j] += min_allocation
            else:  # Negative position
                allocation[i, j] -= min_allocation
        
        return True
    
    def _find_loop(self, entering_var, basic_vars):
        """Find a closed loop of basic variables including the entering variable"""
        # This is a simplified implementation
        # In a real-world scenario, a more sophisticated algorithm would be needed
        
        # Start with the entering variable
        loop = [entering_var]
        
        # Add a basic variable in the same row
        for j in range(self.num_buses):
            if (entering_var[0], j) in basic_vars:
                loop.append((entering_var[0], j))
                break
        
        # Add a basic variable in the same column as the last added variable
        for i in range(self.num_students):
            if (i, loop[-1][1]) in basic_vars and i != loop[-1][0]:
                loop.append((i, loop[-1][1]))
                break
        
        # Complete the loop by adding the variable in the same row as the last added
        # and same column as the entering variable
        loop.append((loop[-1][0], entering_var[1]))
        
        return loop


def calculate_route_distances(bus_assignments, student_locations, college_location):
    """
    Calculate the total distance for each bus route
    
    Args:
        bus_assignments: Dictionary mapping bus IDs to lists of student indices
        student_locations: List of (lat, lng) tuples for student locations
        college_location: (lat, lng) tuple for college location
        
    Returns:
        Dictionary mapping bus IDs to total route distances
    """
    route_distances = {}
    
    for bus_id, student_indices in bus_assignments.items():
        if not student_indices:
            route_distances[bus_id] = 0
            continue
            
        # Start from college
        current_location = college_location
        total_distance = 0
        
        # Find optimal order to visit students (simple greedy approach)
        remaining_indices = student_indices.copy()
        visit_order = []
        
        while remaining_indices:
            # Find closest student
            min_distance = float('inf')
            closest_idx = None
            
            for idx in remaining_indices:
                distance = geodesic(current_location, student_locations[idx]).kilometers
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = idx
            
            # Add to route
            visit_order.append(closest_idx)
            total_distance += min_distance
            current_location = student_locations[closest_idx]
            remaining_indices.remove(closest_idx)
        
        # Return to college
        total_distance += geodesic(current_location, college_location).kilometers
        
        route_distances[bus_id] = total_distance
    
    return route_distances
