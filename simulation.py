"""
Monte Carlo Simulation Framework for EV Charging Policy Evaluation
AMS 553 Final Project

This module simulates EV charging over time to evaluate different policies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from baseline_policies import ChargingPolicy


@dataclass
class SimulationResult:
    """Container for simulation results"""
    policy_name: str
    total_cost: float
    electricity_cost: float
    degradation_cost: float
    penalty_cost: float
    avg_soc: float
    min_soc: float
    num_violations: int  # Times SoC went below minimum
    num_charges: int  # Number of charging events
    total_energy_charged: float
    soc_history: np.ndarray
    price_history: np.ndarray
    action_history: np.ndarray
    cost_history: np.ndarray
    hour_history: np.ndarray


class EVSimulator:
    """Simulator for EV charging behavior"""
    
    def __init__(self, battery_capacity: float = 75.0,
                 min_soc: float = 0.20,
                 max_soc: float = 1.0,
                 charging_efficiency: float = 0.92,
                 degradation_cost_per_kwh: float = 0.05,
                 penalty_cost_per_kwh: float = 10.0):
        
        self.battery_capacity = battery_capacity
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.charging_efficiency = charging_efficiency
        self.degradation_cost_per_kwh = degradation_cost_per_kwh
        self.penalty_cost_per_kwh = penalty_cost_per_kwh
        
    def simulate_policy(self, policy: ChargingPolicy, 
                       price_model, 
                       trip_model,
                       n_days: int = 365,
                       initial_soc: float = 0.80,
                       seed: int = 42) -> SimulationResult:
        """
        Simulate a charging policy over multiple days
        
        Args:
            policy: Charging policy to evaluate
            price_model: Model for generating electricity prices
            trip_model: Model for generating trip demands
            n_days: Number of days to simulate
            initial_soc: Initial battery state of charge
            seed: Random seed
            
        Returns:
            SimulationResult with complete history and metrics
        """
        np.random.seed(seed)
        
        # Initialize tracking arrays
        n_hours = n_days * 24
        soc_history = np.zeros(n_hours + 1)
        price_history = np.zeros(n_hours)
        action_history = np.zeros(n_hours)
        cost_history = np.zeros(n_hours)
        hour_history = np.zeros(n_hours, dtype=int)
        
        # Costs
        total_electricity_cost = 0.0
        total_degradation_cost = 0.0
        total_penalty_cost = 0.0
        
        # Metrics
        num_violations = 0
        num_charges = 0
        total_energy_charged = 0.0
        
        # Initial state
        soc = initial_soc
        soc_history[0] = soc
        
        # Simulation loop
        for t in range(n_hours):
            day = t // 24
            hour = t % 24
            hour_history[t] = hour
            
            # Get current price
            price = price_model.sample_price(hour)
            price_history[t] = price
            
            # Get action from policy
            action = policy.get_action(soc, price, hour)
            action_history[t] = action
            
            # Calculate costs
            # 1. Electricity cost
            energy_charged_from_grid = (action * 1.0) / self.charging_efficiency
            electricity_cost = price * energy_charged_from_grid
            
            # 2. Degradation cost
            degradation_cost = self.degradation_cost_per_kwh * energy_charged_from_grid
            
            # 3. Penalty for low SoC
            if soc < self.min_soc:
                penalty_cost = self.penalty_cost_per_kwh * \
                              (self.min_soc - soc) * self.battery_capacity
                num_violations += 1
            else:
                penalty_cost = 0.0
            
            # Total cost for this hour
            hour_cost = electricity_cost + degradation_cost + penalty_cost
            cost_history[t] = hour_cost
            
            # Update cumulative costs
            total_electricity_cost += electricity_cost
            total_degradation_cost += degradation_cost
            total_penalty_cost += penalty_cost
            
            # Update battery state
            # Add charge
            energy_to_battery = (action * 1.0 * self.charging_efficiency) / self.battery_capacity
            soc = min(soc + energy_to_battery, self.max_soc)
            
            # Track charging events
            if action > 0.5:  # Threshold to count as charging
                num_charges += 1
                total_energy_charged += energy_charged_from_grid
            
            # Generate trip demand
            trip_demand = trip_model.get_trip_demand(hour, self.battery_capacity)
            soc = max(soc - trip_demand, 0.0)
            
            # Store SoC
            soc_history[t + 1] = soc
        
        # Calculate aggregate metrics
        total_cost = total_electricity_cost + total_degradation_cost + total_penalty_cost
        avg_soc = np.mean(soc_history)
        min_soc = np.min(soc_history)
        
        # Create result object
        result = SimulationResult(
            policy_name=policy.get_name(),
            total_cost=total_cost,
            electricity_cost=total_electricity_cost,
            degradation_cost=total_degradation_cost,
            penalty_cost=total_penalty_cost,
            avg_soc=avg_soc,
            min_soc=min_soc,
            num_violations=num_violations,
            num_charges=num_charges,
            total_energy_charged=total_energy_charged,
            soc_history=soc_history,
            price_history=price_history,
            action_history=action_history,
            cost_history=cost_history,
            hour_history=hour_history
        )
        
        return result
    
    def compare_policies(self, policies: Dict[str, ChargingPolicy],
                        price_model,
                        trip_model,
                        n_days: int = 365,
                        n_trials: int = 10,
                        initial_soc: float = 0.80) -> pd.DataFrame:
        """
        Compare multiple policies across multiple trials
        
        Args:
            policies: Dictionary of policy name -> policy object
            price_model: Stochastic price model
            trip_model: Trip demand model
            n_days: Number of days per trial
            n_trials: Number of independent trials
            initial_soc: Initial SoC for each trial
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for trial in range(n_trials):
            print(f"\nTrial {trial + 1}/{n_trials}")
            seed = 42 + trial
            
            for policy_name, policy in policies.items():
                print(f"  Simulating {policy.get_name()}...")
                
                result = self.simulate_policy(
                    policy=policy,
                    price_model=price_model,
                    trip_model=trip_model,
                    n_days=n_days,
                    initial_soc=initial_soc,
                    seed=seed
                )
                
                # Store key metrics
                results.append({
                    'trial': trial,
                    'policy': policy_name,
                    'policy_name': result.policy_name,
                    'total_cost': result.total_cost,
                    'electricity_cost': result.electricity_cost,
                    'degradation_cost': result.degradation_cost,
                    'penalty_cost': result.penalty_cost,
                    'avg_soc': result.avg_soc,
                    'min_soc': result.min_soc,
                    'num_violations': result.num_violations,
                    'num_charges': result.num_charges,
                    'total_energy_charged': result.total_energy_charged
                })
        
        df = pd.DataFrame(results)
        return df
    
    def print_comparison_summary(self, comparison_df: pd.DataFrame):
        """Print summary statistics for policy comparison"""
        print("\n" + "=" * 80)
        print("POLICY COMPARISON SUMMARY")
        print("=" * 80)
        
        summary = comparison_df.groupby('policy_name').agg({
            'total_cost': ['mean', 'std'],
            'electricity_cost': ['mean', 'std'],
            'degradation_cost': ['mean', 'std'],
            'penalty_cost': ['mean', 'std'],
            'avg_soc': ['mean', 'std'],
            'min_soc': ['mean', 'min'],
            'num_violations': ['mean', 'max'],
            'num_charges': ['mean', 'std'],
            'total_energy_charged': ['mean', 'std']
        })
        
        print(f"\n{summary}")
        
        # Find best policy by total cost
        best_policy = comparison_df.groupby('policy_name')['total_cost'].mean().idxmin()
        best_cost = comparison_df.groupby('policy_name')['total_cost'].mean().min()
        
        print(f"\n{'*' * 80}")
        print(f"BEST POLICY: {best_policy}")
        print(f"Average Total Cost: ${best_cost:.2f}")
        print(f"{'*' * 80}\n")


if __name__ == "__main__":
    print("EV Charging Simulator Module")
    print("=" * 50)
    print("This module contains the Monte Carlo simulation framework.")
    print("Run main.py to execute the full simulation pipeline.")
