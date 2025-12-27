"""
EV Charging Optimization using Stochastic Dynamic Programming
AMS 553 Final Project

This module implements the core SDP solver for optimal EV charging under price uncertainty.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import pickle
from dataclasses import dataclass


@dataclass
class EVParameters:
    """Parameters for the EV and charging system"""
    battery_capacity: float = 75.0  # kWh
    min_soc: float = 0.20  # Minimum SoC (20%)
    max_soc: float = 1.0   # Maximum SoC (100%)
    charging_efficiency: float = 0.92  # Charging efficiency
    max_charging_rate: float = 50.0  # kW
    degradation_cost_per_kwh: float = 0.05  # $/kWh charged
    penalty_cost_per_kwh: float = 10.0  # $/kWh below minimum SoC
    
    # Discretization parameters
    soc_states: int = 21  # Discretize SoC into 21 states (0%, 5%, ..., 100%)
    price_states: int = 10  # Discretize price into 10 states
    hour_states: int = 24  # 24 hours in a day
    action_states: int = 11  # Charging actions (0, 5, 10, ..., 50 kW)


class StochasticPriceModel:
    """Model for stochastic electricity prices"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Time-of-use pricing structure (cents/kWh)
        self.peak_hours = [17, 18, 19, 20, 21]  # 5 PM - 9 PM
        self.off_peak_hours = [0, 1, 2, 3, 4, 5, 6, 23]  # 11 PM - 7 AM
        
        # Base prices with some randomness
        self.base_off_peak = 0.08  # $0.08/kWh
        self.base_mid_peak = 0.15  # $0.15/kWh
        self.base_peak = 0.25      # $0.25/kWh
        
        # Volatility parameters
        self.price_volatility = 0.20
        
    def get_price_distribution(self, hour: int) -> Tuple[float, float]:
        """Get mean and std of price for a given hour"""
        if hour in self.off_peak_hours:
            mean_price = self.base_off_peak
        elif hour in self.peak_hours:
            mean_price = self.base_peak
        else:
            mean_price = self.base_mid_peak
            
        std_price = mean_price * self.price_volatility
        return mean_price, std_price
    
    def sample_price(self, hour: int) -> float:
        """Sample a price for given hour"""
        mean, std = self.get_price_distribution(hour)
        price = np.random.lognormal(np.log(mean), std / mean)
        return np.clip(price, 0.05, 0.50)  # Reasonable bounds
    
    def discretize_prices(self, n_states: int = 10) -> np.ndarray:
        """Create discrete price levels"""
        return np.linspace(0.05, 0.50, n_states)


class TripDemandModel:
    """Model for stochastic trip demands"""
    
    def __init__(self, ev_data: pd.DataFrame, seed: int = 42):
        np.random.seed(seed)
        self.ev_data = ev_data
        
        # Analyze trip patterns from data
        self._analyze_trip_patterns()
        
    def _analyze_trip_patterns(self):
        """Extract trip demand statistics from data"""
        # Calculate energy consumption per trip
        self.energy_per_trip = self.ev_data['Energy_Consumption_Rate_kWh/km'] * \
                               self.ev_data['Distance_to_Destination_km']
        
        # Trip probability by hour
        self.trip_prob_by_hour = self.ev_data.groupby('Session_Start_Hour').size() / len(self.ev_data)
        
        # Typical trip patterns
        self.morning_commute_hours = [7, 8, 9]
        self.evening_commute_hours = [16, 17, 18, 19]
        
    def get_trip_demand(self, hour: int, battery_capacity: float) -> float:
        """Sample trip demand for given hour (as fraction of battery capacity)"""
        # Higher probability of trips during commute hours
        if hour in self.morning_commute_hours or hour in self.evening_commute_hours:
            trip_prob = 0.7
            mean_consumption = 0.15  # 15% of battery
            std_consumption = 0.08
        elif 10 <= hour <= 22:  # Daytime
            trip_prob = 0.3
            mean_consumption = 0.10
            std_consumption = 0.05
        else:  # Night
            trip_prob = 0.05
            mean_consumption = 0.05
            std_consumption = 0.03
            
        # Decide if trip occurs
        if np.random.random() < trip_prob:
            consumption_fraction = np.abs(np.random.normal(mean_consumption, std_consumption))
            return np.clip(consumption_fraction, 0.0, 0.4)  # Max 40% per trip
        else:
            return 0.0


class EVChargingMDP:
    """Markov Decision Process for EV Charging"""
    
    def __init__(self, params: EVParameters, price_model: StochasticPriceModel, 
                 trip_model: TripDemandModel):
        self.params = params
        self.price_model = price_model
        self.trip_model = trip_model
        
        # Discretization
        self.soc_levels = np.linspace(0, 1, params.soc_states)
        self.price_levels = price_model.discretize_prices(params.price_states)
        self.hours = np.arange(24)
        self.actions = np.linspace(0, params.max_charging_rate, params.action_states)
        
        # Value function and policy
        self.V = np.zeros((params.soc_states, params.price_states, params.hour_states))
        self.policy = np.zeros((params.soc_states, params.price_states, params.hour_states), dtype=int)
        
    def immediate_cost(self, soc: float, price: float, action: float) -> float:
        """Calculate immediate cost for taking action"""
        # Energy charged (accounting for efficiency)
        energy_charged = (action * 1.0) / self.params.charging_efficiency  # 1 hour charging
        
        # Electricity cost
        electricity_cost = price * energy_charged
        
        # Degradation cost
        degradation_cost = self.params.degradation_cost_per_kwh * energy_charged
        
        # Penalty for low SoC
        if soc < self.params.min_soc:
            penalty = self.params.penalty_cost_per_kwh * (self.params.min_soc - soc) * self.params.battery_capacity
        else:
            penalty = 0.0
            
        return electricity_cost + degradation_cost + penalty
    
    def transition(self, soc: float, action: float, trip_demand: float) -> float:
        """Calculate next SoC after charging and trip"""
        # Charge added (fraction of capacity)
        charge_added = (action * 1.0 * self.params.charging_efficiency) / self.params.battery_capacity
        
        # New SoC after charging
        new_soc = soc + charge_added
        
        # Apply trip demand
        new_soc -= trip_demand
        
        # Clip to valid range
        return np.clip(new_soc, 0.0, self.params.max_soc)
    
    def get_state_index(self, soc: float, price: float, hour: int) -> Tuple[int, int, int]:
        """Convert continuous values to discrete state indices"""
        soc_idx = np.argmin(np.abs(self.soc_levels - soc))
        price_idx = np.argmin(np.abs(self.price_levels - price))
        hour_idx = hour % 24
        return soc_idx, price_idx, hour_idx
    
    def value_iteration(self, n_iterations: int = 100, gamma: float = 0.95, 
                       n_samples: int = 50, verbose: bool = True):
        """
        Perform value iteration to find optimal policy
        
        Args:
            n_iterations: Number of iterations
            gamma: Discount factor
            n_samples: Number of samples for expectation approximation
            verbose: Print progress
        """
        for iteration in range(n_iterations):
            V_old = self.V.copy()
            max_delta = 0
            
            # Iterate over all states
            for s_idx, soc in enumerate(self.soc_levels):
                for p_idx, price in enumerate(self.price_levels):
                    for h_idx, hour in enumerate(self.hours):
                        
                        best_value = float('inf')
                        best_action_idx = 0
                        
                        # Try all actions
                        for a_idx, action in enumerate(self.actions):
                            
                            # Skip invalid actions (can't charge above max)
                            charge_added = (action * 1.0 * self.params.charging_efficiency) / \
                                          self.params.battery_capacity
                            if soc + charge_added > self.params.max_soc + 0.01:
                                continue
                            
                            # Monte Carlo sampling for expectation
                            expected_value = 0.0
                            
                            for _ in range(n_samples):
                                # Sample next hour's price and trip demand
                                next_hour = (hour + 1) % 24
                                next_price = self.price_model.sample_price(next_hour)
                                trip_demand = self.trip_model.get_trip_demand(
                                    next_hour, self.params.battery_capacity
                                )
                                
                                # Calculate immediate cost
                                cost = self.immediate_cost(soc, price, action)
                                
                                # Calculate next state
                                next_soc = self.transition(soc, action, trip_demand)
                                
                                # Get next state indices
                                next_s_idx, next_p_idx, next_h_idx = self.get_state_index(
                                    next_soc, next_price, next_hour
                                )
                                
                                # Bellman equation
                                value = cost + gamma * V_old[next_s_idx, next_p_idx, next_h_idx]
                                expected_value += value
                            
                            expected_value /= n_samples
                            
                            if expected_value < best_value:
                                best_value = expected_value
                                best_action_idx = a_idx
                        
                        # Update value function and policy
                        self.V[s_idx, p_idx, h_idx] = best_value
                        self.policy[s_idx, p_idx, h_idx] = best_action_idx
                        
                        max_delta = max(max_delta, abs(best_value - V_old[s_idx, p_idx, h_idx]))
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{n_iterations}, Max Delta: {max_delta:.6f}")
            
            # Check convergence
            if max_delta < 1e-4:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
    
    def get_action(self, soc: float, price: float, hour: int) -> float:
        """Get optimal action for given state"""
        s_idx, p_idx, h_idx = self.get_state_index(soc, price, hour)
        action_idx = self.policy[s_idx, p_idx, h_idx]
        return self.actions[action_idx]
    
    def save_policy(self, filename: str):
        """Save learned policy to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'V': self.V,
                'policy': self.policy,
                'params': self.params,
                'soc_levels': self.soc_levels,
                'price_levels': self.price_levels,
                'actions': self.actions
            }, f)
        print(f"Policy saved to {filename}")
    
    def load_policy(self, filename: str):
        """Load policy from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.V = data['V']
            self.policy = data['policy']
        print(f"Policy loaded from {filename}")


if __name__ == "__main__":
    print("EV Charging SDP Module")
    print("=" * 50)
    print("This module contains the core SDP implementation.")
    print("Run main.py to execute the full optimization pipeline.")
