"""
Data Processing Module for EV Charging Optimization
AMS 553 Final Project

This module processes the EV charging dataset and extracts relevant features.
"""

import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt


class DataProcessor:
    """Process EV charging dataset for analysis"""
    
    def __init__(self, ev_data_path: str = "ev_charging_dataset.csv"):
        self.ev_data_path = ev_data_path
        self.ev_data = None
        
    def load_data(self):
        """Load the EV charging dataset"""
        print(f"Loading data from {self.ev_data_path}...")
        self.ev_data = pd.read_csv(self.ev_data_path)
        print(f"Loaded {len(self.ev_data)} records")
        return self.ev_data
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        if self.ev_data is None:
            self.load_data()
        
        print("\n" + "=" * 80)
        print("DATASET EXPLORATION")
        print("=" * 80)
        
        print(f"\nDataset shape: {self.ev_data.shape}")
        print(f"\nColumns: {list(self.ev_data.columns)}")
        
        print("\n--- Key Statistics ---")
        print(f"\nBattery Capacity (kWh):")
        print(self.ev_data['Battery_Capacity_kWh'].describe())
        
        print(f"\nState of Charge (%):")
        print(self.ev_data['State_of_Charge_%'].describe())
        
        print(f"\nCharging Rate (kW):")
        print(self.ev_data['Charging_Rate_kW'].describe())
        
        print(f"\nEnergy Drawn (kWh):")
        print(self.ev_data['Energy_Drawn_kWh'].describe())
        
        print(f"\nCharging Load (kW):")
        print(self.ev_data['Charging_Load_kW'].describe())
        
        return self.ev_data
    
    def extract_price_patterns(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract electricity price patterns from charging load data
        Using charging load as proxy for demand, which correlates with price
        """
        if self.ev_data is None:
            self.load_data()
        
        # Use charging load and hour to estimate prices
        # Normalize charging load to price range
        charging_load = self.ev_data['Charging_Load_kW'].values
        
        # Estimate price based on load (higher load = higher price)
        # Normalize to reasonable price range ($0.05 - $0.40/kWh)
        min_load = charging_load.min()
        max_load = charging_load.max()
        
        estimated_price = 0.05 + (charging_load - min_load) / (max_load - min_load) * 0.35
        
        # Get hour of day
        if 'Session_Start_Hour' in self.ev_data.columns:
            hours = self.ev_data['Session_Start_Hour'].values
        else:
            # Extract from datetime
            self.ev_data['Date_Time'] = pd.to_datetime(self.ev_data['Date_Time'])
            hours = self.ev_data['Date_Time'].dt.hour.values
        
        # Calculate average price by hour
        hourly_prices = np.zeros(24)
        hourly_price_std = np.zeros(24)
        
        for hour in range(24):
            hour_mask = (hours == hour)
            if hour_mask.sum() > 0:
                hourly_prices[hour] = estimated_price[hour_mask].mean()
                hourly_price_std[hour] = estimated_price[hour_mask].std()
        
        print("\n--- Extracted Price Patterns ---")
        print(f"Average hourly prices ($/kWh):")
        for hour in range(24):
            print(f"  Hour {hour:02d}: ${hourly_prices[hour]:.3f} ± ${hourly_price_std[hour]:.3f}")
        
        return hourly_prices, hourly_price_std, estimated_price
    
    def analyze_trip_patterns(self):
        """Analyze trip demand patterns from dataset"""
        if self.ev_data is None:
            self.load_data()
        
        print("\n--- Trip Pattern Analysis ---")
        
        # Calculate energy consumption per trip
        energy_per_trip = self.ev_data['Energy_Consumption_Rate_kWh/km'] * \
                         self.ev_data['Distance_to_Destination_km']
        
        print(f"\nEnergy per trip statistics:")
        print(f"  Mean: {energy_per_trip.mean():.2f} kWh")
        print(f"  Std: {energy_per_trip.std():.2f} kWh")
        print(f"  Min: {energy_per_trip.min():.2f} kWh")
        print(f"  Max: {energy_per_trip.max():.2f} kWh")
        
        # Trip frequency by hour
        if 'Session_Start_Hour' in self.ev_data.columns:
            hours = self.ev_data['Session_Start_Hour']
        else:
            self.ev_data['Date_Time'] = pd.to_datetime(self.ev_data['Date_Time'])
            hours = self.ev_data['Date_Time'].dt.hour
        
        trip_freq = hours.value_counts().sort_index()
        print(f"\nTrip frequency by hour:")
        for hour in range(24):
            if hour in trip_freq.index:
                print(f"  Hour {hour:02d}: {trip_freq[hour]} trips")
        
        return energy_per_trip, trip_freq
    
    def plot_data_overview(self, save_path: str = None):
        """Create overview plots of the dataset"""
        if self.ev_data is None:
            self.load_data()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. SoC Distribution
        ax = axes[0, 0]
        ax.hist(self.ev_data['State_of_Charge_%'], bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('State of Charge (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of State of Charge', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. Charging Rate Distribution
        ax = axes[0, 1]
        ax.hist(self.ev_data['Charging_Rate_kW'], bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Charging Rate (kW)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of Charging Rates', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Energy Drawn vs Time Charging
        ax = axes[1, 0]
        sample = self.ev_data.sample(n=min(1000, len(self.ev_data)))
        ax.scatter(sample['Time_Spent_Charging_mins'], sample['Energy_Drawn_kWh'], 
                  alpha=0.5, s=10, color='red')
        ax.set_xlabel('Time Spent Charging (mins)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Energy Drawn (kWh)', fontsize=11, fontweight='bold')
        ax.set_title('Energy Drawn vs Charging Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 4. Charging Load by Hour
        ax = axes[1, 1]
        if 'Session_Start_Hour' in self.ev_data.columns:
            hourly_load = self.ev_data.groupby('Session_Start_Hour')['Charging_Load_kW'].mean()
        else:
            self.ev_data['Date_Time'] = pd.to_datetime(self.ev_data['Date_Time'])
            hourly_load = self.ev_data.groupby(self.ev_data['Date_Time'].dt.hour)['Charging_Load_kW'].mean()
        
        ax.bar(hourly_load.index, hourly_load.values, color='orange', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Charging Load (kW)', fontsize=11, fontweight='bold')
        ax.set_title('Average Charging Load by Hour', fontsize=12, fontweight='bold')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def get_typical_battery_params(self) -> dict:
        """Extract typical battery parameters from dataset"""
        if self.ev_data is None:
            self.load_data()
        
        params = {
            'battery_capacity': self.ev_data['Battery_Capacity_kWh'].median(),
            'typical_soc': self.ev_data['State_of_Charge_%'].mean() / 100.0,
            'max_charging_rate': self.ev_data['Charging_Rate_kW'].quantile(0.90)
        }
        
        print("\n--- Typical Battery Parameters ---")
        print(f"Battery Capacity: {params['battery_capacity']:.2f} kWh")
        print(f"Typical SoC: {params['typical_soc']*100:.1f}%")
        print(f"Max Charging Rate: {params['max_charging_rate']:.2f} kW")
        
        return params


def preprocess_data_for_simulation(ev_data_path: str = "ev_charging_dataset.csv",
                                   output_dir: str = "processed_data"):
    """
    Complete data preprocessing pipeline
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    processor = DataProcessor(ev_data_path)
    
    # Load and explore
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    processor.explore_data()
    
    # Extract patterns
    hourly_prices, price_std, estimated_prices = processor.extract_price_patterns()
    energy_per_trip, trip_freq = processor.analyze_trip_patterns()
    params = processor.get_typical_battery_params()
    
    # Create visualizations
    processor.plot_data_overview(save_path=f"{output_dir}/data_overview.png")
    
    # Save processed data
    processed_data = {
        'hourly_prices': hourly_prices,
        'price_std': price_std,
        'estimated_prices': estimated_prices,
        'params': params
    }
    
    np.save(f"{output_dir}/processed_data.npy", processed_data)
    print(f"\nProcessed data saved to '{output_dir}/' directory")
    
    return processor, processed_data


if __name__ == "__main__":
    print("Data Processing Module")
    print("=" * 50)
    
    # Run preprocessing
    processor, data = preprocess_data_for_simulation()
