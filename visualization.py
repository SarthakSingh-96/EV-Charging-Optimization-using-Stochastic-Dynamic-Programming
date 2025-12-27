"""
Visualization Module for EV Charging Optimization
AMS 553 Final Project

This module creates comprehensive visualizations for the project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from simulation import SimulationResult


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EVChargingVisualizer:
    """Comprehensive visualization for EV charging analysis"""
    
    def __init__(self, figsize_default=(12, 6)):
        self.figsize_default = figsize_default
        self.colors = sns.color_palette("husl", 8)
        
    def plot_soc_and_price(self, results: List[SimulationResult], 
                          n_days_to_plot: int = 7,
                          save_path: str = None):
        """
        Plot SoC and electricity prices over time for multiple policies
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        hours_to_plot = n_days_to_plot * 24
        time_hours = np.arange(hours_to_plot) / 24  # Convert to days
        
        # Plot SoC
        ax = axes[0]
        for i, result in enumerate(results):
            ax.plot(time_hours, result.soc_history[:hours_to_plot] * 100, 
                   label=result.policy_name, linewidth=2, alpha=0.8)
        
        ax.axhline(y=20, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Min SoC')
        ax.set_ylabel('State of Charge (%)', fontsize=12, fontweight='bold')
        ax.set_title('Battery State of Charge Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Plot Price
        ax = axes[1]
        # Use price from first result (same for all policies)
        ax.plot(time_hours, results[0].price_history[:hours_to_plot] * 100, 
               color='green', linewidth=1.5, alpha=0.7)
        ax.fill_between(time_hours, 0, results[0].price_history[:hours_to_plot] * 100,
                        alpha=0.3, color='green')
        
        ax.set_xlabel('Time (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Electricity Price (¢/kWh)', fontsize=12, fontweight='bold')
        ax.set_title('Electricity Price Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_charging_actions(self, results: List[SimulationResult],
                             n_days_to_plot: int = 7,
                             save_path: str = None):
        """
        Plot charging actions for different policies
        """
        fig, axes = plt.subplots(len(results), 1, figsize=(14, 3*len(results)), sharex=True)
        
        if len(results) == 1:
            axes = [axes]
        
        hours_to_plot = n_days_to_plot * 24
        time_hours = np.arange(hours_to_plot) / 24
        
        for i, (result, ax) in enumerate(zip(results, axes)):
            ax.bar(time_hours, result.action_history[:hours_to_plot], 
                  width=1/24, color=self.colors[i], alpha=0.7, edgecolor='none')
            
            ax.set_ylabel('Charging Rate (kW)', fontsize=11, fontweight='bold')
            ax.set_title(f'{result.policy_name} - Charging Actions', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 55])
        
        axes[-1].set_xlabel('Time (days)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_cost_comparison(self, comparison_df: pd.DataFrame, save_path: str = None):
        """
        Plot cost comparison across policies
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Average costs by policy
        ax = axes[0]
        cost_summary = comparison_df.groupby('policy_name')[
            ['electricity_cost', 'degradation_cost', 'penalty_cost']
        ].mean()
        
        cost_summary.plot(kind='bar', stacked=True, ax=ax, 
                         color=['#3498db', '#e74c3c', '#f39c12'])
        ax.set_ylabel('Average Cost ($)', fontsize=12, fontweight='bold')
        ax.set_title('Cost Breakdown by Policy', fontsize=14, fontweight='bold')
        ax.legend(['Electricity', 'Degradation', 'Penalty'], fontsize=10)
        ax.set_xlabel('Policy', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Total cost distribution
        ax = axes[1]
        policies = comparison_df['policy_name'].unique()
        data_to_plot = [comparison_df[comparison_df['policy_name'] == p]['total_cost'].values 
                       for p in policies]
        
        bp = ax.boxplot(data_to_plot, labels=policies, patch_artist=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
        ax.set_title('Total Cost Distribution Across Trials', fontsize=14, fontweight='bold')
        ax.set_xlabel('Policy', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_policy_heatmap(self, mdp, save_path: str = None):
        """
        Plot policy heatmap showing optimal charging decisions
        """
        # Create heatmap for a specific hour (e.g., midnight)
        hour = 0
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        hours_to_show = [0, 12, 18]  # Midnight, Noon, Evening
        
        for idx, hour in enumerate(hours_to_show):
            ax = axes[idx]
            
            # Extract policy for this hour
            policy_slice = mdp.policy[:, :, hour]
            actions_slice = mdp.actions[policy_slice]
            
            # Create heatmap
            im = ax.imshow(actions_slice, aspect='auto', cmap='YlOrRd', origin='lower')
            
            # Labels
            ax.set_xlabel('Price State', fontsize=11, fontweight='bold')
            ax.set_ylabel('SoC State', fontsize=11, fontweight='bold')
            ax.set_title(f'Optimal Charging Policy at {hour}:00', 
                        fontsize=12, fontweight='bold')
            
            # Tick labels
            if idx == 0:
                soc_ticks = np.linspace(0, len(mdp.soc_levels)-1, 5, dtype=int)
                ax.set_yticks(soc_ticks)
                ax.set_yticklabels([f'{mdp.soc_levels[i]*100:.0f}%' for i in soc_ticks])
            else:
                ax.set_yticks([])
            
            price_ticks = np.linspace(0, len(mdp.price_levels)-1, 5, dtype=int)
            ax.set_xticks(price_ticks)
            ax.set_xticklabels([f'{mdp.price_levels[i]:.2f}' for i in price_ticks])
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Charging Rate (kW)', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_performance_metrics(self, comparison_df: pd.DataFrame, save_path: str = None):
        """
        Plot various performance metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Average SoC
        ax = axes[0, 0]
        soc_summary = comparison_df.groupby('policy_name')['avg_soc'].agg(['mean', 'std'])
        soc_summary['mean'].plot(kind='bar', ax=ax, color=self.colors, alpha=0.7, 
                                yerr=soc_summary['std'], capsize=5)
        ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Minimum SoC')
        ax.set_ylabel('Average SoC', fontsize=11, fontweight='bold')
        ax.set_title('Average State of Charge', fontsize=12, fontweight='bold')
        ax.set_xlabel('Policy', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Number of violations
        ax = axes[0, 1]
        viol_summary = comparison_df.groupby('policy_name')['num_violations'].agg(['mean', 'std'])
        viol_summary['mean'].plot(kind='bar', ax=ax, color=self.colors, alpha=0.7,
                                 yerr=viol_summary['std'], capsize=5)
        ax.set_ylabel('Number of Violations', fontsize=11, fontweight='bold')
        ax.set_title('SoC Violations (Below Minimum)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Policy', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Number of charging events
        ax = axes[1, 0]
        charge_summary = comparison_df.groupby('policy_name')['num_charges'].agg(['mean', 'std'])
        charge_summary['mean'].plot(kind='bar', ax=ax, color=self.colors, alpha=0.7,
                                   yerr=charge_summary['std'], capsize=5)
        ax.set_ylabel('Number of Charging Events', fontsize=11, fontweight='bold')
        ax.set_title('Charging Frequency', fontsize=12, fontweight='bold')
        ax.set_xlabel('Policy', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Total energy charged
        ax = axes[1, 1]
        energy_summary = comparison_df.groupby('policy_name')['total_energy_charged'].agg(['mean', 'std'])
        energy_summary['mean'].plot(kind='bar', ax=ax, color=self.colors, alpha=0.7,
                                   yerr=energy_summary['std'], capsize=5)
        ax.set_ylabel('Total Energy Charged (kWh)', fontsize=11, fontweight='bold')
        ax.set_title('Total Energy Consumption', fontsize=12, fontweight='bold')
        ax.set_xlabel('Policy', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_daily_pattern(self, result: SimulationResult, save_path: str = None):
        """
        Plot average daily pattern of SoC, price, and charging
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Reshape data into days × hours
        n_days = len(result.soc_history) // 24
        hours = np.arange(24)
        
        # Average by hour of day
        soc_by_hour = result.soc_history[:n_days*24].reshape(n_days, 24).mean(axis=0) * 100
        price_by_hour = result.price_history[:n_days*24].reshape(n_days, 24).mean(axis=0) * 100
        action_by_hour = result.action_history[:n_days*24].reshape(n_days, 24).mean(axis=0)
        
        # Plot SoC
        ax = axes[0]
        ax.plot(hours, soc_by_hour, marker='o', linewidth=2, markersize=6, color='blue')
        ax.fill_between(hours, 0, soc_by_hour, alpha=0.3, color='blue')
        ax.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Min SoC')
        ax.set_ylabel('Avg SoC (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{result.policy_name} - Average Daily Pattern', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Price
        ax = axes[1]
        ax.plot(hours, price_by_hour, marker='s', linewidth=2, markersize=6, color='green')
        ax.fill_between(hours, 0, price_by_hour, alpha=0.3, color='green')
        ax.set_ylabel('Avg Price (¢/kWh)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot Charging Action
        ax = axes[2]
        ax.bar(hours, action_by_hour, color='orange', alpha=0.7, edgecolor='black')
        ax.set_ylabel('Avg Charging (kW)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax.set_xticks(hours)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()


def create_all_visualizations(results: List[SimulationResult],
                             comparison_df: pd.DataFrame,
                             mdp=None,
                             output_dir: str = "figures"):
    """
    Create all visualizations for the project
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = EVChargingVisualizer()
    
    print("\nGenerating visualizations...")
    
    # 1. SoC and Price
    visualizer.plot_soc_and_price(results, n_days_to_plot=14,
                                  save_path=f"{output_dir}/soc_and_price.png")
    
    # 2. Charging Actions
    visualizer.plot_charging_actions(results, n_days_to_plot=7,
                                     save_path=f"{output_dir}/charging_actions.png")
    
    # 3. Cost Comparison
    visualizer.plot_cost_comparison(comparison_df,
                                   save_path=f"{output_dir}/cost_comparison.png")
    
    # 4. Performance Metrics
    visualizer.plot_performance_metrics(comparison_df,
                                       save_path=f"{output_dir}/performance_metrics.png")
    
    # 5. Daily Patterns (for each policy)
    for result in results:
        safe_name = result.policy_name.replace(' ', '_').replace('-', '_').lower()
        visualizer.plot_daily_pattern(result,
                                     save_path=f"{output_dir}/daily_pattern_{safe_name}.png")
    
    # 6. Policy Heatmap (if MDP provided)
    if mdp is not None:
        visualizer.plot_policy_heatmap(mdp,
                                      save_path=f"{output_dir}/policy_heatmap.png")
    
    print(f"\nAll visualizations saved to '{output_dir}/' directory")


if __name__ == "__main__":
    print("Visualization Module")
    print("=" * 50)
    print("This module creates all visualizations for the project.")
    print("Run main.py to generate visualizations automatically.")
