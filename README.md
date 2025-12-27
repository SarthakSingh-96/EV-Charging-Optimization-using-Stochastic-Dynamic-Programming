# EV Charging Optimization using Stochastic Dynamic Programming

An intelligent electric vehicle charging system that uses reinforcement learning and stochastic dynamic programming to minimize charging costs while maintaining optimal battery health under uncertain electricity prices.

---

## 🚗 Overview

This project tackles the challenge of optimizing electric vehicle charging in a dynamic pricing environment. With the rapid adoption of EVs and smart grid technologies, efficiently managing charging schedules can lead to significant cost savings while extending battery life.

### Key Features
- 🔋 **Smart Charging Optimization** - Uses Stochastic Dynamic Programming (SDP) with value iteration
- 💰 **Cost Minimization** - Reduces electricity costs by 15-25% compared to naive charging strategies
- 📊 **Data-Driven** - Trained on real-world EV charging dataset (64,947 records)
- 🤖 **Multiple Policies** - Includes 4+ baseline policies for comparison and benchmarking
- 📈 **Comprehensive Analytics** - Monte Carlo simulation framework with detailed visualizations
- ⚡ **Battery Health Focus** - Considers degradation costs and optimal State of Charge maintenance

---

## 🎯 Problem Statement

Electric vehicle owners face several challenges:
- **Variable electricity prices** that change throughout the day
- **Uncertain trip demands** requiring adequate battery charge
- **Battery degradation** costs from frequent charging
- **Need for cost-effective** charging strategies

This system learns an optimal charging policy that balances these competing objectives using reinforcement learning techniques.

---

## 📁 Project Structure

```
EV-Charging/
├── ev_charging_dataset.csv          # Real-world EV charging data
├── location_dataset.csv             # Location and trip information
├── requirements.txt                 # Python dependencies
│
├── main.py                          # Main execution pipeline
├── ev_charging_sdp.py              # Core SDP algorithm implementation
├── baseline_policies.py            # Alternative charging strategies
├── simulation.py                   # Monte Carlo simulation engine
├── visualization.py                # Analytics and plotting tools
├── data_processing.py              # Data preprocessing pipeline
│
└── results/                        # Generated outputs
    ├── figures/                    # Visualization plots
    ├── models/                     # Trained policy models
    ├── policy_comparison.csv       # Performance metrics
    └── summary_report.txt          # Analysis summary
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SarthakSingh-96/EV-Charging.git
cd EV-Charging
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the System

Execute the complete optimization pipeline:

```bash
python main.py
```

This will:
1. 📊 Preprocess and analyze the EV charging dataset
2. 🧠 Train the SDP model using value iteration
3. 🎲 Simulate policies over 10 trials × 365 days each
4. 📈 Generate comprehensive visualizations
5. 📄 Create performance reports and comparisons

**Expected Runtime:** 15-30 minutes (varies by hardware)

---

## 🧠 Technical Approach

### Markov Decision Process Formulation

The charging optimization problem is modeled as an MDP with:

**State Space:** `(SoC, Price, Hour)`
- **State of Charge (SoC):** 21 discrete levels (0%, 5%, ..., 100%)
- **Electricity Price:** 10 discrete levels ($0.05 - $0.50/kWh)
- **Time of Day:** 24 hours

**Action Space:** Charging rate selection
- 11 discrete charging rates (0, 5, 10, ..., 50 kW)

**Cost Function:**
```
Total Cost = Electricity Cost + Battery Degradation Cost + Penalty for Low SoC
```

**Optimization Method:** Value Iteration with Monte Carlo sampling

### Baseline Policies for Comparison

1. **Myopic Policy** - Charge only when critically low (greedy approach)
2. **Fixed Schedule Policy** - Predetermined charging times (e.g., midnight)
3. **Threshold-Based Policy** - Charge when price < threshold AND SoC < target
4. **Time-of-Use Optimized** - Smart charging based on typical price patterns
### Simulation & Evaluation

**Monte Carlo Simulation Engine**
- Evaluates policies over 365-day periods
- Runs multiple trials for statistical significance
- Tracks costs, SoC violations, and charging patterns
- Compares policy performance across metrics

**Visualization Suite**
- Battery SoC trajectories over time
- Electricity price patterns and trends
- Charging decision timelines
- Policy heatmaps showing learned behavior
- Comprehensive cost breakdowns and comparisons

### Data Processing Pipeline

The system analyzes real-world EV charging data to:
- Extract realistic electricity price patterns
- Model trip demand distributions
- Calibrate battery parameters
- Generate data insights and statistics

---

## 📈 Results & Performance

### Key Achievements

The SDP-optimized policy demonstrates:
- ✅ **15-25% cost reduction** compared to myopic/greedy strategies
- ✅ **Fewer SoC violations** - better battery health maintenance
- ✅ **Intelligent price-responsive behavior** - charges during low-price periods
- ✅ **Adaptive decision-making** based on current state and predictions

### Generated Visualizations

**1. State of Charge & Price Dynamics**
- Real-time battery charge levels for all policies
- Overlaid electricity price patterns
- Clear demonstration of policy responsiveness

**2. Policy Heatmaps**
- Optimal charging decisions across (SoC, Price) states
- Learned behavior patterns at different times of day
- Visual representation of strategic charging logic

**3. Cost Analysis**
- Detailed breakdown: electricity, degradation, penalties
- Statistical comparison across multiple trials
- Performance ranking and insights

**4. Performance Metrics**
- Average SoC maintenance levels
- Violation frequency analysis
- Charging pattern statistics
- Total energy consumption tracking

---

## 🧪 Testing Individual Components

Run specific modules independently:

```bash
# Test data processing and analysis
python data_processing.py

# Test baseline charging policies
python baseline_policies.py

# Quick demo of SDP core algorithm
python ev_charging_sdp.py
```

---

## ⚙️ Configuration

Customize the optimization parameters in `main.py`:

```python
# EV Specifications
ev_params = EVParameters(
    battery_capacity=75.0,           # kWh
    min_soc=0.20,                    # Minimum safe charge level (20%)
    max_charging_rate=50.0,          # Maximum kW
    degradation_cost_per_kwh=0.05,   # Battery wear cost ($/kWh)
    penalty_cost_per_kwh=10.0,       # Low charge penalty ($/kWh)
    soc_states=21,                   # State space discretization
    price_states=10,
    action_states=11
)

# Simulation Parameters
n_days = 365          # Simulation period
n_trials = 10         # Number of independent runs
gamma = 0.95          # Discount factor for future costs
n_iterations = 100    # Value iteration steps
```

---

## 📊 Dataset

**EV Charging Dataset** - 64,947 real-world charging records

Includes:
- Vehicle specifications (ID, battery capacity, SoC)
- Energy consumption and efficiency metrics
- Charging station information and rates
- Environmental factors (temperature, weather)
- Traffic conditions and trip patterns
- Session timing and duration data

**Source:** Kaggle EV Charging Dataset  
**Purpose:** Realistic modeling of price uncertainty and trip demand patterns

---

## 🔬 Technologies Used

- **Python 3.8+** - Core programming language
- **NumPy** - Numerical computations and matrix operations
- **Pandas** - Data processing and analysis
- **Matplotlib/Seaborn** - Visualization and plotting
- **SciPy** - Statistical distributions and optimization
- **Pickle** - Model serialization and storage

---

## 🚀 Future Enhancements

- 🔌 **Vehicle-to-Grid (V2G)** integration for bi-directional charging
- ☀️ **Renewable energy** integration (solar/wind forecasting)
- 🌐 **Multi-vehicle fleet** optimization
- 📱 **Real-time adaptation** with online learning
- 🧪 **Deep reinforcement learning** approaches (DQN, PPO)
- 📍 **Location-aware** charging station selection

---

## 📚 References

1. Puterman, M. L. (2014). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.
2. Bertsekas, D. P. (2017). *Dynamic Programming and Optimal Control*. Athena Scientific.
3. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
4. EV Charging Dataset - Kaggle Open Data
5. IEEE Transactions on Smart Grid - EV Charging Optimization Literature

---

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Share your results and modifications

---

## 📧 Contact

For questions or collaboration opportunities:
- GitHub: [@SarthakSingh-96](https://github.com/SarthakSingh-96)
- Repository: [EV-Charging](https://github.com/SarthakSingh-96/EV-Charging)

---

**Built with ❤️ for sustainable transportation and smart energy management**
