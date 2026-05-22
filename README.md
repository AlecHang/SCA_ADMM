## Configuration Guide

The main configuration file is `Simulations/config.yml`, which includes the following sections:

### Basic Simulation Parameters
- `simulation_time`: Simulation duration (seconds)
- `cache_capacity`: Cache capacity
- `bin_size`: Size of each bin
- `nb_videos`: Number of videos per CP
- `request_rate`: Number of requests per second
- `interval_size`: Interval duration (seconds)
- `delta`: Step size for allocation updates
- `method`: Optimization method (SARSA, Q_learning, R_learning)
- `D`: Set of coefficients for delta

### Service Provider Parameters
- `count`: Number of service providers
- `probabilities`: Request probability for each SP
- `cacheability`: Cacheable ratio for each SP
- `zipf_alphas`: Zipf distribution alpha for each SP

### Inter-Node Latency Configuration
- `enabled`: Whether to enable network latency simulation
- `nodes`: Number of nodes
- `latency_matrix`: Inter-node latency matrix (milliseconds)
- `bandwidth`: Bandwidth (Mbps)

## Usage

1. **Modify configuration**: Edit the `config.yml` file as needed.
2. **Run simulation**:
   ```bash
   cd Simulations
   python simulation_code.py
   ```
   Or use the execution script:
   ```bash
   cd Simulations
   ./run.sh
   ```

3. **View results**:
   - Result data is saved in the `results/` directory
   - Figures are saved in the `figures/` directory

## Output Description

- **Allocation history**: `allocations_*.txt` files record cache allocations for each iteration.
- **Result data**: `results_*.csv` files contain cost and latency data.
- **Cost figure**: `cost_evolution_*.png` shows cost evolution over time.
- **Latency figure**: `latency_evolution_*.png` shows latency evolution over time (only when network simulation is enabled).

## Cache Allocation Methods

1. **Equal allocation** (`equal_allocation`): Divides the cache capacity equally among all service providers.
2. **Best allocation** (`best_allocation`): Computes the optimal cache allocation based on historical request data.
3. **SCA_ADMM** (`SCA_ADMM`): Uses the SCA-ADMM algorithm for cache allocation optimization.
4. **Proportional allocation** (`proportional_allocation`): Allocates cache proportionally according to the request probabilities of service providers.

### Proportional Allocation Method

The proportional allocation method is a simple yet effective cache allocation strategy that distributes cache space based on the request probabilities of service providers. The specific steps are as follows:

1. Calculate the request probability share for each service provider.
2. Allocate the cache capacity of each node proportionally to each service provider according to their request probability shares.
3. Handle remainders during allocation to ensure the total allocated amount equals the node's cache capacity.
4. Ensure all allocations are non-negative.

This method is simple to compute, fast to execute, and automatically adjusts the allocation ratio based on the request frequency of service providers.

## Example Configuration

The default configuration simulates a scenario with 3 service providers and 3 nodes, with latency differences among nodes. You can modify the configuration file to simulate different scenarios as needed.

## Dependencies

- Python 3.6+
- numpy
- pandas
- matplotlib
- pyyaml
