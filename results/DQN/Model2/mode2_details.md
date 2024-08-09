# DQN Training Summary

## Overview
This document provides details of the DQN training conducted with custom reward functions. The training focused on a specific environment with several parameters adjusted to optimize performance.

## Training Setup
- **Algorithm:** DQN (Deep Q-Network)
- **Training Steps:** 3 million time steps
- **Number of Bins:** 5
- **Excluded Periods:** Some periods where there was nothing to learn were excluded from training.

## Reward Function
A custom reward function was implemented to calculate the objective based on several KPIs. The reward function is detailed below:

```python
import numpy as np
import requests

class BoptestGymEnvCustomReward(BoptestGymEnv):
    
    def calculate_objective(self, kpis):
        """
        Calculate the objective based on the given KPI values.
        """
        cost_tot = kpis.get('cost_tot', 0) or 0
        pdih_tot = kpis.get('pdih_tot', 0) or 0
        pele_tot = kpis.get('pele_tot', 0) or 0
        tdis_tot = kpis.get('tdis_tot', 0) or 0
        idis_tot = kpis.get('idis_tot', 0) or 0

        objective = (
            cost_tot +
            4.25 * (pdih_tot + pele_tot) +
            0.005 * tdis_tot +
            0.0001 * idis_tot
        )

        return objective

    def get_reward(self):
        try:
            # Use this one running on local server
            kpis = requests.get(f'{self.url}/kpi').json()['payload']
        except requests.exceptions.RequestException as e:
            print(f"Error fetching KPIs: {e}")
            return 0  # In case of error, return zero reward

        current_objective = self.calculate_objective(kpis)

        self.step_count += 1

        if self.objective_integrand is not None:
            # Calculate the increase in objective
            increase = current_objective - self.objective_integrand
            print("Current objective:", current_objective)
            print("Previous objective:", self.objective_integrand)
            # Update cumulative increase
            self.cumulative_increase += abs(increase)

            # Calculate average absolute increase up to this point
            average_increase = self.cumulative_increase / self.step_count

            # Calculate relative performance (closer to 0 is better)
            relative_performance = increase / (average_increase + self.epsilon)

            # Transform relative performance to a reward between -1 and 1
            reward = -np.tanh(relative_performance)

            # Add a small positive reward if increase is less than average
            if abs(increase) < average_increase:
                reward += 0.1

            # Add a larger reward if we managed to decrease the objective
            if increase < 0:
                reward += 0.5
        else:
            reward = 0
        print("Reward:", reward)
        # Update the stored previous objective
        self.objective_integrand = current_objective

        return reward
