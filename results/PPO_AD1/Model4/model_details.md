# PPO Model: 111python

### Model Summary:
- **Model**: PPO
- **Simple reward function**: `reward = -(curr-prev)`
- **Convergence**: Got converged around `-0.95`
- **Training**: Trained for around `1.2m steps`
- **Further Work**:
  - Add Occupancy and forecasting constraint
  - Train for more variables
  - Try with continuous actions
- **Action Space**: Discrete actions with `bins=15`
- **Performance**: For peak heat day, `90k` model is performing better than the `1.2m` model.
- **Info**: model_ppo_90k if performing better than ppo_model_120000_latest_final

### Python Code Implementation

```python
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

def train_PPO_with_callback(model_path=None,
                            log_dir=os.path.join('results', 'PPO', 'Model3'),
                            tensorboard_log=os.path.join('results', 'PPO', 'Model3')):
    """
    Method to train a PPO agent using a callback to save the model periodically.

    Parameters
    ----------
    model_path : str, optional
        Path to a pre-trained model. If provided, the model will be loaded and further trained.
    log_dir : str
        Directory where monitoring data and best-trained model are stored.
    tensorboard_log : str
        Path to directory to load tensorboard logs.
    """
    
    excluding_periods = []
    excluding_periods.append((173*24*3600, 266*24*3600))  # Summer period

    env = BoptestGymEnvCustomReward(
        url=url,
        actions=['ahu_oveFanSup_u', 'oveValCoi_u', 'oveValRad_u'],
        observations={
            'time': (0, 31536000),
            'reaTZon_y': (200., 400.),
            'reaCO2Zon_y': (200., 2000.),
            'weaSta_reaWeaTDryBul_y': (250., 350.)
        },
        scenario={'electricity_price': 'highly_dynamic'},
        random_start_time=True,
        max_episode_length=3*24*3600,
        step_period=3600,
        log_dir=tensorboard_log,
        excluding_periods=excluding_periods
    )

    print(env.observation_space)
    # env = NormalizedObservationWrapper(env)
    # env = NormalizedActionWrapper(env)  
    env = DiscretizedActionWrapper(env, n_bins_act=15)
    os.makedirs(log_dir, exist_ok=True)
    
    env = Monitor(env=env, filename=os.path.join(log_dir, 'monitor.csv'))
    
    # Callback to save model every 2000 steps
    callback = SaveAndTestCallback(check_freq=48, save_freq=500, env=env, log_dir=tensorboard_log)
    
    # Set up logger with TensorBoard logging continuation
    new_logger = configure(log_dir, ['stdout', 'csv', 'tensorboard'])
    
    # Load existing model if model_path is given, else create a new one
    if model_path and os.path.isfile(model_path):
        model = PPO.load(model_path, env=env, tensorboard_log=tensorboard_log)
        print(f"Loaded pre-trained model from {model_path}")
        model.set_logger(new_logger)  # Reconfigure the logger to continue logging
    else:
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=1, 
            gamma=0.99,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            clip_range=0.2,
            gae_lambda=0.95,
            ent_coef=0.01,
            tensorboard_log=tensorboard_log,
        )
        model.set_logger(new_logger)
        print("Starting training from scratch.")
    
    # Train the agent with the callback
    model.learn(total_timesteps=int(500000), callback=callback)
    
    return env, model

if __name__ == "__main__":
    model_path = "results/PPO/Model3/ppo_model_81000_latest.zip"  # Update this with the correct path if needed
    env, model = train_PPO_with_callback(model_path=model_path)
    model.save(os.path.join('results', 'PPO', 'final_model'))
    print("Training completed. Model saved in results/PPO/")
    print("TensorBoard logs saved in results/PPO/")
