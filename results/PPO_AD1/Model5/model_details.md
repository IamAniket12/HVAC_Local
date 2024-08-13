# PPO Model Training Script

This document describes the training of a model using the Proximal Policy Optimization (PPO) algorithm.

### Key Details:

- **Model**: The model is trained using PPO.
- **Forecasting Variables**: The model incorporates forecasting variables.
- **Forecasting Horizon**: The forecasting horizon is set to `5 * 3600` seconds.
- **Observation State**: The observation state has a shape of `(22,)`.
- **Reward Function**: The reward is a simple difference calculation, defined as `-(curr - prev)`.

### Python Script

```python
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

def train_PPO_with_callback(model_path=None,
                            log_dir=os.path.join('results', 'PPO_AD1', 'Model5'),
                            tensorboard_log=os.path.join('results', 'PPO_AD1', 'Model5')):
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
            'weaSta_reaWeaTDryBul_y': (250., 350.),
            'PriceElectricPowerHighlyDynamic':(-0.4,0.4),
            'LowerSetp[1]':(280.,310.),
            'UpperSetp[1]':(280.,310.),
        },
        predictive_period     = 5*3600,
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
    env = DiscretizedActionWrapper(env,n_bins_act=15)
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
    model_path = "results/PPO_AD1/Model5/model_500_latets.zip"  # Update this with the correct path if needed
    env, model = train_PPO_with_callback(model_path=model_path)
    model.save(os.path.join('results', 'PPO', 'final_model'))
    print("Training completed. Model saved in results/PPO/")
    print("TensorBoard logs saved in results/PPO/")
