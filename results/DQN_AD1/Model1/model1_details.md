# Model Training Summary

### Training Details
- **Model:** Trained on 3 million time steps
- **Final Loss:** 0 (Loss plot looks good)
- **Reward:** The reward did not converge at 3 million steps, and it hovered around -3.5

### Environment Details
- **Environment:** `singlezone_commercial_hydronic`
- **Reward Function:** Custom reward function used, which is simple:  
  `reward = -(current - prev)`
- **Performance:**
  - **Peak Heat Day:** Showed good performance with a value around `0.000005`
  - **Typical Heat Day:** Performance was worse than the baseline

### Key Points to Notice
- **Action Bins:** The actions were discretized into 3 bins, so the agent could only operate at 0, 0.33, or 0.66
- **Forecast Variables:** No forecast variables were used
- **Loss Function:** The loss function looked perfect, with a final value of 0 at the end of training

---

# Python Training Script

```python
def train_DQN_with_callback(log_dir=os.path.join('results','DQN'),
                            tensorboard_log=os.path.join('results','DQN')):
    '''Method to train a DQN agent using a callback to save the model 
    upon performance improvement.  
    
    Parameters
    ----------
    log_dir : string
        Directory where monitoring data and the best trained model are stored.
    tensorboard_log : path
        Path to directory to load tensorboard logs.
    '''
    
    env = BoptestGymEnvCustomReward(
        url=url,
        actions=['ahu_oveFanSup_u', 'oveValCoi_u', 'oveValRad_u'],
        observations={
            'reaTZon_y': (200., 400.),       # Current zone air temperature
            'reaCO2Zon_y': (200., 2000.),    # Current zone CO2 concentration
            'reaPFan_y': (0., 100000.),
            'reaPPum_y': (0., 100000.),
            'weaSta_reaWeaTDryBul_y': (250., 350.),
            'weaSta_reaWeaWinSpe_y': (0., 100.)
        },
        scenario={'electricity_price': 'highly_dynamic'},
        random_start_time=True,
        max_episode_length=7*24*3600,
        step_period=3600,
        log_dir=tensorboard_log     
    )
    
    # Discretizing actions into 3 bins
    env = DiscretizedActionWrapper(env, n_bins_act=3)
    os.makedirs(log_dir, exist_ok=True)
    
    # Monitor the environment with callbacks
    env = Monitor(env=env, filename=os.path.join(log_dir, 'monitor.csv'))
    
    # Callback setup: check every 48 steps, save every 1000 steps
    callback = SaveAndTestCallback(env=env, check_freq=48, save_freq=1000, log_dir=tensorboard_log)
    
    # Initialize the DQN agent
    model = DQN(
        'MlpPolicy', 
        env, 
        verbose=1, 
        gamma=0.98,  
        learning_rate=1e-3,  
        batch_size=128,  
        buffer_size=10000,  
        learning_starts=1000,  
        train_freq=8,  
        gradient_steps=4,  
        target_update_interval=500,  
        exploration_fraction=0.6,  
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,  
        tensorboard_log=tensorboard_log
    )
    
    # Set up logger
    new_logger = configure(log_dir, ['stdout', 'csv', 'tensorboard'])
    model.set_logger(new_logger)

    # Train the agent with callback for saving
    model.learn(total_timesteps=int(500000), callback=callback)
    
    return env, model

if __name__ == "__main__":
    env, model = train_DQN_with_callback()
    model.save("sac_adrenalin1_multiple_action_")
    print("Training completed. Best model saved in results/monitored_DQN/")
    print("TensorBoard logs saved in results/tensorboard_logs/")
