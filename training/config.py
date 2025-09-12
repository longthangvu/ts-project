from util.config_util import ShapeConfig, dotdict

def get_C_Q_ranges():
    C_range = (4, 64)
    Q_range = (1, 8)
    return C_range, Q_range

def get_config_shape():
    # Configuration for synthetic data generation
    return ShapeConfig(
            n_context=16,
            n_sequence=500,  # Length of synthetic series
            n_features=1,    # Univariate series
            n_heldout=4,
            n_prompt=60
        )

def get_hyperprior_params(seasonality_base = 2.0):
    # Hyperprior parameters for realistic time series
    w, m, a = seasonality_base*1, seasonality_base*2, seasonality_base*4
    # Seasonality parameters
    return dotdict({
            'a_min': -a, 'a_max': a, 'a_fixed_variance': 0.15,
            'm_min': -m, 'm_max': m, 'm_fixed_variance': 0.15,
            'w_min': -w, 'w_max': w, 'w_fixed_variance': 0.15,
            
            # Trend parameters
            'trend_lin_min': -0.01, 'trend_lin_max': 0.01, 'trend_lin_fixed_variance': 0.005,
            'trend_exp_min': 1 - 0.003, 'trend_exp_max': 1 + 0.003, 'trend_exp_fixed_variance': 0.001,
            'trend_exp_multiplier': 507,
            
            # Noise and resolution
            'noise_k_min': 0.5, 'noise_k_max': 3.0,
            'resolution_min': 0.1, 'resolution_max': 1.0, 'resolution_multiplier': 53.6,
            
            # Other parameters
            'harmonics_min': 4, 'harmonics_max': 8,
            'discreteness_min': 1, 'discreteness_max': 5,
            'bias_zi_min': 1, 'bias_zi_max': 5,
            'amplitude_min': 1, 'amplitude_max': 5,
            'non_negative_prob': 0.3,
            'offset_lin_min': -0.5, 'offset_lin_max': 1.0,
            'offset_exp_min': -0.5, 'offset_exp_max': 1.0,
            'f_zi_min': 0.0, 'f_zi_max': 0.8, 'f_zi_fixed_variance': 0.3
        })
    
def get_train_config():
    return {
        'total_tasks': 500_000,          # 500K tasks
        'tasks_per_epoch': 5_000,        # 5K tasks per epoch  
        'validate_every': 10,            # Validate every 10 epochs
        'save_every': 20,                # Save every 20 epochs
        'log_every': 500,                # Log every 500 tasks
        'n_val_tasks': 100,              # Validation tasks
        'early_stopping_patience': 20,   # Patience for 500K training
        'warmup_tasks': 5_000,           # LR warmup period
    }