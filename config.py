def get_config(weight, model = 'CB_APM'):
    if model == 'CB_APM':
        return {
            # model
            'input_size': 146,
            'concept_hidden_sizes': [64, 32],
            'final_hidden_sizes': [],
            'concept_output_size': 9,
            'final_output_size': 1,

            # learning
            'lr': 0.0005,
            'epochs': 100,
            'early_stopping_patience': 5,
            'scheduling_patience': 2,
            'scheduling_factor': 0.2,
            'batch_size': 5000,
            'ensemble': 10,

            # Regularization
            'weight_decay': 0.005,
            'clip_value': 1,

            # weight lambda
            'weight_lambda': weight
        }
    elif model == 'autoencoder':
        return {
            'num_runs': 1,
            'num_epochs': 30_000,
            'patience': 2_500,
            'min_delta': 1e-5,
            'print_every': 100
        }