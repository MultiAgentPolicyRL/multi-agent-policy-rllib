exp_name = "experiment 0"
# gym-id unused
learning_rate = 2.5e-4 # learning rate of the optimizer
seed = 1
total_timesteps = 25000
torch_deterministic = True
cuda = True
track = False  # if toggled, this experiment will be tracked with Weights and Biases
wandb_project_name = "PPO-implementation-details" # wandb's project name
wandb_entity = None # the entity (team) of wandb's project
# video unused and unavailable

# num-envs unusable as this env can't be vectorized
num_steps = 128 # the number of steps to run in each environment per policy rollout
anneal_lr = True # Toggle learning rate annealing for policy and value networks
gae = True # Use GAE for advantage computation
gamma = 0.99 # the discount factor gamma
gae_lambda = 0.95 # the lambda for the general advantage estimation
num_minibatches = 4 # the number of mini-batches
update_epochs = 4  # the K epochs to update the policy
norm_adv = True # Toggles advantages normalization
clip_coef = 0.2 # the surrogate clipping coefficient
clip_vloss = True # Toggles whether or not to use a clipped loss for the value function, as per the paper.")
ent_coef = 0.01 # coefficient of the entropy
vf_coef = 0.5 # coefficient of the value function
max_grad_norm = 0.5 # the maximum norm for the gradient clipping
target_kl = None # the target KL divergence threshold

batch_size = num_steps
minibatch_size = batch_size // num_minibatches



args = {'exp_name': "experiment 0",
        # gym-id unused
        'learning_rate': 2.5e-4,  # learning rate of the optimizer
        'seed': 1,
        'total_timesteps': 25000,
        'torch_deterministic': True,
        'cuda': True,
        'track': False,  # if toggled, this experiment will be tracked with Weights and Biases
        'wandb_project_name': "PPO-implementation-details",  # wandb's project name
        'wandb_entity': None,  # the entity (team) of wandb's project
        # video unused and unavailable

        # num-envs unusable as this env can't be vectorized
        'num_steps': 128,  # the number of steps to run in each environment per policy rollout
        'anneal_lr': True,  # Toggle learning rate annealing for policy and value networks
        'gae': True,  # Use GAE for advantage computation
        'gamma': 0.99,  # the discount factor gamma
        'gae_lambda': 0.95,  # the lambda for the general advantage estimation
        'num_minibatches': 4,  # the number of mini-batches
        'update_epochs': 4,  # the K epochs to update the policy
        'norm_adv': True,  # Toggles advantages normalization
        'clip_coef': 0.2,  # the surrogate clipping coefficient
        # Toggles whether or not to use a clipped loss for the value function, as per the paper.")
        'clip_vloss': True,
        'ent_coef': 0.01,  # coefficient of the entropy
        'vf_coef': 0.5,  # coefficient of the value function
        'max_grad_norm': 0.5,  # the maximum norm for the gradient clipping
        'target_kl': None,  # the target KL divergence threshold
        
        # = num_steps
        'batch_size' : 128,
        # = batch_size // num_minibatches
        'minibatch_size' : 128 // 4
        }
