# Deep toughts about PPO loss implementation for this project

# In `model`:
## `forward`
```
def forward(self, observation: dict):
    """
        Model's forward. Given an agent observation, action distribution and value function
        prediction are returned.

        Args:
            observation: agent observation

        Returns:
            logits: actions probability distribution
            value: value function prediction
    """
```
I would modify returns:
- policy_action: (single) action taken by the actor, example: `2`
- policy_probability: (logits) probability distribution (after `action_mask` and `torch.sigmoid` are applied)
- value_function: value function action prediction -> now it's called `value`

=> this changes need to be propagated where `forward` is used: `policy.act`
    => `algorithm.get_actions`
        => `memory`

## fit
create pseudo code from here:
    [code](https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py)
## loss
Integrated in `fit`

# In `memory`:
Memory is a top level class used only by algorithm. So it has to manage multi-agent inputs. 

    Example of data structure:
        observation: {
            '0': [...],
            '1': [...],
            ...
            'p': [...]
        }

        policy_action: {
            '0': [...],
            '1': [...],
            ...
            'p': [...]
        }

## `__init__`:
It should manage multiple agents data such as:
- `observation`
- `policy_action`
- `policy_probability` (logits)
- `vf_action`
- `reward`
- `done` -> not needed

## `reset_memory`:
Clears the memory keeping data structures (TODO)
## `update_memory`:
```
def update_memory(observation, policy_action, policy_probability, vf_action, reward):
"""
    Splits each input for each agent and appends its data to the correct structure

    Args:
        observation: environment observation with all agents ('0', '1', '2', '3', 'p')
        policy_action: policy's taken action, single for '0', '1', '2', '3', multiple for 'p'
        policy_probability: policy's action distribution for each agent
        vf_action: value function action prediction for each agent
        reward: environment given reward for each agent

    Return:
        nothing
"""
```
## `get_memory`:
Add `epochs` (how many agents in the batch) and `steps_per_epoch` (batch size per agent)
```
def get_memory(self, key):
"""
    Each memorized input is retrived from the memory and merged by agent
    
    Return example:
        observations = [observation['0'], observation['1'], observation['2'], observation['3']]
        policy_actions = [policy_action['0'], policy_action['1'], policy_action['2'], ...

    Args:
        key: selected group

    Returns:
     observations, policy_actions, policy_probabilitiess, value_functions, rewards, epochs, steps_per_epoch
"""
    
```
    
# In `policy`:
## `act`
It will get simplified:
```
def act(self, state):
    with torch.no_grad():
        policy_action, policy_probability, vf_action = self.Model(input_state)
    
    return policy_action, policy_probability, vf_action
```
## `learn`
TODO
Probably it will just call `Model.fit` and wait for it to execute and improve the network.
## `get_gaes`
Moved to `Model`

# In `algorithm`:
## `get_actions`
Modify how it interacts with `policy.act`
```
def get_actions(self, observation: dict) -> dict:

"""
    Build action dictionary from env observations. 

    Output has this structure:

        policy_action: {
            '0': ...,
            '1': ...,
            '2': ...,
            ...
            'p': [...]
        }

    FIXME: Planner

    Arguments:
        observation: observation of the environment, it contains all observations for each agent

    Returns:
        policy_actions dict: predicted actions for each agent
        policy_probability dict: action probabilities for each agent
        vf_actions dict: value function action predicted for each agent
"""
```

## `data_preprocess`
```
def data_preproces(self, observation: dict) -> dict:
"""
    Takes as an input a dict of np.arrays and trasforms them to Torch.tensors.

    Args:
        observation: observation of the environment\

    Returns:
        observation_tensored: {
            '0': {
                'var': Tensor
                ...
            },
            ...
        }
"""
```
The idea behind `data preproces` is that the input of the nn must be Torch.tensor (otherwise it doesn't work), so we can prepare all the data not only before the network, but also before memory, so that when `Model.fit` is called all the data just needs to be retrieved but not preprocessed again.

## `train_one_step` - `batch`
All we need to do is merge the two functions and have everything easier to understand.
