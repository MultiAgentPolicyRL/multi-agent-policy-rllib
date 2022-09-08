## AI-Econimist trained with PPO and Interpretable Decision Trees

## Project goal
Train AI-Economist agents with PPO (as in the paper) and the social planner with interpretable decision trees, then compare the results.

## AI-Economist
[The AI Economist: Optimal Economic Policy Design via Two-level Deep Reinforcement Learning](https://arxiv.org/abs/2108.02755)

## Algorithms
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[Evolutionary learning of interpretable decision trees](https://arxiv.org/abs/2012.07723)

## Installation instructions
*Installing from Source*  
1. Clone this repository to your machine
`git clone https://github.com/sa1g/ai-economist-ppo-decision-tree`
2. Create a new conda environment and activate it
`conda create --name ai-economist python=3.7 --yes`
`conda actiavte ai-economist`
3. Install dependencies
`pip install -r requirementsUpdated.txt`
4. The 3 Levels custom training is located at
`ai-economist/tutorials/rllib/training_2_algos.py`


