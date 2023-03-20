# Here we define all standard methods and practices that have been used on this project

## in '/experiments/'
- all new experiments have a standardized name made of many parts sticked together with an underscore. An example is: ;  INT_PPO_DATA_model-exp-id_STEPS
    - There are two different styles: one for Train and one for Interact.
    Train example: `PPO_P1_01-01-1970_111111_200`  
    Interact example: `INT_01-01-1970_1000_PPO_P1_01-01-1970_111111_200`
    - Train
        - `PPO`/`DT`: references to the used training algorithm
        - `P1`/`P2`: refers to Phase 1 and Phase 2 of the experiment, so if we are training only agents, or both agents and the social planner
        - date+time.now: unique experiment id, easy to read
        - steps: steps done during this training, useful to have an idea of how well this model should behave in the environment.
    - Interact 
        - `INT`: we are just playing with the environment, everything in this directory is not about training.
        - date+time.now: as for `Train`
        - steps: how many steps the environment has been stepped for
        - source_models_id: reference to the model's source
    
- As already said there are two different types of directories which contains:
    - Train
        - config.yaml
        - models/
            - a.pt
            - p.pt
        - logs/... : logs of each worker to create graphs during training
            - at the moment (13-03-2023) only PPO supports it
        - plot (reward, losses)
    - Interact
        - config.yaml
        - models copy
        - plot (environment plots)
        - saved environment

- Models' names will be with this shape:
    - a.pt
    - p.pt
    nothing more.