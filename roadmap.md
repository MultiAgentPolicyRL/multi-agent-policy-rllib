## About PPOTrainer (RLLib) and config.yaml
In `config.yaml` trainer section is about how the PPO trainer should work. 


## Roadmap
**29 lgl 22**: abandoning the idea of casting data and using self made everything. Probabily it's easier to just create a custom algorithm in RLLib.  

**1 ago 22**: l'idea attuale e' di fare un unico `Trainer` che contiene `PPO` + `Elidt`. Devo poter sincronizzare la batch di `PPO` con l'altro in un modo sensato -> ogni quanto sincronizzo? Sarebbe giusto ad ogni step della batch?


**Riflessioni**
Attualmente devo leggere come e' sviluppato l'algoritmo del prof, svilupparlo e testarlo con CartPole -> cosi' so come e quanto funziona.  
Successivamente inizio a lavorare su una `custom policy` per `RLLib` includendo solo `PPO`. Testato che `PPO` funziona posso implementare anche `Elidt` e continuare il progetto. Sara' utile leggere bene [custom training workflow](https://github.com/ray-project/ray/blob/master/rllib/examples/two_trainer_workflow.py) che forse ha qualcosa di utile.  

https://docs.ray.io/en/releases-1.4.1/rllib-training.html#advanced-python-apis

I need to do a `custom training loop` in RLLib -> qui ci sono 2 env separati, dubito torni utile.
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_train_fn.py

and a custom policy (policy == algorithm)
https://docs.ray.io/en/releases-1.4.1/rllib-concepts.html

execution-plans: easily express the execution of an RL algorithm as a sequence of steps that occur either sequentially in the learner, or in parallel across many actors -- concurrent training available via a [custom training workflow](https://github.com/ray-project/ray/blob/master/rllib/examples/two_trainer_workflow.py).  TODO: leggerlo bene.  
https://docs.ray.io/en/releases-1.4.1/rllib-concepts.html#example-multi-agent

TODO: leggere algoritmo del prof.

**19 ago 22**: Abbandonata l'idea di usare RLLib, riduciamo la complessita' del progetto, susu.
Ho creato `understanding AI-Economist` e `env_wrapper_tmp` (senza RLLib), ora so che i dati passati per il wrapper sono tutti come dict di `numpy`. 
Ora devo studiare bene come funziona la PPO, ho capito l'idea generale, ma devo capire come funziona `GAE (generalized advantage estimation)` (https://arxiv.org/pdf/1506.02438.pdf).  
Utile da studiare e' anche `Distributed PPO(DPPO)`  (https://arxiv.org/pdf/1707.02286.pdf) (https://i.stack.imgur.com/AYjQN.png) (https://github.com/TianhongDai/distributed-ppo)

questo video c'e' il tipo che spiega cosa c'e' scritto nel paper, nice https://www.youtube.com/watch?v=HR8kQMTO8bk