- imlementare ppo_train, dt_ql_train, interact
    - ppo e' diverso da dt_ql_train principalemtne perche' crea e usa la batch
- portare i modelli pytorch in /src/common/models
- in /src/train/ creare cartella `algorithms` che conterra' i 2 algoritmi di apprendiemnto: online learning e offline learning.
- portare `rollout_buffer` in un file di PPO -> solo lei lo usa
- portare `exec_time` in /src/common
- creare modulo per la comunicazoine tra processi -> salvataggio dati

1. portare `algorithm` e i suoi parametri in `ppo_train`
    1. creare modulo comunicazione
2. portare `rollout_worker` e configurarlo correttamente
