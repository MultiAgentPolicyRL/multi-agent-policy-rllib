Update e controlli da fare:
- provare a passare i dati con le code
- [FATTO] spostare il to_tensor dal rollout_buffer alla policy -> la policy quando riceve il dato, se necessario, chiama "to tensor" (cosi si evita cast inutili e, in caso, si puo' passare tutto a gpu senza troppi problemi)
- cambiare come vengono salvati i reward dai rollout_worker -> fare in modo che vengano restituiti direttamente al processo padre e sia lui a tenerli (cosi quando l'esecuzione termina vengono salvati in un file e via) - oppure - lasciare cosi' com'e' e trovare un'opzione buona per gestire la policy_empty e quelli che non sono policy in modo decente -> forse fare check sul tipo di classe che viene chiamata puo' tornare utile
- IMPLEMENTARE IL DT -> non e' policy_learning, quindi non ha cose da restituire. Forse alla fine permette di visualizzare l'albero, ma non ne sarei cosi' sicuro.
    - come singolo dato: flat
    <!-- - rew totale solo planner -->