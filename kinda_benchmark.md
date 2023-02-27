# Compare data transfer file vs queue
I'm not going to try that: I have to implement the DT+Q_learning stuff, I'm not going to do anything else.

# Compare pickling a class or a dict with same internal structure. -> to do this training is disabled
EPOCHS = 1
BATCH_SIZE = 6000
SEED = 1
NUM_WORKERS = 12
rollout_fragment_length = 200
K_epochs = 16

## with class
UNPICKLING TIME: 1.1958273330001248
Function train_one_step Took 3.1188907399991876 seconds

## with dict
UNPICKLING TIME: 1.1831192890003877
Function train_one_step Took 3.1823382780003158 seconds