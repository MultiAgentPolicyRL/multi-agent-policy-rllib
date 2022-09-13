import os

ckpt = 'phase1/ckpts/agent.tf.weights.global-step-4000' 
assert os.path.isfile(ckpt)
with open(ckpt, "rb") as f:
    print(f)