# from tf_models import KerasConvLSTM 

def ppo_policy_params_1():
    params = {
        "ppo_agent_policy": {
            "clip_param": 0.3,
            "entropy_coeff": 0.025,
            "entropy_coeff_schedule": None,
            "gamma": 0.998,
            "grad_clip": 10.0,
            "kl_coeff": 0.0,
            "kl_target": 0.01,
            "lambda": 0.98,
            "lr": 0.0003,
            "lr_schedule": None,
            "model": {
                "custom_model": "keras_conv_lstm",
                "custom_options": {
                    "fc_dim": 128,
                    "idx_emb_dim": 4,
                    "input_emb_vocab": 100,
                    "lstm_cell_size": 128,
                    "num_conv": 2,
                    "num_fc": 2, },
                "max_seq_len": 25, },
            "use_gae": True,
            "vf_clip_param": 50.0,
            "vf_loss_coeff": 0.05,
            "vf_share_layers": False,
        },
        "ppo_planner_policy": {
            "clip_param": 0.3,
            "entropy_coeff": 0.025,
            "entropy_coeff_schedule": None,
            "gamma": 0.998,
            "grad_clip": 10.0,
            "kl_coeff": 0.0,
            "kl_target": 0.01,
            "lambda": 0.98,
            "lr": 0.0003,
            "lr_schedule": None,
            "model": {
                "custom_model": "random",
                "custom_options": {},
                "max_seq_len": 25, },
            "use_gae": True,
            "vf_clip_param": 50.0,
            "vf_loss_coeff": 0.05,
            "vf_share_layers": False,
        }
    }

    return params

