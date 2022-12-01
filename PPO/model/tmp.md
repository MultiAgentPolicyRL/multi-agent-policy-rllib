**Eager Mode (w/o predict, nor mean), with sqeeze**:

2022-12-01 11:46:56,014 Function reset_memory Took 0.0000 seconds
2022-12-01 11:47:16,792 Function batch Took 20.7758 seconds
2022-12-01 11:47:16,797 Function get_memory Took 0.0050 seconds
2022-12-01 11:47:16,989      Values and next_values required 0.19156479835510254s
2022-12-01 11:47:17,018      Gaes required 0.0285341739654541s
2022-12-01 11:47:17,407      Data prep required: 0.3891177177429199s
2022-12-01 11:48:46,079      Fit Actor Network required 88.67226791381836s
2022-12-01 11:48:46,079         Actor loss: -0.0014942341949790716
2022-12-01 11:48:46,080     Prep 2 required 7.581710815429688e-05
2022-12-01 11:49:07,976      Fit Critic Network required 21.896177768707275s
2022-12-01 11:49:07,976         Critic loss: 2.247931480407715

**No Eager, no predict**

2022-12-01 11:50:25,737 Function reset_memory Took 0.0000 seconds
2022-12-01 11:51:28,350 Function batch Took 62.6113 seconds
2022-12-01 11:51:28,356 Function get_memory Took 0.0052 seconds
2022-12-01 11:51:28,594      Values and next_values required 0.2386770248413086s
2022-12-01 11:51:28,625      Gaes required 0.030011653900146484s
2022-12-01 11:51:29,047      Data prep required: 0.42260026931762695s
2022-12-01 11:51:37,732      Fit Actor Network required 8.684509754180908s
2022-12-01 11:51:37,732         Actor loss: 0.0002220274182036519
2022-12-01 11:51:37,732     Prep 2 required 0.00010991096496582031
2022-12-01 11:51:40,123      Fit Critic Network required 2.390604257583618s
2022-12-01 11:51:40,123         Critic loss: 2.247931480407715

**No Eager, no predict, np.sqeeze**

2022-12-01 11:53:03,261 Function reset_memory Took 0.0000 seconds
2022-12-01 11:53:23,834 Function batch Took 20.5700 seconds
2022-12-01 11:53:23,839 Function get_memory Took 0.0050 seconds
2022-12-01 11:53:24,089      Values and next_values required 0.2501680850982666s
2022-12-01 11:53:24,119      Gaes required 0.029522180557250977s
2022-12-01 11:53:24,530      Data prep required: 0.41078972816467285s
2022-12-01 11:53:33,429      Fit Actor Network required 8.899378538131714s
2022-12-01 11:53:33,430         Actor loss: -0.0014954011421650648
2022-12-01 11:53:33,430     Prep 2 required 0.0001220703125
2022-12-01 11:53:36,040      Fit Critic Network required 2.610748767852783s
2022-12-01 11:53:36,041         Critic loss: 2.247931480407715

**No Eager, no predict, np.sqeeze, action/np.sum(action)**

2022-12-01 11:54:42,241 Function reset_memory Took 0.0000 seconds
2022-12-01 11:55:02,558 Function batch Took 20.3143 seconds
2022-12-01 11:55:02,563 Function get_memory Took 0.0055 seconds
2022-12-01 11:55:02,797      Values and next_values required 0.23387575149536133s
2022-12-01 11:55:02,823      Gaes required 0.025678157806396484s
2022-12-01 11:55:03,228      Data prep required: 0.4047391414642334s
2022-12-01 11:55:11,157      Fit Actor Network required 7.928555965423584s
2022-12-01 11:55:11,157         Actor loss: 0.0002220274182036519
2022-12-01 11:55:11,157     Prep 2 required 0.00010919570922851562
2022-12-01 11:55:13,491      Fit Critic Network required 2.3338396549224854s
2022-12-01 11:55:13,491         Critic loss: 2.247931480407715
Trained step 0 in 32.26854062080383 seconds