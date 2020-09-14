from easydict import EasyDict

config = EasyDict()

config.training = True
config.gpu = False
config.soft_max = True
config.decay_lr = True
config.max_memory = 170
config.memory_size = 10
config.strategy = "m10"
config.player_name = "player_" + config.strategy
config.learning_rate = 1e-5
config.belief_learning_rate = 1e-4
config.start_iteration = 0
config.end_iteration = 1e5
config.save_iteration = (config.end_iteration - config.start_iteration) / 20
config.print_iteration = 1000
config.update_q_target_frequency = 100
config.batch_size = 1
config.max_batch = 10000
config.decay_step = (config.end_iteration - config.start_iteration) / 2
config.decay_rate = 0.5
config.switch_iters = 5e4
config.epsilon_decay = config.switch_iters / 3
config.epsilon_start = 0.95
config.epsilon_min = 0.01
config.acting_boltzman_beta = 20

config.num_deck = 4
config.num_obv = 8
config.num_memory = 10
config.num_action = 2
config.num_belief = 10
config.step_reward = 0
config.fail_reward = -1
config.success_reward = 1

config.in_x_dim = config.num_memory
config.out_x_dim = config.in_x_dim
config.hidden_dim = config.out_x_dim
config.gamma = 0.9

config.num_layer = 3
config.scale_factor = 32
config.q_input_dim = config.num_obv + config.memory_size + config.hidden_dim

