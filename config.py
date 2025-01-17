class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config = {
    # gomoku
    'n': 19,                                    # board size
    'n_in_row': 5,                              # n in row

    # mcts
    'libtorch_use_gpu' : True,                  # libtorch use cuda
    'num_mcts_threads': 4,                      # mcts threads number
    'num_mcts_sims': 1600,                      # mcts simulation times
    'c_puct': 5,                                # puct coeff
    'c_virtual_loss': 3,                        # virtual loss coeff
    'contest_mcts': 4,
    'loop_from_center': False,
    'alternate_center_loop': False,

    # neural_network
    'train_use_gpu' : True,                     # train neural network using cuda
    'lr': 1e-3,                                 # learning rate
    'lr_schedule': {
        0: 1e-2,
        400: 1e-3,
        600: 1e-4
    },
    'l2': 0.0001,                               # L2
    'num_channels': 256,                        # convolution neural network channel size
    'num_layers' : 4,                           # residual layer number
    'epochs': 1.5,                              # train epochs
    'batch_size': 512,                          # batch size
    'use_radam': False,

    # train
    'num_iters': 10000,                         # train iterations
    'num_eps': 10,                              # self play times in per iter
    'num_train_threads': 10,                    # self play in parallel
    'num_explore': 5,                           # explore step in a game
    'temp': 1,                                  # temperature
    'dirichlet_alpha': 0.3,                     # action noise in self play games
    'update_threshold': 0.55,                   # update model threshold
    'num_contest': 10,                          # new/old model compare times
    'check_freq': 20,                           # test model frequency
    'examples_buffer_len': (4, 20, 5, 2),       # (min, max, start_change_iter, n_iter_to_change)
    'prob_multiplier': 0.9,
    'second_prob_multiplier': 0.9,

    # test
    'human_color': 1,                            # human player's color
    'show_ram': False,

    # checkpoint uploading
    'drive_dir': 'drive/My Drive/Colab Notebooks/alpha-zero-caro/',
    'iteration_path': 'iteration.txt',
    'best_path': 'best.txt',
    'games_path': 'games.txt',
    'upload_now': False,
}

config = DotDict(config)

# action size
config.action_size = config.n ** 2
