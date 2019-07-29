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

    # neural_network
    'train_use_gpu' : True,                     # train neural network using cuda
    'lr': 1e-3,                                 # learning rate
    'lr_map': {
        0: 1e-2,
        400: 1e-3,
        600: 1e-4
    },
    'l2': 0.0001,                               # L2
    'num_channels': 256,                        # convolution neural network channel size
    'num_layers' : 4,                           # residual layer number
    'epochs': 1.5,                              # train epochs
    'batch_size': 512,                          # batch size

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

    # test
    'human_color': 1,                            # human player's color

    'drive_path': 'drive/My Drive/Colab Notebooks/alpha-zero-caro/',
    'iteration_path': 'models/iteration.txt',
    'best_path': 'models/best.txt'
}

# action size
config['action_size'] = config['n'] ** 2
