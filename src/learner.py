from collections import deque
from os import path, mkdir
import os
import threading
import time
import math
import numpy as np
import pickle
import concurrent.futures
import random
import shutil
from functools import reduce
from threading import Thread

import sys
sys.path.append('../build')
from library import MCTS, Gomoku, NeuralNetwork

from neural_network import NeuralNetWorkWrapper
from gomoku_cmd import GomokuCMD

def tuple_2d_to_numpy_2d(tuple_2d):
    # help function
    # convert type
    res = [None] * len(tuple_2d)
    for i, tuple_1d in enumerate(tuple_2d):
        res[i] = list(tuple_1d)
    return np.array(res)


class Learner():
    def __init__(self, config):
        # see config.py
        # gomoku
        self.n = config['n']
        self.n_in_row = config['n_in_row']
        self.gomoku_gui = GomokuCMD(config['n'], config['human_color'])
        self.action_size = config['action_size']

        # train
        self.num_iters = config['num_iters']
        self.num_eps = config['num_eps']
        self.num_train_threads = config['num_train_threads']
        self.check_freq = config['check_freq']
        self.num_contest = config['num_contest']
        self.dirichlet_alpha = config['dirichlet_alpha']
        self.temp = config['temp']
        self.update_threshold = config['update_threshold']
        self.num_explore = config['num_explore']

        self.examples_buffer_len = config['examples_buffer_len']
        self.examples_buffer = deque([], maxlen=self.examples_buffer_len[1])

        # mcts
        self.num_mcts_sims = config['num_mcts_sims']
        self.c_puct = config['c_puct']
        self.c_virtual_loss = config['c_virtual_loss']
        self.num_mcts_threads = config['num_mcts_threads']
        self.libtorch_use_gpu = config['libtorch_use_gpu']

        # neural network
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.nnet = NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'],
                                         config['num_channels'], config['n'], self.action_size, config['train_use_gpu'], self.libtorch_use_gpu)

        self.drive_path = config['drive_path']
        self.iteration_path = config['iteration_path']
        self.best_path = config['best_path']
        
        self.lr = config['lr']
        self.lr_schedule = config['lr_schedule']
        self.gomoku_gui.use_gpu = config['train_use_gpu']
        
        self.gomoku_gui.show_ram = config['show_ram']

        # start gui
        t = threading.Thread(target=self.gomoku_gui.loop)
        t.start()
    
    def update_lr(self, i):
        if not self.lr_schedule: return

        new_lr = self.lr
        for upper in self.lr_schedule:
            if i < upper:
                break
            new_lr = self.lr_schedule[upper]
        if new_lr != self.lr:
            self.lr = new_lr
            print('New Learning Rate:', self.lr)

    def update_examples_buffer_max_len(self, i):
        start, end, start_change, change_speed = self.examples_buffer_len
        change = max(0, (i - start_change) // change_speed)
        new_max_len = min(start + change, end)

        if new_max_len != self.examples_buffer.maxlen:
            self.examples_buffer = deque(self.examples_buffer, maxlen=new_max_len)
            print('New Examples Max Length:', new_max_len)

    def learn(self):
        # train the model by self play        

        if path.exists('models/checkpoint.example'):
            print("Loading checkpoint...")
            self.nnet.load_model()
            self.load_samples()
        else:
            # save torchscript
            self.nnet.save_model()
            self.nnet.save_model('models', "best_checkpoint")
        
        start_iter = 1
        if path.exists(self.iteration_path):
            with open(self.iteration_path, 'r') as file:
                start_iter = int(file.read())
        
        upload_thread = Thread(target=self.upload_models)

        for i in range(start_iter, self.num_iters + 1):
            # self play in parallel

            self.update_lr(i)
            self.update_examples_buffer_max_len(i)
            
            libtorch = NeuralNetwork('./models/checkpoint.pt',
                                     self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads)
            
            if i > start_iter and not upload_thread.is_alive():
                print(f'Uploading models to Drive...\n{self.gomoku_gui.get_time()}\n')
                upload_thread = Thread(target=self.upload_models)
                upload_thread.start()
            
            self.gomoku_gui.iteration = i
            print('ITER ::', i)

            itr_examples = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_train_threads) as executor:
                futures = [executor.submit(self.self_play, 1 if i % 2 else -1, libtorch, k == 1) for k in range(1, self.num_eps + 1)]
                for k, f in enumerate(futures):
                    examples = f.result()
                    itr_examples += examples

                    # decrease libtorch batch size
                    remain = min(len(futures) - (k + 1), self.num_train_threads)
                    libtorch.set_batch_size(max(remain * self.num_mcts_threads, 1))
                    print(f'EPS: {k+1} - EXAMPLES: {len(examples)} ({self.gomoku_gui.get_time()})')

            # release gpu memory
            del libtorch

            # prepare train data
            self.examples_buffer.append(itr_examples)

            print('Examples Size:', len(self.examples_buffer), 'iters')

            train_data = reduce(lambda a, b : a + b, self.examples_buffer)
            random.shuffle(train_data)

            print('Train Data Size:', len(train_data))

            # train neural network
            epochs = self.epochs * (len(itr_examples) + self.batch_size - 1) // self.batch_size

            print(f'Training {int(epochs)} epochs...')

            self.nnet.train(train_data, self.batch_size, int(epochs))
            self.nnet.save_model()
            self.save_samples()

            # compare performance
            if i % self.check_freq == 0:
                print('Pitting aganst best model...')
                libtorch_current = NeuralNetwork('./models/checkpoint.pt',
                                         self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2)
                libtorch_best = NeuralNetwork('./models/best_checkpoint.pt',
                                              self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2)

                one_won, two_won, draws = self.contest(libtorch_current, libtorch_best, self.num_contest)
                text = f'\nNEW/BEST WINS : {one_won} / {two_won} | DRAWS : {draws}\n'

                if one_won + two_won > 0 and float(one_won) / (one_won + two_won) >= self.update_threshold:
                    text += 'ACCEPTING NEW MODEL'
                    self.nnet.save_model('models', "best_checkpoint")
                else:
                    text += 'REJECTING NEW MODEL'
                
                text += '\n\n'
                print(text)
                text = f'ITER :: {i}{text}'

                with open(self.best_path, 'a+') as file:
                    file.write(text)

                # release gpu memory
                del libtorch_current
                del libtorch_best
            
            with open(self.iteration_path, 'w+') as file:
                file.write(str(i+1))
                
    def upload_models(self):
        shutil.make_archive(self.drive_path + 'models', 'zip', 'models')
        shutil.copy2(self.iteration_path, self.drive_path)
        if path.exists(self.best_path):
            shutil.copy2(self.best_path, self.drive_path)
        print('Uploaded models to Drive!', self.gomoku_gui.get_time(), '\n')
        sys.stdout.flush()

    def self_play(self, first_color, libtorch, show):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.
        """
        train_examples = []

        player1 = MCTS(libtorch, self.num_mcts_threads, self.c_puct,
                    self.num_mcts_sims, self.c_virtual_loss, self.action_size)
        player2 = MCTS(libtorch, self.num_mcts_threads, self.c_puct,
            self.num_mcts_sims, self.c_virtual_loss, self.action_size)
        players = [player2, None, player1]
        player_index = 1

        gomoku = Gomoku(self.n, self.n_in_row, first_color)

        if show:
            self.gomoku_gui.reset_status()

        episode_step = 0
        while True:
            episode_step += 1
            player = players[player_index + 1]

            # get action prob
            if episode_step <= self.num_explore:
                prob = np.array(list(player.get_action_probs(gomoku, self.temp)))
            else:
                prob = np.array(list(player.get_action_probs(gomoku, 0)))

            # generate sample
            board = tuple_2d_to_numpy_2d(gomoku.get_board())
            last_action = gomoku.get_last_move()
            cur_player = gomoku.get_current_color()

            sym = self.get_symmetries(board, prob, last_action)
            for b, p, a in sym:
                train_examples.append([b, a, cur_player, p])

            # dirichlet noise
            legal_moves = list(gomoku.get_legal_moves())
            noise = 0.1 * np.random.dirichlet(self.dirichlet_alpha * np.ones(np.count_nonzero(legal_moves)))

            prob = 0.9 * prob
            j = 0
            for i in range(len(prob)):
                if legal_moves[i] == 1:
                    prob[i] += noise[j]
                    j += 1
            prob /= np.sum(prob)

            # execute move
            action = np.random.choice(len(prob), p=prob)

            if show:
                self.gomoku_gui.execute_move(cur_player, action)
            gomoku.execute_move(action)
            player1.update_with_move(action)
            player2.update_with_move(action)

            # next player
            player_index = -player_index

            # is ended
            ended, winner = gomoku.get_game_status()
            if ended == 1:
                # b, last_action, cur_player, p, v
                return [(x[0], x[1], x[2], x[3], x[2] * winner) for x in train_examples]

    def contest(self, network1, network2, num_contest):
        """compare new and old model
           Args: player1, player2 is neural network
           Return: one_won, two_won, draws
        """
        one_won, two_won, draws = 0, 0, 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_train_threads) as executor:
            futures = [executor.submit(\
                self._contest, network1, network2, 1 if k <= num_contest // 2 else -1, k == 1) for k in range(1, num_contest + 1)]
            for k, f in enumerate(futures):
                winner = f.result()
                print(f'GAME {k + 1}: ', end='')
                if winner == 1:
                    one_won += 1
                    print('WON ', end='')
                elif winner == -1:
                    two_won += 1
                    print('LOST', end='')
                else:
                    draws += 1
                    print('DRAW', end='')
                print(f' - NEW/BEST WINS : {one_won} / {two_won} | DRAWS : {draws}')

        return one_won, two_won, draws

    def _contest(self, network1, network2, first_player, show):
        # create MCTS
        player1 = MCTS(network1, self.num_mcts_threads, self.c_puct,
            self.num_mcts_sims, self.c_virtual_loss, self.action_size)
        player2 = MCTS(network2, self.num_mcts_threads, self.c_puct,
                    self.num_mcts_sims, self.c_virtual_loss, self.action_size)

        # prepare
        players = [player2, None, player1]
        player_index = first_player
        gomoku = Gomoku(self.n, self.n_in_row, first_player)
        if show:
            self.gomoku_gui.reset_status()

        # play
        while True:
            player = players[player_index + 1]

            # select best move
            prob = player.get_action_probs(gomoku)
            best_move = int(np.argmax(np.array(list(prob))))

            # execute move
            gomoku.execute_move(best_move)
            if show:
                self.gomoku_gui.execute_move(player_index, best_move)

            # check game status
            ended, winner = gomoku.get_game_status()
            if ended == 1:
                return winner

            # update search tree
            player1.update_with_move(best_move)
            player2.update_with_move(best_move)

            # next player
            player_index = -player_index

    def get_symmetries(self, board, pi, last_action):
        # mirror, rotational
        assert(len(pi) == self.action_size)  # 1 for pass

        pi_board = np.reshape(pi, (self.n, self.n))
        last_action_board = np.zeros((self.n, self.n))
        last_action_board[last_action // self.n][last_action % self.n] = 1
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                newAction = np.rot90(last_action_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                    newAction = np.fliplr(last_action_board)
                l += [(newB, newPi.ravel(), np.argmax(newAction) if last_action != -1 else -1)]
        return l

    def play_with_human(self, human_first=True, checkpoint_name="best_checkpoint"):
        # load best model
        libtorch_best = NeuralNetwork('./models/best_checkpoint.pt', self.libtorch_use_gpu, 12)
        mcts_best = MCTS(libtorch_best, self.num_mcts_threads * 3, \
             self.c_puct, self.num_mcts_sims * 6, self.c_virtual_loss, self.action_size)

        # create gomoku game
        human_color = self.gomoku_gui.get_human_color()
        gomoku = Gomoku(self.n, self.n_in_row, human_color if human_first else -human_color)

        players = ["alpha", None, "human"] if human_color == 1 else ["human", None, "alpha"]
        player_index = human_color if human_first else -human_color

        self.gomoku_gui.reset_status()
        self.gomoku_gui.output_util = False

        while True:
            player = players[player_index + 1]

            # select move
            if player == "alpha":
                prob = mcts_best.get_action_probs(gomoku)
                best_move = int(np.argmax(np.array(list(prob))))
                self.gomoku_gui.execute_move(player_index, best_move)
            else:
                self.gomoku_gui.set_is_human(True)
                # wait human action
                while self.gomoku_gui.get_is_human():
                    time.sleep(0.1)
                best_move = self.gomoku_gui.get_human_move()

            # execute move
            gomoku.execute_move(best_move)

            # check game status
            ended, winner = gomoku.get_game_status()
            if ended == 1:
                break

            # update tree search
            mcts_best.update_with_move(best_move)

            # next player
            player_index = -player_index

        print("HUMAN WIN" if winner == human_color else "ALPHA ZERO WIN")

    def load_samples(self, folder="models", filename="checkpoint.example"):
        """load self.examples_buffer
        """

        filepath = path.join(folder, filename)
        with open(filepath, 'rb') as f:
            self.examples_buffer = deque(pickle.load(f), self.examples_buffer.maxlen)

    def save_samples(self, folder="models", filename="checkpoint.example"):
        """save self.examples_buffer
        """

        if not path.exists(folder):
            mkdir(folder)

        filepath = path.join(folder, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.examples_buffer, f, -1)
