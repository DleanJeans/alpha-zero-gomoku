import shutil
import sys
import os
from threading import Thread
import time
import datetime

class Uploader:
    def __init__(self, config):
        self.upload_thread = Thread(target=self.upload_models)
        self.upload_now = config.upload_now

        self.cwd = f'{os.getcwd()}/'
        self.drive_dir = self.cwd + config.drive_dir
        self.models_dir = self.cwd + 'models/'
        self.iteration_path = self.models_dir + config.iteration_path
        self.best_path = self.models_dir + config.best_path
        self.games_path = self.models_dir + config.games_path
        self.contest_games_path = self.models_dir + 'contest_games.txt'

        self.files_to_save = [self.iteration_path, self.best_path, self.games_path, self.contest_games_path]

        self.start_iter = 1
        self.num_eps = config.num_eps
    
    def request_upload(self, i = 0):
        if self.should_upload(i):
            self.start_upload_thread()
        elif self.upload_thread.is_alive():
            print('Still uploading... Aborting this upload request...\n')

    def should_upload(self, i):
        return not self.upload_thread.is_alive() and self.upload_now or i > self.start_iter

    def start_upload_thread(self):
        self.upload_thread = Thread(target=self.upload_models)
        self.upload_thread.start()
        
    def get_time_log(self):
        return f'At {self.get_time()} | Taken {self.get_time_elapsed()}s'
        
    def upload_models(self):
        print(f'Uploading models to Drive... {self.get_time()}\n')
        self.start_time = time.time()

        if not os.path.exists(self.drive_dir):
            os.makedirs(self.drive_dir)

        for file in self.files_to_save:
            if os.path.exists(file):
                shutil.copy2(file, self.drive_dir)

        shutil.make_archive('models', 'zip', 'models')
        
        print(f'Archive created!', self.get_time_log(), '\n'); sys.stdout.flush()
        self.start_time = time.time()

        shutil.copy2('models.zip', self.drive_dir)

        print(f'Models uploaded to Drive!', self.get_time_log(), '\n'); sys.stdout.flush()
    
    def get_time_elapsed(self):
        now = time.time()
        elapsed = round(now - self.start_time)
        delta = datetime.timedelta(seconds = elapsed)
        delta = str(delta).split(':', 1)[1].replace(':', 'm')
        return delta

    def save_iteration(self, i):
        with open(self.iteration_path, 'w+') as file:
            file.write(str(i))
    
    def add_best_log(self, text):
        with open(self.best_path, 'a+') as file:
            file.write(text)
    
    def read_iteration(self):
        if os.path.exists(self.iteration_path):
            with open(self.iteration_path, 'r') as file:
                self.start_iter = int(file.read())
        return self.start_iter
    
    def upload_best_model(self, i):
        checkpoints_folder = self.drive_dir + 'checkpoints/'
        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)
        shutil.copyfile('models/best_checkpoint.pt', f'{checkpoints_folder}{i}.pt')
    
    def reset_game_history(self):
        history_on_drive = self.drive_dir + 'games.txt'
        if os.path.exists(history_on_drive):
            shutil.copy2(history_on_drive, self.models_dir)

    def save_game(self, i, eps, moves, contest=False):
        file_path = self.contest_games_path if contest else self.games_path
        with open(file_path, 'a+') as file:
            if eps == 1:
                file.write(f'ITER :: {i}\n')
            
            game = ' '.join([self.number_to_alphanum(m) for m in moves])
            file.write(f'EPS {eps} - {game}\n')
            
            if eps == self.num_eps:
                file.write('\n')
    
    def read_game(self, i, eps, contest=False):
        iter_text = f'ITER :: {i}'
        eps_text = f'EPS {eps}:'

        file_path = self.contest_games_path if contest else self.games_path
        
        with open(file_path, 'r') as file:
            text = file.read()
            if '-' in text:
                text = text.split('-')[1].strip()
        
        try:
            iter_pos = text.index(iter_text)
            start_pos = text.index(eps_text, iter_pos)
            end_pos = text.index('\n', start_pos)
        except:
            print(f'Game of ITER {i} EPS {eps} not found!')
            return ''
        
        text = text[start_pos:end_pos]
        return text