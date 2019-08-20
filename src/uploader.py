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

        self.start_iter = 1
        self.game_iter = 0
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

        shutil.copy2(self.iteration_path, self.drive_dir)
        shutil.copy2(self.games_path, self.drive_dir)
        if os.path.exists(self.best_path):
            shutil.copy2(self.best_path, self.drive_dir)

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
                self.game_iter = self.start_iter - 1
        return self.start_iter
    
    def upload_best_model(self, i):
        checkpoints_folder = self.drive_dir + 'checkpoints/'
        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)
        shutil.copyfile('models/best_checkpoint.pt', f'{checkpoints_folder}{i}.pt')
    
    def save_game(self, i, eps, moves):
        eps += 1
        with open(self.games_path, 'a+') as file:
            if i > self.game_iter:
                file.write(f'ITER :: {i}\n')
                self.game_iter += 1
            
            game = ' '.join([self.number_to_alphanum(m) for m in moves])
            file.write(f'EPS {eps}: {game}\n')
            
            if eps == self.num_eps:
                file.write('\n')
    
    def read_game(self, i, eps):
        iter_text = f'ITER :: {i}'
        eps_text = f'EPS {eps}:'
        
        with open(self.games_path, 'r') as file:
            text = file.read()
        
        try:
            iter_pos = text.index(iter_text)
            start_pos = text.index(eps_text, iter_pos)
            end_pos = text.index('\n', start_pos)
        except:
            print(f'Game of ITER {i} EPS {eps} not found!')
            return ''
        
        text = text[start_pos:end_pos]
        return text