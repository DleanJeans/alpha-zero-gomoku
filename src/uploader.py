import shutil
import sys
import os
from threading import Thread
import time
import datetime

class Uploader:
    def __init__(self, config):
        self.upload_thread = Thread(target=self.upload_models)
        self.cwd = f'{os.getcwd()}/'
        self.drive_dir = self.cwd + config.drive_dir
        self.models_dir = self.cwd + 'models/'
        self.iteration_path = self.models_dir + config.iteration_path
        self.best_path = self.models_dir + config.best_path

        print()
        print('Current Directory:', self.cwd)
        print('Drive Directory:  ', self.drive_dir)
        print('Models Directory: ', self.models_dir)
        print('iteration.txt:    ', self.iteration_path)
        print('best.txt:         ', self.best_path)
        print()
    
    def start_thread_uploading(self):
        if self.upload_thread.is_alive():
            print('Still uploading... Aborting this upload request...\n')
            return

        self.start_time = time.time()
        print(f'Uploading models to Drive... {self.get_time()}\n')
        self.upload_thread = Thread(target=self.upload_models)
        self.upload_thread.start()

    def upload_models(self):
        if not os.path.exists(self.drive_dir):
            os.makedirs(self.drive_dir)

        shutil.make_archive('models', 'zip', 'models')
        shutil.copy2('models.zip', self.drive_dir)
        
        shutil.copy2(self.iteration_path, self.drive_dir)

        if os.path.exists(self.best_path):
            shutil.copy2(self.best_path, self.drive_dir)
        
        print(f'Uploaded models to Drive! At {self.get_time()} | Took {self.get_time_elapsed()}s\n')
        sys.stdout.flush()
    
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
                return int(file.read())
        return 1
    
    def upload_best_model(self, i):
        checkpoints_folder = self.drive_dir + 'checkpoints/'
        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)
        shutil.copyfile('models/best_checkpoint.pt', f'{checkpoints_folder}{i}.pt')