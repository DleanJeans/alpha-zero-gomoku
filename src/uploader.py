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
        self.upload_now = config.upload_now
        self.start_iter = 1
    
    def request_upload(self, i):
        if self.should_upload(i):
            self.start_upload_thread()
        else:
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
        return self.start_iter
    
    def upload_best_model(self, i):
        checkpoints_folder = self.drive_dir + 'checkpoints/'
        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)
        shutil.copyfile('models/best_checkpoint.pt', f'{checkpoints_folder}{i}.pt')