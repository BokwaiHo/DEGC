import os
import sys
import shutil
import glob


class Logger(object):
    def __init__(self, logname, log_save_name, now, log_files='/data/yuening/graph_incremental_logs/log-files/version2'):
        self.terminal = sys.stdout
        self.file = None

        # path = os.path.join(log_files, logname, now)
        path = logname
        if not os.path.exists(path):
            os.makedirs(path)

        model_path = os.path.join(log_files, log_save_name, now)
        os.makedirs(model_path)


    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError
