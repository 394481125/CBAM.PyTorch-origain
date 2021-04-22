import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import traceback

class Logger(object):
    '''Save training process to log file with simple plot function
        使用简单的绘图功能将训练过程保存到日志文件
    '''
    def __init__(self, fpath,resume=False): 
        self.file = None
        self.resume = resume
        if os.path.isfile(fpath):
            if resume:
                self.file = open(fpath, 'a') 
            else:
                self.file = open(fpath, 'w')
        else:
            self.file = open(fpath, 'w')

    def append(self, target_str):
        if not isinstance(target_str, str):
            try:
                target_str = str(target_str)
            except:
                traceback.print_exc()
            else:
                print(target_str)
                self.file.write(target_str + '\n')
                self.file.flush()
        else:
            print(target_str)
            self.file.write(target_str + '\n')
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()