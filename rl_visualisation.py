import sqlite3
import pickle
import torch
from datetime import datetime
from RL_wrapper_gym.DDPG import DDPGagent
import io
import bz2

class VisualisationDB:
     
    def __init__(self, fn_db):
        self._open_db(fn_db)
        
    def _open_db(self, fn_db):
        self.db = sqlite3.connect(fn_db)
        self.cur = self.db.cursor()
        #create Table if not Exists:
        self.cur.execute('CREATE TABLE IF NOT EXISTS rewards ( '
                '   episode INTEGER, '
                '   agent INTEGER, '
                '   reward REAL'
                ');' )
        self.db.commit()

