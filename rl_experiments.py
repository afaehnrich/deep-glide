import sqlite3
import pickle
import torch
from datetime import datetime
from RL_wrapper_gym.DDPG import DDPGagent
import io
import bz2
import numpy as np

class RL_Experiment:

    def __init__(self, fn_db, cfg=None, name = '', time = None, create = True):
        self._open_db(fn_db)
        self.id = None
        self.live = False
        if time==None: time = datetime.now()
        if cfg==None: return
        elif create: self.id = self._create(time, cfg, name)



    def _open_db(self, fn_db):
        self.db = sqlite3.connect(fn_db)
        self.cur = self.db.cursor()
        #create Table if not Exists:
        self.cur.execute('CREATE TABLE IF NOT EXISTS experiments ( '
                '   id INTEGER PRIMARY KEY AUTOINCREMENT, '
                '   time TEXT, '
                '   cfg TEXT, '
                '   episodes INTEGER,'
                '   actor_dicts BLOB, '
                '   critic_dicts BLOB, '
                '   name TEXT UNIQUE, '
                '   comment TEXT '
                ');' )
        self.cur.execute('CREATE TABLE IF NOT EXISTS rewards ( '
                '   experiment_id INTEGER, '
                '   episode INTEGER, '
                '   agent INTEGER, '
                '   reward REAL, '
                '   final_state BLOB, '
                '   PRIMARY KEY (experiment_id, episode, agent)'
                ');' )
        self.cur.execute('CREATE TABLE IF NOT EXISTS route ( '
                '   experiment_id INTEGER, '
                '   episode INTEGER, '
                '   key TEXT, '
                '   val BLOB, '
                '   PRIMARY KEY (experiment_id, episode, key)'
                ');' )
        self.cur.execute('CREATE TABLE IF NOT EXISTS episode ( '
                '   experiment_id INTEGER, '
                '   episode INTEGER, '
                '   key TEXT, '
                '   val BLOB, '
                '   PRIMARY KEY (experiment_id, episode, key)'
                ');' )

        self.cur.execute(  'CREATE TABLE IF NOT EXISTS live_experiment ( '
                            '   id INTEGER);')
        self.db.commit()

    def _start_livesession(self):
        self.cur.execute('INSERT INTO live_experiment (id) VALUES(?);', (self.id,))
        self.db.commit()
        self.live = True


    def _create(self, time, cfg, name = "", comment = ""):
        dt_string = time.strftime("%d-%m-%Y %H.%M.%S")
        self.cur.execute(   'INSERT INTO experiments (time, cfg, episodes, comment) '
                            'VALUES (?, ?, 0, ?);', 
                            (dt_string, pickle.dumps(cfg), comment))
        id = self.cur.lastrowid    
        self.episodes = 0
        if name == '': name = 'experiment {}'.format(id)
        self.cur.execute( 'UPDATE experiments '
                          'SET name = ? '
                          'WHERE id = ?;', (name, id) )
        self.db.commit()
        return id

    def add_rewards(self, rewards, final_state):
        if self.live == False : self._start_livesession()

        '''        
        for agent, reward in enumerate(rewards):
            self.cur.execute( 'INSERT INTO rewards (experiment_id, episode, agent, reward, final_state) '
                              'VALUES (?, ?, ?, ?, ?)',
                            (self.id, self.episodes, agent, reward, pickle.dumps(final_state)) )
        '''
        data =[(self.id, self.episodes, agent, reward, pickle.dumps(final_state))
                for agent, reward in enumerate(rewards)]
        self.cur.executemany( 'INSERT INTO rewards (experiment_id, episode, agent, reward, final_state) '
                              'VALUES (?, ?, ?, ?, ?)', data)
        self.episodes += 1
        self.cur.execute( 'UPDATE experiments '
                          'SET episodes = ? '
                          'WHERE id = ?;', (self.episodes, self.id))
        self.db.commit()

    def load_rewards(self, agent_count):
        rewards=[]
        for agent_num in range(agent_count):
            self.cur.execute( 'SELECT (reward) '
                              'FROM rewards '
                              'WHERE experiment_id=? AND agent=? '
                              'ORDER BY episode',
                            (self.id, agent_num) )
            rew = self.cur.fetchall()
            rewards.append([r for (r,) in rew])
        return rewards    

    def load_episodes(self):
        self.cur.execute( 'SELECT episodes '
                          'FROM experiments '
                          'WHERE id = ? '
                          'LIMIT 1;', (self.id,))
        (self.episodes, ) = self.cur.fetchone()
        return self.episodes

    def _serialize_statedicts(self, torch_dicts):
        if torch_dicts == None: return torch_dicts
        bin_dicts = []
        for dict in torch_dicts:
            buffer = io.BytesIO()
            torch.save(dict,buffer)
            bin_dicts.append(buffer.getvalue())
        bin_dicts = pickle.dumps(bin_dicts)
        #bin_dicts = bz2.compress(bin_dicts)
        return bin_dicts

    def _deserialize_statedicts(self, bin_dicts):
        if bin_dicts == None: return bin_dicts
        bin_dicts = pickle.loads(bin_dicts)
        torch_dict =[]
        for dict in bin_dicts:
            buffer = io.BytesIO(dict)
            torch_dict.append(torch.load(buffer))
        return torch_dict



    def save_state_dicts(self, agents:DDPGagent):
        actor_dicts = self._serialize_statedicts([agent.actor.state_dict() for agent in agents])
        critic_dicts = self._serialize_statedicts([agent.critic.state_dict() for agent in agents])
        self.cur.execute(   'UPDATE experiments '
                            'SET actor_dicts=?, critic_dicts=? '
                            'WHERE id=? ',
                            (actor_dicts, critic_dicts, self.id))
        self.db.commit()

    def load_state_dicts(self, id):
        self.cur.execute(   'SELECT actor_dicts, critic_dicts '
                            'FROM experiments '
                            'WHERE id = ? '
                            'LIMIT 1', (id,))
        actor_dicts, critic_dicts = self.cur.fetchone()
        actor_dicts = self._deserialize_statedicts(actor_dicts)
        critic_dicts = self._deserialize_statedicts(critic_dicts)
        return actor_dicts, critic_dicts

    def load_by_id(self, id):
        self.cur.execute(   'SELECT id, cfg, episodes '
                            'FROM experiments '
                            'WHERE id = ? '
                            'LIMIT 1', (id,))
        return self._load()

    def load_by_name(self, name):
        self.cur.execute(   'SELECT id, cfg, episodes'
                            'FROM experiments '
                            'WHERE name = ? '
                            'LIMIT 1', (name,))
        return self._load()

    
    def _load(self):
        self.id, cfg, self.episodes = self.cur.fetchone()
        cfg = pickle.loads(cfg)
        actor_dicts, critic_dicts = self.load_state_dicts(self.id)
        rewards = self.load_rewards(len(actor_dicts))
        episodes = self.load_episodes()
        return cfg, actor_dicts, critic_dicts, rewards, episodes

    def save_routeitem(self, key, val):
        self.cur.execute('INSERT INTO route (experiment_id, episode, key, val) '
                         ' VALUES( ?,?,?,?);',
                         (self.id, self.episodes, key, pickle.dumps(val)))

    def save_route(self, data:dict):
        rows = [( self.id, self.episodes, key, pickle.dumps(val) ) 
                for (key, val) in data.items()]
        self.cur.executemany('INSERT INTO route (experiment_id, episode, key, val) '    
                         ' VALUES( ?,?,?,?);', rows)
        self.db.commit()       

    def save_episode(self, data:dict):
        rows = [( self.id, self.episodes, key, pickle.dumps(val) ) 
                for (key, val) in data.items()]
        self.cur.executemany('INSERT INTO episode (experiment_id, episode, key, val) '    
                         ' VALUES( ?,?,?,?);', rows)
        self.db.commit()                

    def load_routeitem(self, key):
        self.cur.execute('SELECT episode ,x, y FROM route '
                        ' '
                        ' ORDER BY episode')
        res = self.db.fetchall()
        routes={}
        for episode, x, y in res:
            routes.update({episode:{"x":pickle.loads(x), "y":pickle.loads(y)}})
        print(routes)
        return routes




    def close(self):
        self.cur.execute('DELETE FROM live_experiment WHERE id = ?;', (self.id,))
        self.db.commit()
        self.live = False
        self.id = False



