import jsbgym_flex
from jsbgym_flex.environment import JsbSimEnv
from jsbgym_flex.tasks import *
import toml
from datetime import datetime
import RL_wrapper_gym

cfg = toml.load('gym-jsbsim-cfg.toml')

def simCycle(name, env, outer, inner):
    print ('Simulation {}: {}x{} steps'.format(name, outer, inner),end='\r') 
    before = datetime.now()
    for o in range(outer):
        env.reset()  
        for _ in range(inner):
            action=env.action_space.sample()
            env.step(action)
            #if o==2: env.render()
    after = datetime.now()
    dt = (after-before)
    print ('Simulation {}: {}x{} steps: {}.{}s'.format(name, outer, inner, dt.seconds, dt.microseconds) )

env = jsbgym_flex.environment.JsbSimEnv(cfg = cfg, task_type = AFHeadingControlTask, shaping = Shaping.STANDARD)
name = 'JSBSIM' 
simCycle(name, env, 1, 100)
simCycle(name, env, 10, 100)
#simCycle(name, env, 100, 100)
env.close()
