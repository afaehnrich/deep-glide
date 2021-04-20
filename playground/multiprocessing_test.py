import multiprocessing as mp
from multiprocessing import Pool
from mayavi import mlab
import numpy as np
from pyface.api import GUI
import time
from multiprocessing import Process, Queue


class Load_Gen:
    proc_id=0
    def __init__(self,s):
        self.proc_id = s

    def generate_load(self, sleep_time):
        self.proc_id +=1
        pid = self.proc_id
        interval = time.time() + sleep_time
        print(pid,':load from', time.time(),'to ',interval)
        # generates some getCpuLoad for interval seconds
        pr = 213123  # generates some load
        while (time.time() < interval):
            pr * pr
            pr = pr + 1
            #print(pid,' calculated ',pr)
        return pr

class mp_handler:
    lg: Load_Gen

    def __init__(self, s):
        self.lg = Load_Gen(s)
    def connect(self, s, q_send: Queue, q_rec:Queue):
        #self.lg = Load_Gen(s)
        while True:
            p = q_send.get()
            if p[0] == 'quit':
                print('quitting proces')
                break
            method, params = p
            res = method(self.lg, *params)
            q_rec.put(res)

def main():
    m= mlab.figure()
    np.random.seed()
    p = mlab.plot3d(np.random.uniform(-100,100,10), np.random.uniform(-100,100,10), np.random.uniform(-100,100,10), tube_radius=20, color=(0,0,1))
    mlab.draw()
    GUI().process_events()
    res = None
    t=time.time()
    mph = mp_handler(20)
    q_send = Queue()
    q_rec = Queue()
    p = Process(target=mph.connect, args=(20, q_send, q_rec))
    p.start()
    q_send.put((Load_Gen.generate_load,(2,)))
    while True:
        if not q_rec.empty():
            res = q_rec.get()
            print ('res of proc:',res)
            print('sending data to proc')
            q_send.put((Load_Gen.generate_load,(2,)))
            if res >40: q_send.put(('quit',))
    while True:
            if res is None:
                print('starting pool process')
                res = pool.apply_async(lg.generate_load, (lg, 6))
            if res.ready(): 
                print ('res of pool:',res.get())
                res = None
            time.sleep(0.01)      
            GUI().process_events()
            

if __name__ == '__main__':
    main()
    input()
    exit()
