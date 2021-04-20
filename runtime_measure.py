import time

class Runtime_Measurement:
    
    def __init__(self):
        self.reset()

    def reset(self):
        self._avg_time = 0
        self._total_time = 0
        self._count = 0
        self._running = False

    def start_sample(self):
        if self._running: raise Exception('RuntimeMeasurementError', 'Trying to start timer, but timer already running')
        self._running = True
        self._t1 = time.time()
            
    def stop_sample(self):
        if not self._running: raise Exception('RuntimeMeasurementError', 'Trying to stop timer, but timer was not running')
        self._running = False
        t2 = time.time()
        self._dt = t2-self._t1
        self._avg_time = (self._dt + self._avg_time*self._count) / (self._count+1)
        self._count +=1
        self._total_time += self._dt
        
    def average(self):
        if self._running: raise Exception('RuntimeMeasurementError', 'Timer still running')
        return self._avg_time
    
    def total(self):
        if self._running: raise Exception('RuntimeMeasurementError', 'Timer still running')    
        return self._total_time
        
    def last(self):
        if self._running: raise Exception('RuntimeMeasurementError', 'Timer still running')
        return self._dt