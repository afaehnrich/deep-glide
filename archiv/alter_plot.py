
class PlotClass:

    def __init__(self):
        #plt.figure()
        #p1 = plt.subplot(2,1,1)
        #p2 = plt.subplot(2,1,2)
        #plt.plot(0,0,'.')
        #plt.figure()
        #plt.figure()
        #plt.ion()
        #plt.show()   
        #plt.pause(0.001)
        self.new_episode()
        #self.avg_rewards = []
        self.max_travel_dist = 0.15
        self.max_travel_dist_m = 10


    def new_episode(self):
        self.data = defaultdict(list)
        self.data['start_x'] = 0
        self.data['start_y'] = 0
        self.data['start_heading'] = env.get_property('initial_heading_rad')
        start_lat = data.cfg.get('environment').get('initial_state').get('initial_latitude_geod_deg')
        start_lon = data.cfg.get('environment').get('initial_state').get('initial_longitude_geoc_deg')
        self.start_gc = (start_lat, start_lon)
        #target_lat= env.get_property('target_lat_geod_deg') #y
        #target_lon= env.get_property('target_lng_geoc_deg') #x
        #self.target_gc = (target_lat,target_lon)
        #dx = distance.distance(self.start_gc, (start_lat,target_lon)).m
        #if start_lon > target_lon: dx = -dx
        #dy = distance.distance(self.start_gc, (target_lat,start_lon)).m
        #if start_lat > target_lat: dy = -dy
        #self.target_m = (dy,dx)
        #self.data['target_x'] = dx
        #self.data['target_y'] = dy
        #self.data['target_heading'] = env.get_property('target_heading') 
        #self.data['max_x'] = 0
        #self.data['max_y'] = 0
        #self.route_gc=[]
        #self.route_m=[]

    def add_data(self):
        
        for prop in data.cfg['environment']['plot_perstep']:
            print('saving ',prop)

    def add_data2(self):
        gc_lat = env.get_property('lat_geod_deg')
        gc_lon = env.get_property('lng_geoc_deg')
        self.route_gc.append((gc_lat,gc_lon))
        dy = env.get_property('dist_travel_lat_m')
        dx = env.get_property('dist_travel_lon_m')
        #self.max_travel_dist_m = max(self.max_travel_dist_m, dx)
        #self.max_travel_dist_m = max(self.max_travel_dist_m, dy)
        if gc_lat < self.start_gc[0]: dy = -dy
        if gc_lon < self.start_gc[1]: dx = -dx
        #self.route_m.append((dy,dx))
        self.data['x'].append(dx)
        self.data['y'].append(dy)
        self.data['max_y'] = max(self.data['max_y'], dx)
        self.data['max_y'] = max(self.data['max_y'], dy)

    def plot_route(self):
        plt.figure(1)
        plt.clf()
        plt.xlim(self.start_gc[0]-self.max_travel_dist, self.start_gc[0]+self.max_travel_dist)
        plt.ylim(self.start_gc[1]-self.max_travel_dist, self.start_gc[1]+self.max_travel_dist)
        plt.plot(*zip(*self.route_gc))
        plt.plot(self.start_gc[0], self.start_gc[1],'g.')
        plt.plot(self.target_gc[0], self.target_gc[1],'r.')

        plt.show()
        plt.figure(2)
        plt.clf()
        plt.xlim(-self.max_travel_dist_m, self.max_travel_dist_m)
        plt.ylim(-self.max_travel_dist_m, self.max_travel_dist_m)
        plt.plot(self.data['x'], self.data['y'],'b')
        plt.plot(0,0,'go')
        plt.plot(self.data['target_x'], self.data['target_y'],'ro')
        plt.show()
        plt.pause(0.001)  

    def plot_reward(self, rewards):
        
        #self.avg_rewards.append(np.mean(self.rewards[-10:]))
        plt.figure(3)
        plt.clf()
        for r in rewards:
            plt.plot(r)
        #plt.plot(self.avg_rewards)
        plt.show()
        plt.pause(0.001)

    def make_permanent(self):
        plt.ioff()
        plt.show()