# myapp.py

from random import random
from collections import defaultdict
from bokeh.layouts import row, column
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.io import show
from bokeh.models import CustomJS, MultiChoice, TextAreaInput, Legend, LegendItem, Line, RangeTool, Range1d, Select, Circle, Slider, Button
from bokeh.palettes import Category20_20
from bokeh.palettes import Viridis6
from typing import List, Set, Dict, Tuple, Optional

from bokeh.models.markers import marker_types
from bokeh.models import ColumnDataSource, HoverTool
from collections.abc import Iterable
import itertools
import math 
import pickle

lineDashes =['solid', 'dashed', 'dotted', 'dotdash', 'dashdot']
dashIterator = lineDashes.__iter__()

import sqlite3

filename_db = './experiments/experiments.db'

def open_db(filename_db):
    db = sqlite3.connect(filename_db)
    cur = db.cursor()
    return db, cur

db, cur = open_db(filename_db)

class Agent:
    legend: str
    line_p: Line
    line_select: Line
    raw: dict
    color: str
    id: int

    def show(self):
        if self.line_p is not None: self.line_p.visible = True
        if self.line_select is not None: self.line_select.visible = True

    def hide(self):
        if self.line_p is not None: self.line_p.visible = False
        if self.line_select is not None: self.line_select.visible = False  

class Experiment:
    rewards: List = []
    routes: List[Dict] =[]
    episode_data: List[Dict] =[]
    id: int = 0
    episodes: int = 0
    comment: str = ''
    name: str = ''

    def __init__(self, id=0, cur=None):
        self.id = id
        if cur is None: return
        cur.execute('WITH exp(agent) AS '
                '(  SELECT agent '
                '   FROM rewards '
                '   WHERE experiment_id = ?) '
                'SELECT DISTINCT agent '
                'FROM exp; ', (self.id,))
        agents = cur.fetchall()
        episodes = cur.execute('SELECT episodes FROM experiments WHERE id=?', (self.id,)).fetchone()
        if episodes is not None: self.episodes = episodes[0]
        colorIterator = Category20_20.__iter__()
        self.rewards=[]
        self.routes = []
        self.episode_data = []
        for (agent_id, ) in agents:
            agent = Agent()
            agent.id = agent_id
            agent.legend = 'experiment {}, agent {}'.format(self.id, agent.id)
            print(agent.legend)
            agent.raw= {'episode':[], 'reward':[]}
            #agent.columnData = ColumnDataSource(agent.raw)
            agent.color=next(colorIterator)
            self.rewards.append(agent)
        self.load_name_comment(cur)

    def load_name_comment(self, cur):
        (self.name, self.comment) = cur.execute('SELECT name, comment '
                                            ' FROM experiments '
                                            ' WHERE id=?', (self.id,)).fetchone()


    def load_rewards(self, cur, only_new=False) -> List[Agent]:
        strFromId = ""
        if only_new:
            strFromId = " AND episode >= {}".format(self.episodes-1)
        for agent in self.rewards:
            exp_data = cur.execute('SELECT episode, reward '
                                'FROM rewards '
                                'WHERE experiment_id = ? '
                                'AND agent = ?' + strFromId, (self.id, agent.id)).fetchall()
            episodes = [e for (e, _) in exp_data]
            rewards = [r for (_, r) in exp_data]
            agent.raw['episode'].extend(episodes)
            agent.raw['reward'].extend(rewards)
        (self.episodes,) = cur.execute('SELECT episodes FROM experiments WHERE id=?',
                                         (self.id,)).fetchone()

    def load_routes(self, cur, only_new=False):
        strFromId = ""
        if only_new:
            strFromId = " AND episode >= {}".format(self.episodes-1)
        data = cur.execute('SELECT episode, key, val '
                        '   FROM route '
                        '   WHERE experiment_id = ? ' + strFromId,
                        (self.id, ) ).fetchall()
        for (episode, key, val) in data:
            val = pickle.loads(val)
            while episode>=len(self.routes): self.routes.append({})
            self.routes[episode].update({key:val})
            self.routes[episode].update({'episode':episode})
        self._load_episode_data(cur, only_new)
        (self.episodes,) = cur.execute('SELECT episodes FROM experiments WHERE id=?',
                                         (self.id,)).fetchone()
        #return routes

    def _load_episode_data(self, cur, only_new=False):
        strFromId = ""
        if only_new:
            strFromId = " AND episode >= {}".format(self.episodes-1)
        data = cur.execute('SELECT episode, key, val '
                        '   FROM episode '
                        '   WHERE experiment_id = ? ' + strFromId,
                        (self.id, ) ).fetchall()
        for (episode, key, val) in data:
            val = pickle.loads(val)
            while episode>=len(self.episode_data): self.episode_data.append({})
            self.episode_data[episode].update({key:val})
            self.episode_data[episode].update({'episode':episode})
        
        #return episode_data

def get_experiment_list(cur):
    cur.execute('SELECT experiments.id, name, time, comment, episodes '
                ' FROM experiments')
                #'FROM experiments INNER JOIN live_experiment '
                #'on experiments.id = live_experiment.id')
    experiments = {str(id): "{} - {} ({} episodes)".format(name,time,episodes) for id, name, time, comment, episodes in cur.fetchall()}
    return experiments

'''def load_experiment(id, cur) -> Experiment:
    cur.execute('WITH exp(agent) AS '
                '(  SELECT agent '
                '   FROM rewards '
                '   WHERE experiment_id = ?) '
                'SELECT DISTINCT agent '
                'FROM exp; ', (id,))
    agents = cur.fetchall()
    (episodes,) = cur.execute('SELECT episodes FROM experiments WHERE id=?', (id,)).fetchone()
    if len(agents) == 0: return None
    experiment=Experiment(id, agents)
    #experiment.id = id
    #experiment.rewards = load_rewards(id, agents, cur)
    #experiment.load_rewards(cur)
    #experiment.routes = load_routes(id, cur)
    #experiment.load_routes(cur)
    #experiment.load_episode_data(cur)
    experiment.episodes = episodes
    print('EXPERIMENT: len(rewards)={} len(routes)={} id={} episodes={}'.format(
        len(experiment.rewards), len(experiment.routes), experiment.id, experiment.episodes))
    return experiment'''

class PlotRewards:
    line_p: List[Line] = []
    line_select: List[Line] = []
    column_data: List[ColumnDataSource] = []
    raw_data: List[Dict] = []
    main: figure
    select: figure
    range_tool:RangeTool
    exp: Experiment = Experiment()
    legend: Legend
    colorIterator = Category20_20.__iter__()
    txtComment= TextAreaInput(value="", title="Comment:")
    txtName= TextAreaInput(value="", title="Name:")
    btnSaveName = Button(label="Speichern", button_type="default", max_width = 200)
    btnSaveComment = Button(label="Speichern", button_type="default", max_width = 200)

    def __init__(self):
        self.main = figure(title='Rewards', x_axis_label = 'Reward', y_axis_label = 'Episode',  toolbar_location='above',
            x_range=(0,100), output_backend = 'svg')
        self.main.xaxis.visible = True
        self.select = figure(title="Drag the middle and edges of the selection box to change the range above",
                        plot_height=130, y_range=self.main.y_range,
                        tools="", toolbar_location=None, background_fill_color="#efefef",  x_range=(0,100))
        self.range_tool = RangeTool(x_range=self.main.x_range)
        self.range_tool.overlay.fill_color = "navy"
        self.range_tool.overlay.fill_alpha = 0.2

        self.select.ygrid.grid_line_color = None
        self.select.add_tools(self.range_tool)
        self.select.toolbar.active_multi = self.range_tool
        self.legend = Legend(title='Legend', items=[])
        self.legend.location = "bottom_right"
        self.legend.click_policy="mute"
        self.main.add_layout(self.legend)

    def set_experiment(self, id, cur):    
        if id != self.exp.id:
            self.exp = Experiment(id, cur)#load_experiment(id, cur)
            self.exp.load_rewards(cur)
            self.txtComment.update(value = self.exp.comment)#)= TextAreaInput(value="", title="Comment:")
            self.txtName.update(value = self.exp.name)
        else: 
            self.exp.load_rewards(cur, True)
        for l in self.line_p: l.update(visible = False)
        for l in self.line_select: l.update(visible = False)
        print('P_raw:AGENTS=',len(self.exp.rewards))
        for no, agent in enumerate(self.exp.rewards):
            if no>= len(self.line_p):
                self.create_lines(agent)
            else:
                self.raw_data[no] = agent.raw
                self.column_data[no].data = self.raw_data[no]
            self.set_line_attributes(no, agent)
        self.update_select_range()
        self.show_legend_axes()
        #txtName= TextAreaInput(value="", title="Name:")


    def show_legend_axes(self):
        self.main.legend.items=[]
        for no, agent in enumerate(self.exp.rewards):
            self.main.legend.items.append(LegendItem(label=agent.legend,renderers=[self.line_p[no]]))  
        self.main.xaxis.axis_label ="Reward"


    def create_lines(self, agent:Agent):
        color =next(self.colorIterator)
        self.raw_data.append(agent.raw)
        self.column_data.append(ColumnDataSource(self.raw_data[-1]))
        line_p = self.main.line(x='episode', y='reward', 
                    line_width=1, line_alpha=0.8, hover_line_alpha=1.0,
                    line_color = color, hover_line_color=color, 
                    muted_color=color, muted_alpha=0.2, name ='line',
                    source = self.column_data[-1])
        line_select = self.select.line(x='episode', y='reward',
                    line_width=1, line_alpha=0.8, name ='line',
                    line_color = color,
                    source = self.column_data[-1])
        self.line_p.append(line_p)
        self.line_select.append(line_select)
        return line_p, line_select

    def set_line_attributes(self, no, agent: Agent):
        self.line_p[no].visible = True
        self.line_select[no].visible = True
       


    def update_select_range(self):
        max_range=1
        cur.execute('SELECT episodes FROM experiments WHERE id =?;', (self.exp.id,))
        (episodes,) = cur.fetchone()
        max_range = max(episodes, max_range)
        print ('max_range:',max_range)
        self.select.x_range.update(start=0,end=max_range)
        self.main.x_range.update(start=0,end=max_range)

class PlotRoutes:
    route = {}
    line: Line
    lines: List[Line]
    column_raw = {}
    column_single = defaultdict(list) 
    column_two = defaultdict(list) 
    cd_line = ColumnDataSource(column_raw)
    cd_single = ColumnDataSource(column_single)
    cd_two = ColumnDataSource(column_two)
    colorIterator = Category20_20.__iter__()
    exp: Experiment = Experiment()

    def adjust_range_y(self):
        min_val = []
        max_val = []
        for line in self.lines:
            if not line.visible: continue
            min_val.append(min(self.cd_line.data[line.glyph.y]))
            max_val.append(max(self.cd_line.data[line.glyph.y]))
        if len(min_val)>0: 
            self.fig2yrange.start = min(min_val) -0.1
            self.fig2yrange.start = min(self.fig2yrange.start, 0)
            self.fig2yrange.end = max(max_val) +0.1
        coords = self.column_raw['pos_dx_m']\
                + self.column_raw['pos_dy_m']\
                + self.column_two['start_x']\
                + self.column_two['start_y']\
                + self.column_two['target_x']\
                + self.column_two['target_y']
        maxr = minr = 0
        #for c in coords:
        #    maxr = max(maxr, max(c))
        #    minr = min(minr, min(c))
        if len(coords)>0:
            maxr = max(coords)
            minr = min(coords)
        space = abs(maxr-minr)*0.05
        self.fig1range.update(start = minr-space, end = maxr+space)

    def slide_route(self, attr, old, new):
        print('SLIDER ',new)
        self.set_route(self.exp.routes[new])
        self.set_episode_data(self.exp.episode_data[new])

    def select_plot(self, value, old, new):
        for l in self.lines: l.visible = False
        self.fig2.legend.items = []
        for n in new:
            id = int(n)
            line = self.lines[id]
            line.visible = True      
            print('line ',id,' min=',min(self.cd_line.data[line.glyph.y]),
                    ' max=',max(self.cd_line.data[line.glyph.y]))      
            self.fig2.legend.items.append(LegendItem(label=line.name ,renderers=[line]))
        self.adjust_range_y()


    def __init__(self):
        self.slider = Slider(start=0, end=10, value=0, step=1, title="Episode")
        self.slider.on_change("value", self.slide_route)
        self.fig = figure(title='Route', x_axis_label = 'x', y_axis_label = 'y',  toolbar_location='above', output_backend='svg')
        self.fig2 = figure(title='Other', x_axis_label = 'step', y_axis_label = 'variables',  toolbar_location='above', output_backend='svg')
        self.figs= [self.fig, self.fig2]
        self.line = self.fig.line(x='pos_dx_m', y='pos_dy_m',
                    line_width=1, line_alpha=0.8, name ='linearrr',
                    source = self.cd_line)
        self.fig1range = Range1d(1,2)
        self.fig.x_range = self.fig1range
        self.fig.y_range = self.fig1range
        self.lines=[]                    
        self.start = self.fig.circle(x=0, y=0, size = 5, line_color="blue", fill_color="white", line_width=3)
        self.target = self.fig.circle(x="target_dx_m", y="target_dy_m", size = 5, line_color="red", fill_color="white", line_width=3, source=self.cd_single)
        self.start_heading = self.fig.line(x='start_x', y='start_y',
                    line_width=1, line_alpha=0.8, name ='line2',
                    line_color='blue', source = self.cd_two)
        self.target_heading = self.fig.line(x='target_x', y='target_y',
                    line_width=1, line_alpha=0.8, name ='line2',
                    line_color='red', source = self.cd_two)
        self.fig.xaxis.visible = True
        self.legend = Legend(title='Legend', items=[])
        self.legend.location = "bottom_right"
        self.legend.click_policy="hide"
        self.fig2.add_layout(self.legend)
        self.mutlichoice = MultiChoice(options=list(self.column_raw.keys()))
        self.mutlichoice.on_change('value', self.select_plot)
        self.fig2yrange = Range1d(1,2)
        self.fig2.y_range = self.fig2yrange

    def set_experiment(self, id, cur):
        if id != self.exp.id:
            self.exp = Experiment(id,cur)
            self.exp.load_routes(cur)
            for l in self.lines: 
                l.name=''
                l.visible = False
            self.mutlichoice.update(value=[])
        else: 
            self.exp.load_routes(cur, True)
        print('ROUTES EXPID=',self.exp.id, ' EPISODES=',self.exp.episodes)
        if len(self.exp.routes)== 0: return        
        self.set_route(self.exp.routes[-1])
        self.set_episode_data(self.exp.episode_data[-1])
        self.slider.end = len(self.exp.routes)-1
        self.slider.value = len(self.exp.routes)-1

    def set_route(self, route: Dict):
        self.route = route
        self.fig.title.text='Episode {}'.format(route['episode'])
        self.column_raw.clear()
        for (k,v) in route.items():
            if k == 'episode': continue
            self.column_raw[k] = v
        self.column_raw['step']=[]
        for step, x in enumerate(self.route['pos_dx_m']):
            self.column_raw['step'].append(step)
        for k, v in self.route.items():
            print(k)
            if k == 'episode': continue
            print('[',len(v),']')
        self.cd_line.data = self.column_raw
        line_no=0
        for k,v in self.column_raw.items():
            if k=='pos_dx_m' or k=='pos_dy_m' or k=='step':continue
            #print('len(self.lines):', len(self.lines))
            if len(self.lines)<=line_no:
                line = self.fig2.line(line_width=1, line_alpha=0.8, name =k, source = self.cd_line,
                        color = next(self.colorIterator))    
                self.lines.append(line)
            self.lines[line_no].glyph.x ='step'
            self.lines[line_no].glyph.y =k
            self.lines[line_no].name = k
            print('len(self.lines):', len(self.lines),'  line_no:', line_no, ' name:', self.lines[line_no].name)
            line_no+=1
        mc=[]
        for no, line in enumerate(self.lines):
            if line.name !='': mc.append((str(no),line.name))
        self.mutlichoice.update(options=mc)
        print(self.route.keys())
        print(self.column_raw.keys())
        self.adjust_range_y()

    def set_episode_data(self, episode_data: Dict):
        self.episode_data = episode_data
        for (k,v) in episode_data.items():
            self.column_single[k]=[v]
        self.cd_single.data = self.column_single
        if 'target_heading' in self.episode_data:
            x = self.episode_data['target_dx_m']
            y = self.episode_data['target_dy_m']
            h = self.episode_data['target_heading']
            self.column_two['target_x'] =[x, x+math.sin(h)*500]
            self.column_two['target_y'] =[y, y+math.cos(h)*500]
        if 'initial_heading_rad' in self.episode_data:
            x = 0
            y = 0
            h = self.episode_data['initial_heading_rad']
            self.column_two['start_x'] =[x, x+math.sin(h)*500]
            self.column_two['start_y'] =[y, y+math.cos(h)*500]
        self.cd_two.data = self.column_two
        print(self.episode_data.keys())
        print(self.column_single)
        print(self.column_two)






def select_experiments(value, old, new):
    #global experiment
    if new == '': return
    id = int(new)
    if id != p_rewards.exp.id:
        btnDelete2.update(visible = False, disabled = True)
        p_rewards.set_experiment(id, cur)
        p_routes.set_experiment(id, cur)

experiment_list = {}
experiment_selector = Select(options=list(experiment_list.items()))
experiment_selector.on_change('value', select_experiments)

p_rewards = PlotRewards()
p_routes = PlotRoutes()

def delete_experiment():
    if btnDelete2.visible:
        btnDelete2.update(visible = False, disabled = True)
    else:
        btnDelete2.update(visible = True, disabled = False)

def delete_experiment2():
    exp_id = p_rewards.exp.id
    print('deleting exp#',exp_id)
    btnDelete2.update(visible = False, disabled = True)
    live = cur.execute(' SELECT id FROM live_experiment WHERE id=?',(exp_id,)).fetchone()
    if live:
        print('cannot delete live experiment')
        return
    cur.execute(' DELETE FROM experiments WHERE id=?',(exp_id,))
    cur.execute(' DELETE FROM episode WHERE experiment_id=?',(exp_id,))
    cur.execute(' DELETE FROM rewards WHERE experiment_id=?',(exp_id,))
    cur.execute(' DELETE FROM route WHERE experiment_id=?',(exp_id,))
    db.commit()
    new_val = ''
    for k in experiment_list.keys():
        if int(k) > exp_id:
            new_val = k
            break
    experiment_selector.update( value=new_val)
    update_selection_list(cur)
    print('deleted exp#',exp_id)

btnDelete = Button(label="Experiment Löschen", button_type="warning", max_width = 200)
btnDelete.on_click(delete_experiment)
btnDelete2 = Button(label="wirklich Löschen", button_type="danger", max_width = 200, visible = False, disabled = True)
btnDelete2.on_click(delete_experiment2)

def save_comment():
    print('saving comment #{}'.format(p_rewards.exp.id))
    p_rewards.exp.comment = p_rewards.txtComment.value
    cur.execute('UPDATE experiments SET comment=? WHERE id=?', (p_rewards.exp.comment, p_rewards.exp.id))
    db.commit()

def save_name():
    print('saving name #{}'.format(p_rewards.exp.id))
    p_rewards.exp.name = p_rewards.txtName.value
    cur.execute('UPDATE experiments SET name=? WHERE id=?', (p_rewards.exp.name, p_rewards.exp.id))
    db.commit()


p_rewards.btnSaveComment.on_click(save_comment)
p_rewards.btnSaveName.on_click(save_name)
#my_col = column([experiment_selector, p_rewards.main, p_rewards.select])
#my_row2 = row([p_routes.fig, p_routes.fig2])
#my_col2 = column([p_routes.slider, my_row2])
#my_row = row([my_col,my_col2])
my_row = row([experiment_selector, p_routes.slider, p_routes.mutlichoice])
my_col2 = column([p_rewards.main, p_rewards.select])
sub_row = row([btnDelete, btnDelete2])
my_col3 = column([p_routes.fig, sub_row,p_rewards.txtComment, p_rewards.btnSaveComment])
my_col4 = column([p_routes.fig2, p_rewards.txtName, p_rewards.btnSaveName])
my_row2 = row([my_col2,my_col3, my_col4])
my_col = column (my_row, my_row2)
doc =curdoc()
#doc.add_root(my_row)
doc.add_root(my_col)

import time
import threading
from tornado import gen
from functools import partial

my_col.sizing_mode = 'fixed'
my_col2.sizing_mode = 'fixed'
my_row.sizing_mode = 'fixed'

@gen.coroutine
def resize():
    p_routes.fig.sizing_mode  = 'scale_width'
    p_routes.fig2.sizing_mode  = 'stretch_width'
    p_rewards.main.sizing_mode  = 'stretch_width'
    p_rewards.select.sizing_mode  = 'stretch_width'
    my_row.sizing_mode = 'stretch_width'
    my_row2.sizing_mode = 'stretch_both'
    my_col.sizing_mode = 'stretch_both'
    my_col2.sizing_mode = 'stretch_both'
    my_col3.sizing_mode = 'stretch_both'
    my_col4.sizing_mode = 'stretch_both'
    experiment_selector.sizing_mode  = 'stretch_width'
    p_routes.mutlichoice.sizing_mode  = 'stretch_width'
    p_routes.slider.sizing_mode  = 'stretch_width'
    sub_row.sizing_mode = 'stretch_width'
    btnDelete.sizing_mode = 'stretch_width'
    btnDelete2.sizing_mode = 'stretch_width'
    print('resized')

def dispatch_resize():
    time.sleep(0.2)
    doc.add_next_tick_callback(resize)

@gen.coroutine
def update_selector(experiments):
    #experiment_selector.options=list(experiments.items())
    experiment_selector.update(options=list(experiments.items()))
    val =experiment_selector.value    
    experiment_selector.update(value='')
    experiment_selector.update(value=val)

def init_selector(experiment_list):
    experiment_selector.trigger('value', '', list(experiment_list.keys())[0])
    experiment_selector.value = list(experiment_list.keys())[0]
    #experiment_selector.update(value=list(experiment_list.keys())[0])

def update_selection_list(cur):
    global experiment_list
    exp_list_new = get_experiment_list(cur)
    updated = False
    for k in exp_list_new.keys():
        if exp_list_new.get(k) != experiment_list.get(k):
            updated = True
            break
    if len(set(exp_list_new.keys()) ^ set(experiment_list.keys()) ) >0: updated = True
    if updated:
        experiment_list = exp_list_new
        doc.add_next_tick_callback(partial(update_selector, experiment_list)) 
        if experiment_selector.value =='' and len(experiment_list) > 0:
            doc.add_next_tick_callback(partial(init_selector, experiment_list))     

def update_live_data_do(id):
    p_rewards.set_experiment(id, cur)
    p_routes.set_experiment(id, cur)
    print('relad exp #',id)

def update_live_data(cur, exp):
    cur.execute(' SELECT experiments.id '
                ' FROM experiments INNER JOIN live_experiment '
                ' on experiments.id = live_experiment.id')
    live_ids =[id for (id,) in cur.fetchall()]
    for id in live_ids:
        if experiment_selector.value == str(id):
            (episodes, ) = cur.execute(' SELECT episodes '
                ' FROM experiments WHERE id=?',(id,)).fetchone()
            if exp is None:
                print('experiment is None')
                return
            if episodes != exp.episodes:
                doc.add_next_tick_callback(partial(update_live_data_do, id)) 



def update_data_thread(experiments):
    db, cur = open_db(filename_db)
    while True:
        time.sleep(0.5)
        update_selection_list(cur)
        update_live_data(cur, p_routes.exp)


thread = threading.Thread(target=dispatch_resize, args=())
thread.start()   
thread2 = threading.Thread(target=update_data_thread, args=(experiment_list,))
thread2.daemon = True
thread2.start()   


