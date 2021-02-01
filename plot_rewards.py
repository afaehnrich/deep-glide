# myapp.py

from random import random
from collections import defaultdict
from bokeh.layouts import row, column
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.io import show
from bokeh.models import CustomJS, MultiChoice, TextAreaInput, Legend, LegendItem, Line, RangeTool, Range1d
from bokeh.palettes import Category20_20
from bokeh.palettes import Viridis6

from bokeh.models.markers import marker_types
from bokeh.models import ColumnDataSource, HoverTool

import itertools

colorIterator = Category20_20.__iter__()
lineDashes =['solid', 'dashed', 'dotted', 'dotdash', 'dashdot']
dashIterator = lineDashes.__iter__()

import sqlite3

filename_db = './experiments/experiments.db'
db = sqlite3.connect(filename_db)
cur = db.cursor()

def experiment_list():
    cur.execute('SELECT id, name, time, comment, episodes '
                'FROM experiments '
                'WHERE episodes > 1 ')
    experiments = [(str(id), "{} - {} ({} episodes)".format(name,time,episodes)) for id, name, time, comment, episodes in cur.fetchall()]
    return experiments

def load_experiment(id) -> list:
    cur.execute('WITH exp(agent) AS '
                '(  SELECT agent '
                '   FROM rewards '
                '   WHERE experiment_id = ?) '
                'SELECT DISTINCT agent '
                'FROM exp; ', (id,))
    agents = cur.fetchall()
    if len(agents) == 0: return None
    color=next(colorIterator)
    dash = 0
    experiment=[]
    for (agent, ) in agents:
        ag_dict= defaultdict(list)
        exp_data = cur.execute('SELECT episode, reward '
                            'FROM rewards '
                            'WHERE experiment_id = ? '
                            'AND agent = ?', (id, agent)).fetchall()
        episodes = [e for (e, _) in exp_data]
        rewards = [r for (_, r) in exp_data]
        ag_dict['agent']='agent {}'.format(agent)
        ag_dict['experiment']='experiment {}'.format(id)
        ag_dict['legend']='experiment {}, agent {}'.format(id, agent)
        ag_dict['data']= {'episode':episodes, 'reward':rewards}
        dash += 1
        if dash >= len(lineDashes):
            color=next(colorIterator)
            dash = 0
        ag_dict['color']=color
        experiment.append(ag_dict)
    return experiment


debug = TextAreaInput(value="default", rows=20, title="Label:")

# create a plot and style its properties

p = figure(title='Rewards', x_axis_label = 'Reward', y_axis_label = 'Episode',  toolbar_location=None,
            x_range=(0,100))


experiments = experiment_list()
#rewards = load_experiment(11)
#r0 = rewards[0]

line_buffer = {}

def show_legend_axes():
    p.legend.items=[]
    for (_,experiment) in line_buffer.items():
        for buf_dat in experiment:
            if buf_dat.line_p.visible: 
                p.legend.items.append(LegendItem(label=buf_dat.raw['legend'],renderers=[buf_dat.line_p]))  
    p.xaxis.axis_label ="Reward"

class LineBufferData:
    line_p: Line
    line_select: Line
    raw: dict
    columnData: ColumnDataSource


def create_lines(exp:dict) -> LineBufferData:
    lbuf = LineBufferData()
    lbuf.raw = exp
    lbuf.columnData = ColumnDataSource(exp['data'])
    lbuf.line_p = p.line(x='episode', y='reward',
                line_width=1, line_color=exp['color'], line_alpha=0.8,
                hover_line_color=exp['color'], hover_line_alpha=1.0,
                muted_color=exp['color'], muted_alpha=0.2,
                name ='line',
                source=lbuf.columnData)
    lbuf.line_select = select.line(x='episode', y='reward',
                line_width=1, line_color=exp['color'], line_alpha=0.8,
                name ='line',
                source=lbuf.columnData)
    if lbuf.line_p is None: return
    if lbuf.line_select is None: return
    return lbuf

def update_select_range():
    max_range=1
    for (id,lbuf) in line_buffer.items():
        for buf_dat in lbuf:
            if buf_dat.line_p.visible:
                cur.execute('SELECT episodes FROM experiments WHERE id =?;', (id,))
                (episodes,) = cur.fetchone()
                max_range = max(episodes, max_range)
    print ('max_range:',max_range)
    select.x_range.update(start=0,end=max_range)
    p.x_range.end = min (p.x_range.end, max_range)

def select_experiments(value, old, new):
    for id in set(new) & set (line_buffer.keys()):
        for buf_dat in line_buffer.get(id):
            buf_dat.line_p.visible = True
            buf_dat.line_select.visible = True
    for id in set(new)-set(old)-set(line_buffer.keys()):
        experiment = load_experiment(int(id))
        if experiment == None: continue
        exp_data=[]
        for exp in experiment:           
            buf_dat = create_lines(exp)
            if buf_dat is None: continue
            exp_data.append(buf_dat)
        line_buffer.update({id: (exp_data)})
    for id in (set(old)-set(new)) & line_buffer.keys():
        for buf_dat in line_buffer.get(id):
            buf_dat.line_p.visible=False
            buf_dat.line_select.visible=False
    update_select_range()
    debug.value +='\n {} {} {}'.format(old, new, list(line_buffer.keys()))
    show_legend_axes()

experiment_selector = MultiChoice(options=experiments)
experiment_selector.on_change('value', select_experiments)
legend = Legend(title='Legend', items=[])
show_legend_axes()
p.add_layout(legend)
p.legend.location = "bottom_right"
p.legend.click_policy="mute"
p.xaxis.visible = True


select = figure(title="Drag the middle and edges of the selection box to change the range above",
                plot_height=130, y_range=p.y_range,
                tools="", toolbar_location=None, background_fill_color="#efefef",  x_range=(0,100))
print (p.x_range)
range_tool = RangeTool(x_range=p.x_range)
range_tool.overlay.fill_color = "navy"
range_tool.overlay.fill_alpha = 0.2

select.ygrid.grid_line_color = None
select.add_tools(range_tool)
select.toolbar.active_multi = range_tool

my_col = column([experiment_selector, p, select])
my_row = row([my_col, debug])
doc =curdoc()
doc.add_root(my_row)
p.sizing_mode = 'fixed'
select.sizing_mode = 'fixed'
my_col.sizing_mode = 'fixed'
my_row.sizing_mode = 'fixed'

import time
import threading
from tornado import gen

@gen.coroutine
def resize():
    p.sizing_mode = 'stretch_both'
    select.sizing_mode = 'stretch_width'
    my_col.sizing_mode = 'stretch_width'
    my_row.sizing_mode = 'stretch_both'
    print('resized')

def dispatch_resize():
    time.sleep(0.4)
    doc.add_next_tick_callback(resize)

thread = threading.Thread(target=dispatch_resize, args=())
thread.start()   

#curdoc().sizing_mode = 'stretch_both'
#curdoc().add_root(my_col)
if len(experiments) > 0:
    experiment_selector.trigger('value', [], [experiments[0][0]])
    experiment_selector.value = [experiments[0][0]]
#else:
#select_experiments('value', [], [experiments[0][0]])
