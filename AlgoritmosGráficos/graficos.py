import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.layouts import gridplot

x = ['LR', 'SVM', 'RF', 'DTREE', 'ENSEMBLE']

flor_e_voo = {
	'algo': x,
	'acc' : [51.54, 62.96, 59.25, 35.80, 58.02],
	'p'   : [42.27, 63.67, 55.78, 24.07, 53.85],
	'r'   : [40.04, 52.99, 38.65, 23.64, 46.40],
	'f1'  : [40.65, 55.16, 40.45, 23.58, 47.77]
}

flor = {
	'algo': x,
	'acc' : [53.42, 66.21, 58.44, 34.70, 58.44],
	'p'   : [46.54, 55.50, 36.82, 16.87, 48.21],
	'r'   : [38.13, 47.11, 30.29, 18.39, 38.78],
	'f1'  : [40.44, 49.39, 29.96, 17.09, 41.29]
}

voo = {
	'algo': x,
	'acc' : [41.90, 58.09, 47.61, 24.76, 48.57],
	'p'   : [29.96, 46.60, 33.59, 14.08, 29.50],
	'r'   : [29.55, 43.05, 32.13, 15.06, 33.03],
	'f1'  : [28.52, 42.64, 30.70, 14.39, 30.73]
}


output_file("sem_grid_search.html")

data1 = {'algo' : x,
	'acc1': flor_e_voo['acc'],
	'acc2'  : flor['acc'],
	'acc3'  : voo['acc'],
}
data2 = {'algo' : x,
	'p1': flor_e_voo['p'],
	'p2'  : flor['p'],
	'p3'  : voo['p'],
}
data3 = {'algo' : x,
	'r1': flor_e_voo['r'],
	'r2'  : flor['r'],
	'r3'  : voo['r'],
}
data4 = {'algo' : x,
	'f11': flor_e_voo['f1'],
	'f12'  : flor['f1'],
	'f13'  : voo['f1'],
}

source1 = ColumnDataSource(data=data1)
source2 = ColumnDataSource(data=data2)
source3 = ColumnDataSource(data=data3)
source4 = ColumnDataSource(data=data4)

def grafico(source, top, title):

	p = figure(x_range=x, y_range=(0, 100), plot_height=150, title=title,
			toolbar_location=None, tools="")

	p.vbar(x=dodge('algo', -0.25, range=p.x_range), top=top[0], width=0.2, source=source,
			color="#c9d9d3", legend_label="Flor e Voo")

	p.vbar(x=dodge('algo',  0.0,  range=p.x_range), top=top[1], width=0.2, source=source,
			color="#718dbf", legend_label="Flor")

	p.vbar(x=dodge('algo',  0.25, range=p.x_range), top=top[2], width=0.2, source=source,
			color="#e84d60", legend_label="Voo")

	p.x_range.range_padding = 0.1
	p.xgrid.grid_line_color = None
	p.legend.location = "top_left"
	p.legend.orientation = "horizontal"

	return p

p1 = grafico(source1, ['acc1', 'acc2', 'acc3'], "Acurácia com parâmetros default")
p2 = grafico(source2, ['p1', 'p2', 'p3'], "Precisão com parâmetros default")
p3 = grafico(source3, ['r1', 'r2', 'r3'], "Recall com parâmetros default")
p4 = grafico(source4, ['f11', 'f12', 'f13'], "F1-score com parâmetros default")

show(gridplot([p1,p2,p3,p4], ncols=2, plot_width=350, plot_height=300, toolbar_location=None))