import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.layouts import gridplot

x = ['LR', 'SVM', 'RF', 'DTREE', 'ENSEMBLE']

flor_e_voo = {
	'algo': x,
	'acc' : [53.08, 63.58, 55.24, 33.64, 59.25],
	'p'   : [56.45, 62.51, 43.64, 24.53, 57.72],
	'r'   : [44.75, 54.96, 34.58, 23.96, 46.84],
	'f1'  : [46.98, 57.20, 35.49, 23.60, 49.24]
}

flor = {
	'algo': x,
	'acc' : [50.22, 59.81, 57.99, 32.87, 60.27],
	'p'   : [43.70, 49.08, 39.26, 18.93, 53.92],
	'r'   : [37.16, 44.59, 29.25, 22.57, 40.36],
	'f1'  : [38.98, 46.15, 29.69, 20.02, 43.86]
}

voo = {
	'algo': x,
	'acc' : [50.47, 59.04, 51.42, 26.66, 55.23],
	'p'   : [38.10, 44.23, 35.75, 19.97, 42.42],
	'r'   : [35.60, 42.45, 36.82, 17.96, 39.77],
	'f1'  : [35.41, 41.48, 34.34, 18.29, 38.52]
}


output_file("com_grid_search.html")

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

p1 = grafico(source1, ['acc1', 'acc2', 'acc3'], "Acurácia com Grid Search")
p2 = grafico(source2, ['p1', 'p2', 'p3'], "Precisão com Grid Search")
p3 = grafico(source3, ['r1', 'r2', 'r3'], "Recall com Grid Search")
p4 = grafico(source4, ['f11', 'f12', 'f13'], "F1-score com Grid Search")

show(gridplot([p1,p2,p3,p4], ncols=2, plot_width=350, plot_height=300, toolbar_location=None))