import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.layouts import gridplot

x = ['LR', 'SVM', 'RF', 'DTREE', 'ENSEMBLE']

flor_e_voo = {
	'algo': x,
	'acc' : [71.42, 76.74, 79.73, 73.75, 77.74],
	'p'   : [68.26, 72.73, 73.54, 68.24, 74.46],
	'r'   : [61.73, 69.52, 64.17, 68.62, 63.19],
	'f1'  : [62.63, 70.39, 66.25, 68.07, 65.91]
}

flor = {
	'algo': x,
	'acc' : [76.69, 80.09, 79.12, 71.35, 80.09],
	'p'   : [69.38, 76.15, 62.84, 54.80, 61.71],
	'r'   : [59.74, 67.37, 49.48, 53.12, 53.32],
	'f1'  : [62.26, 70.31, 52.12, 53.63, 55.41]
}

voo = {
	'algo': x,
	'acc' : [64.21, 61.81, 67.36, 73.68, 60.00],
	'p'   : [53.71, 48.81, 44.92, 58.45, 42.15],
	'r'   : [53.17, 47.76, 44.02, 58.48, 41.68],
	'f1'  : [52.99, 47.68, 42.39, 56.27, 40.34]
}


output_file("especies_caracFisicas.html")

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
			color="#c9d9d3", legend_label="Full flight")

	labels1 = LabelSet(x=dodge('algo', -0.25, range=p.x_range), y=top[0], text=top[0], text_align='center', 
	x_offset=10, y_offset=12, source=source, angle=20, text_font_size="8pt")

	p.vbar(x=dodge('algo',  0.0,  range=p.x_range), top=top[1], width=0.2, source=source,
			color="#718dbf", legend_label="Flight in flower")

	labels2 = LabelSet(x=dodge('algo', 0.0, range=p.x_range), y=top[1], text=top[1], text_align='center', 
	x_offset=10, y_offset=12, source=source, angle=20, text_font_size="8pt")

	p.vbar(x=dodge('algo',  0.25, range=p.x_range), top=top[2], width=0.2, source=source,
			color="#e84d60", legend_label="Flight out of flower")

	labels3 = LabelSet(x=dodge('algo', 0.25, range=p.x_range), y=top[2], text=top[2], text_align='center', 
	x_offset=10, y_offset=12, source=source, angle=20, text_font_size="8pt")

	p.x_range.range_padding = 0.1
	p.xgrid.grid_line_color = None
	p.yaxis.axis_label = "Score %"
	p.legend.location = "top_left"
	p.legend.orientation = "horizontal"
	p.legend.label_text_font_size = "8pt"
	p.add_layout(labels1)
	p.add_layout(labels2)
	p.add_layout(labels3)

	return p

p1 = grafico(source1, ['acc1', 'acc2', 'acc3'], None)
p2 = grafico(source2, ['p1', 'p2', 'p3'], None)
p3 = grafico(source3, ['r1', 'r2', 'r3'], None)
p4 = grafico(source4, ['f11', 'f12', 'f13'], None)

show(gridplot([p1,p2,p3,p4], ncols=2, plot_width=400, plot_height=450, toolbar_location=None))