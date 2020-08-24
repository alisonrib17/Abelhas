import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.layouts import gridplot

x = ['LR', 'SVM', 'RF', 'DTREE', 'ENSEMBLE']

flor_e_voo = {
	'algo': x,
	'acc' : [72.83, 84.87, 88.27, 90.12, 86.41],
	'p'   : [71.19, 85.06, 77.74, 82.73, 85.99],
	'r'   : [71.42, 82.62, 72.47, 84.72, 80.06],
	'f1'  : [70.52, 83.16, 74.23, 83.58, 81.95]
}

flor = {
	'algo': x,
	'acc' : [79.90, 87.67, 87.21, 86.30, 84.93],
	'p'   : [77.01, 89.22, 78.37, 70.60, 79.48],
	'r'   : [73.37, 81.39, 67.94, 75.20, 63.22],
	'f1'  : [74.71, 84.53, 71.26, 72.49, 67.55]
}

voo = {
	'algo': x,
	'acc' : [67.61, 75.23, 78.09, 79.04, 76.19],
	'p'   : [59.71, 63.75, 61.05, 59.11, 61.09],
	'r'   : [57.02, 62.84, 57.13, 60.67, 55.72],
	'f1'  : [57.17, 62.75, 56.48, 59.38, 55.68]
}


output_file("generos_caracFisicas.html")

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

	p = figure(x_range=x, y_range=(0, 100), plot_height=50, title=title,
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
	p.legend.location = "top_left"
	p.legend.orientation = "horizontal"
	p.legend.label_text_font_size = "8pt"
	p.add_layout(labels1)
	p.add_layout(labels2)
	p.add_layout(labels3)

	return p

p1 = grafico(source1, ['acc1', 'acc2', 'acc3'], "Accuracy")
p2 = grafico(source2, ['p1', 'p2', 'p3'], "Precision")
p3 = grafico(source3, ['r1', 'r2', 'r3'], "Recall")
p4 = grafico(source4, ['f11', 'f12', 'f13'], "F1-score")

show(gridplot([p1,p2,p3,p4], ncols=2, plot_width=550, plot_height=600, toolbar_location=None))