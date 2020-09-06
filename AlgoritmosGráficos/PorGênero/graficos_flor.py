import numpy as np
import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, LabelSet, FactorRange
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.layouts import gridplot
from bokeh.io import export_png

x = ['LR', 'SVM', 'RF', 'DTREE', 'ENSEMBLE']

flor = {
	'algo': x,
	'acc' : [69.86, 75.34, 71.23, 50.22, 76.71],
	'p'   : [55.40, 66.98, 64.76, 36.73, 66.67],
	'r'   : [52.76, 58.04, 49.43, 33.17, 56.69],
	'f1'  : [53.78, 61.19, 53.14, 33.76, 59.59]
}

source = ColumnDataSource(data=flor)

p = figure(x_range=x, y_range=(0, 100), plot_height=350, title=None,
	toolbar_location=None, tools="")

p.vbar(x=dodge('algo', -0.25, range=p.x_range), top="acc", width=0.15, source=source,
	color="#c9d9d3", legend_label="Accuracy")

labels_acc = LabelSet(x=dodge('algo', -0.25, range=p.x_range), y="acc", text="acc", text_align='center', 
	x_offset=10, y_offset=12, source=source, angle=20, text_font_size="8pt")

p.vbar(x=dodge('algo',  -0.08,  range=p.x_range), top="p", width=0.15, source=source,
	color="#718dbf", legend_label="Precision")

labels_p = LabelSet(x=dodge('algo', -0.08, range=p.x_range), y="p", text="p", text_align='center', 
	x_offset=10, y_offset=12, source=source, angle=20, text_font_size="8pt")

p.vbar(x=dodge('algo',  0.09, range=p.x_range), top="r", width=0.15, source=source,
	color="#e84d60", legend_label="Recall")

labels_r = LabelSet(x=dodge('algo', 0.09, range=p.x_range), y="r", text="r", text_align='center', 
	x_offset=10, y_offset=12, source=source, angle=20, text_font_size="8pt")

p.vbar(x=dodge('algo',  0.26, range=p.x_range), top="f1", width=0.15, source=source,
	color="#fa8072", legend_label="F1-score")

labels_f1 = LabelSet(x=dodge('algo', 0.26, range=p.x_range), y="f1", text="f1", text_align='center', 
	x_offset=10, y_offset=12, source=source, angle=20, text_font_size="8pt")

p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.yaxis.axis_label = "Score %"
p.legend.location = "top_left"
p.legend.orientation = "horizontal"
p.add_layout(labels_acc)
p.add_layout(labels_p)
p.add_layout(labels_r)
p.add_layout(labels_f1)
output_file("generos_flor.html")
show(p)