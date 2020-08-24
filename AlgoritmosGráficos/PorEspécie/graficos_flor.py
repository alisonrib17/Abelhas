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
	'acc' : [67.61, 72.33, 67.47, 47.57, 70.87],
	'p'   : [61.50, 68.81, 55.69, 27.23, 59.99],
	'r'   : [48.73, 58.79, 39.70, 27.92, 45.90],
	'f1'  : [52.34, 62.20, 43.02, 27.19, 49.55]
}

output_file("especies_flor.html")

source = ColumnDataSource(data=flor)

p = figure(x_range=x, y_range=(0, 100), plot_height=350, title="Results lower-based classification",
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
p.legend.location = "top_left"
p.legend.orientation = "horizontal"
p.add_layout(labels_acc)
p.add_layout(labels_p)
p.add_layout(labels_r)
p.add_layout(labels_f1)
show(p)