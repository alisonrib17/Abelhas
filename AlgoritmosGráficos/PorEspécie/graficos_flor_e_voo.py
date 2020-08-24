import numpy as np
import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, LabelSet, FactorRange
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.layouts import gridplot
from bokeh.io import export_png

x = ['LR', 'SVM', 'RF', 'DTREE', 'ENSEMBLE']

flor_e_voo = {
	'algo': x,
	'acc' : [62.79, 66.11, 63.45, 40.86, 69.43],
	'p'   : [62.57, 64.51, 56.33, 31.06, 67.60],
	'r'   : [53.36, 56.39, 44.31, 30.88, 56.08],
	'f1'  : [55.75, 58.35, 46.17, 30.45, 58.70]
}

output_file("especies_flor_e_voo.html")

source = ColumnDataSource(data=flor_e_voo)

p = figure(x_range=x, y_range=(0, 100), plot_height=350, title="Results the full flight",
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

