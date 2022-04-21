import matplotlib.pyplot as plt
from pyparsing import col
from mysklearn import myutils

def bar_chart(x, y, title, x_label, y_label, tilt_x=False, has_buckets=False, width=1, align="center"):
    plt.figure(figsize=(14, 5))
    if has_buckets:
        plt.bar(x, y, width=width, edgecolor="black", align=align)
    else:
        plt.bar(x, y)
    plt.grid(axis='y', linestyle='--')
    if tilt_x:
        plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def pie_chart(x, y, title):
    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.title(title)
    plt.show()
    
def histogram(data, title, x_label, y_label, names=[], width=4):
    # data is a 1D list of data values
    print(len(data))
    plt.figure(figsize=(width, 5))
    if len(data) > 1:
        for vals, name in zip(data, names):
            plt.hist(vals, bins=10, label=name)
    else:
        plt.hist(data[0], bins=10) 
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis='y', linestyle = "--")
    plt.legend()
    plt.show()

def scatter_chart(x, y, title, x_label, y_label):
    plt.figure() # make a new current figure
    plt.scatter(x, y)
    m, b = myutils.compute_slope_intercept_plot(x, y)
    r = myutils.compute_correlation_coefficient(x, y)
    cov = myutils.compute_covariance(x, y)
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r", lw=5)
    plt.grid(linestyle='--')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.annotate(f"corr: {format(r, '.2f')}\ncov: {format(cov, '.2f')}", xy=(0.88, 0.88), xycoords="axes fraction", 
                 horizontalalignment="center", color="r", bbox=dict(boxstyle="round", fc="1", color="r"))
    plt.show()

def box_plot(distributions, labels, title, x_label, y_label): # distributions and labels are parallel
    # distributions: list of 1D lists of values
    plt.figure(figsize=(14, 5))
    plt.boxplot(distributions)
    plt.xticks(list(range(1, len(distributions) + 1)), labels)
    plt.grid(axis='y', linestyle='--')
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()