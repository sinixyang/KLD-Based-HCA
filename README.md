A Kullback-Leibler Divergence based Hierarchical Clustering Analysis for geochemcial datasets to classify geo-objects. 
There are two files in the folder, "KLD_HCA.py" is the program, "ht_geo_sm.csv" is the demo data. The library includes Numpy, Pandas, and Matplotlib, which are required to run the program.The program has been test in Python 3.6, when it runs in Python 2.7, it would yield a result with a layout problem. The agorithm relies on the HCA framework provide by scipy, we just simply change the promixity matrix used in the framework. So it is easy to rewrite the algorithm by yourself.
The program includes three HCA methods,1) HCA using KLDSM and its derivates as the promixities, 2) HCA uses Euclidean distance as the promixity, 3) HCA uses Aitchison distance as the promixity. The introductions about the functions and their parameters are embedded in the codes.

An example for the HCA that uses KLDSM as the proximity:
from KLD_HCA import *
dataframe = pandas.read_csv("ht_geo_sm.csv") #load the data
target_field = "Geo_object" #the field saving the labels of data in the dataframe
variable_fields = ['Cu', 'Zn', 'Ni', 'Pb', 'Ag', 'Co',  'As', 'Hg'] #interested elements
dataframe[variable_fields] = numpy.log(dataframe[variable_fields].values) #Do log-transform before HCA
target_blacklist = ["QSE","QC","QW","SPNW"] #a list saving geo-objects' name that we are not interest, those geo-objects are not involved in the HCA
kld_HCA(dataframe,target_field,variable_fields, "a", target_blacklist) #progress the data use KLDSM.
