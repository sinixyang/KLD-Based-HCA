# KLD-Based-HCA
Kullback-Leibler Divergence based Hierarchical Clustering Analysis
The program has been test in Python 3.6

How to use it, for we can use example:
dataframe = pandas.read_csv("ht_geo_sm.csv")
target_field = "Geo_object"
variable_fields = ['Cu', 'Zn', 'Ni', 'Pb', 'Ag', 'Co',  'As', 'Hg'] 
dataframe[variable_fields] = numpy.log(dataframe[variable_fields].values) #log-transform are used here
target_blacklist = ["QSE","QC","QW","SPNW"] #
kld_HCA(dataframe,target_field,variable_fields, "a", target_blacklist)
