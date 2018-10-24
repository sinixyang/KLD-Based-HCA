# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:59:34 2018

@author: Sinix Studio
"""
import pandas
from matplotlib import pyplot
from scipy.stats import zscore
import numpy
import scipy.cluster.hierarchy as sch
import scipy
from itertools import combinations

def closure(mat,residue = False, summation = 1000000):
    '''
    Close the geochemical data, a preprocess step before log-ratio transform, more inforation can be found in the books about composition data
    Keyword arguments:
    
        *mat*: a 2d numpy array, n rows stand for samples, m columns stand for variables.   
        *residue*: True or False. If True the output numpy array in a shape of n×(m+1), the last column is the residue
        *summuation*: The summation of composition data
    '''
    if residue:
        mat = numpy.hstack( (mat / summation, (summation - mat.sum(1)).reshape(-1,1)))
    else:
        mat = mat / mat.sum(axis=1, keepdims=True)
    return mat

def clr(mat, residue = True, summation = 1000000):
    '''
    Apply centered log ratio transform on composition data
    
    Keyword arguments:
        *mat*: a 2d numpy array, n rows stand for samples, m columns stand for variables.   
        *residue*: True or False. If True the output numpy array in a shape of n×(m+1), the last column is the residue
        *summuation*: The summation of composition data
    '''
    mat = closure(mat,residue = residue, summation = summation)
    lmat = numpy.log(mat)
    gm = lmat.mean(axis=-1, keepdims=True)
    if residue:
        return (lmat - gm)[:,:-1]
    else:     
        return lmat - gm

def cal_kldivergence(cov0, cov1, mu0 = numpy.array([[0.0]]), mu1 = numpy.array([[0.0]]) , mode = "a"):
    '''
    Calculate the KL divergence between N(mu0,cov0) and N(mu1,cov1)
    Keyword arguments:
    *cov0*: covariance matrix of N0
    *cov1*: covariance matrix of N1    
    *mu0*: mean vector of N0
    *mu1*: mean vector of N1
    *mode*: 'c' stands for the part about covariance; 'm' stands for the part about mean;'a' stands for both of them
    '''
    nof_variable = cov1.shape[0]
    mu1 = mu1.reshape((nof_variable,1))
    mu0 = mu0.reshape((nof_variable,1))
    mu1 = numpy.matrix(mu1)
    mu0 = numpy.matrix(mu0)
    cov1 = numpy.matrix(cov1)
    cov0 = numpy.matrix(cov0)
    if mode == "a":
        return (numpy.trace(cov1.I * cov0) + (mu1 - mu0).T * cov1.I * (mu1 - mu0) - nof_variable - numpy.log( numpy.linalg.det(cov0)/numpy.linalg.det(cov1))) * 0.5
        #print("kld", kld)
    elif mode == "c":
        return (numpy.trace(cov1.I * cov0) - nof_variable - numpy.log( numpy.linalg.det(cov0)/numpy.linalg.det(cov1))) * 0.5
    elif mode == "m":
        return ((mu1 - mu0).T * cov1.I * (mu1 - mu0)) * 0.5
    else:
        return 0

def matrix_filter(matrix,name_list,white_list = [],black_list = []):
    '''
    return a filtered distance matrix and an element name list according to white list or black list:
    retain elements in the white list, remove elements in the black list.
    
    Keyword arguments:
        *matrix*: distance matrix of elements
        *name_list*: element name list of distance matrix
        *white_list*: white list of elements
        *black_list*: black list of elements
    '''
    matrix_dataframe = pandas.DataFrame(matrix, index = name_list, columns = name_list)
    name_dict = {}
    for index,key in enumerate(name_list):
        name_dict[key] = index
    if white_list != [] and black_list ==[]:
        filtered_names = white_list
    elif white_list ==[] and black_list != []:
        filtered_names = list(set(name_list) - set(black_list))
    elif white_list != [] and black_list !=[]:
        filtered_names = name_list[:]
    elif white_list == [] and black_list == []:
        filtered_names = name_list[:]
    matrix_dataframe = matrix_dataframe.loc[filtered_names,filtered_names]
    return matrix_dataframe.values, matrix_dataframe.columns

def draw_dendrogram(distance_matrix,field_names, mode = 'single', tick_fontsize = 15, tick_direction =  "v"):
    '''
    Draw dendrogram according to the distance matrix and corresponding element name list.
    
    Keyword arguments:
        *distance_matrix*: distance matrix
        *field_names*: a list of elements names
        *mode*: linkage method, four available method: single, complete,average or weighted
        *tick_fontsize*: The font size of ticks
        *tick_direction*: The direction of ticks, v for vertical, h for horizonal    
    '''

    cm = scipy.spatial.distance.squareform(distance_matrix)
    linkage = sch.linkage(cm, method= mode )
    fig = pyplot.figure(figsize=(8,8))
    ax1 = fig.add_axes([0.0,0.1,0.4,0.9]) #setup the layout of dendrogram
    arch_dict = sch.dendrogram(linkage, orientation='left')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.invert_yaxis()
    idx = arch_dict['leaves']
    m = distance_matrix[idx,:]
    m = m[:,idx]
    n = numpy.array(field_names)
    n = n[idx]    
    
    # Plot distance matrix.
    axmatrix = fig.add_axes([0.5,0.1,1.0,0.9]) 
    im = axmatrix.matshow(m, aspect='auto', origin='upper', cmap=pyplot.cm.YlGnBu)
    axcolor = fig.add_axes([0.5,0.0,1.0,0.05]) 
    fig.colorbar(im, cax=axcolor,orientation='horizontal')
    axmatrix.set_xticks(range(len(n)))
    axmatrix.set_xticklabels(n, minor= False, fontsize= tick_fontsize)
    axmatrix.set_yticks(range(len(n)))
    axmatrix.set_yticklabels(n, minor= False, fontsize= tick_fontsize)
    for tick in axcolor.xaxis.get_major_ticks():
        tick.label.set_fontsize(tick_fontsize*0.8) 
    if tick_direction == "v":
        for tick in axcolor.get_xticklabels():
            tick.set_rotation(90)
        for tick in axmatrix.get_xticklabels():
            tick.set_rotation(90)     
        for tick in axmatrix.get_yticklabels():
            tick.set_rotation(0)        
    fig.show()

def euclidean_HCA(dataframe,target_field,variable_fields,target_blacklist):
    '''
    Hierarchical cluster analysis using Euclidean distance between centroids of datasets
    
    Keyword arguments:
    *dataframe*: pandas dataframe, a column stands for geo-objects' names
    *target_field*: column name for geo-objects' names
    *variable_fields*: column name list for elements' names
    *target_blacklist*: blacklist for removing some geo-objects in HCA
    '''
    
    target_names = dataframe[target_field].unique()
    dataframe[variable_fields] = zscore(dataframe[variable_fields].values)
    nof_target = len(target_names)
    target_mask_dict = {}
    target_mean_dict = {}
    for target_name in target_names:
        mask = target_mask_dict[target_name] = (dataframe[target_field] == target_name).values
        data_array = dataframe[variable_fields].values[mask]
        mean_array = data_array.mean(axis = 0)
        mean_array = mean_array.reshape((1,mean_array.shape[0]))
        target_mean_dict[target_name] = mean_array
        #print("clr_array",clr_array)
    euclidean_dis_matrix = numpy.zeros((nof_target,nof_target))
    for i,j in combinations(range(nof_target),2):
        target_name1 = target_names[i]
        target_name0 = target_names[j]
        euclidean_dis = (((target_mean_dict[target_name0] - target_mean_dict[target_name1])**2).sum())**0.5
        euclidean_dis_matrix[i,j] = euclidean_dis_matrix[j,i] = euclidean_dis
    euclidean_dis_matrix_filtered,name_list = matrix_filter(euclidean_dis_matrix,target_names,black_list = target_blacklist)
    draw_dendrogram(euclidean_dis_matrix_filtered,name_list,mode = 'complete',tick_fontsize = 12,tick_direction =  "v")    


def atichison_HCA(dataframe,target_field,variable_fields,target_blacklist):
    '''
    Hierarchical cluster analysis using Atichison distance between centroids of datasets
    
    Keyword arguments:
    *dataframe*: pandas dataframe, a column stands for geo-objects' names
    *target_field*: column name for geo-objects' names
    *variable_fields*: column name list for elements' names
    *target_blacklist*: blacklist for removing some geo-objects in HCA
    '''
    
    target_names = dataframe[target_field].unique()
    nof_target = len(target_names)
    target_mask_dict = {}
    target_clr_dict = {}
    for target_name in target_names:
        mask = target_mask_dict[target_name] = (dataframe[target_field] == target_name).values
        data_array = dataframe[variable_fields].values[mask]
        mean_array = data_array.mean(axis = 0)
        mean_array = mean_array.reshape((1,mean_array.shape[0]))
        clr_array = clr(mean_array, residue = False, summation = 1000000)
        target_clr_dict[target_name] = clr_array
        print("clr_array",clr_array)
    aitchison_dis_matrix = numpy.zeros((nof_target,nof_target))
    for i,j in combinations(range(nof_target),2):
        target_name1 = target_names[i]
        target_name0 = target_names[j]
        aitchison_dis = (((target_clr_dict[target_name0] - target_clr_dict[target_name1])**2).sum())**0.5
        aitchison_dis_matrix[i,j] = aitchison_dis_matrix[j,i] = aitchison_dis
    aitchison_dis_matrix_filtered,name_list = matrix_filter(aitchison_dis_matrix,target_names,black_list = target_blacklist)
    draw_dendrogram(aitchison_dis_matrix_filtered,name_list,mode = 'complete',tick_fontsize = 12,tick_direction =  "v")        

def kld_HCA(dataframe,target_field,variable_fields,kld_mode = "a",target_blacklist = [" "]):
    '''
    Hierarchical cluster analysis using KL divergence between datasets
    
    Keyword arguments:
    *dataframe*: pandas dataframe, a column stands for geo-objects' names
    *target_field*: column name for geo-objects' names
    *variable_fields*: column name list for elements' names
    *target_blacklist*: blacklist for removing some geo-objects in HCA
    '''
    
    target_names = dataframe[target_field].unique()
    nof_target = len(target_names)
    target_mask_dict = {}
    target_cov_dict = {}
    target_mu_dict = {}
    for target_name in target_names:
        mask = target_mask_dict[target_name] = (dataframe[target_field] == target_name).values
        data_array = dataframe[variable_fields].values[mask]
        target_cov_dict[target_name] = numpy.cov(data_array.T)
        target_mu_dict[target_name] = numpy.mean(data_array,axis = 0)
    kld_matrix = numpy.zeros((nof_target,nof_target))
    
    for i,j in combinations(range(nof_target),2):
        target_name1 = target_names[i]
        target_name0 = target_names[j]
        cov0 = target_cov_dict[target_name0]
        cov1 = target_cov_dict[target_name1]
        mu0 = target_mu_dict[target_name0]
        mu1 = target_mu_dict[target_name1]
        kld0to1 = cal_kldivergence(cov0,cov1,mu0,mu1,mode = kld_mode)
        kld1to0 = cal_kldivergence(cov1,cov0,mu1,mu0,mode = kld_mode)
        kld = kld0to1 + kld1to0
        kld_matrix[i,j] = kld_matrix[j,i] = kld
    kld_matrix_filtered,name_list = matrix_filter(kld_matrix,target_names,black_list = target_blacklist)
    draw_dendrogram(kld_matrix_filtered,name_list,mode = 'complete',tick_fontsize = 12,tick_direction =  "v")
