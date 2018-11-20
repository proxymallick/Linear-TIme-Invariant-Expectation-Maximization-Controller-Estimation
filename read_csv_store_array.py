
import numpy as np
from scipy import *
import csv

def readcsv(filename):	
    ifile = open(filename, 'rU')
    reader = csv.reader(ifile, delimiter=",")
    
    rownum = 0	
    a = []#np.zeros((495,14))
    #a.append(reader)
    for row in reader:
        a_new=[float(i) for i in row]
        #print a_new
        a.append (a_new)
        rownum += 1
    
    ifile.close()
    return a 

policy_K_file=readcsv('/home/prakash/gps/python/gps/K_policy.csv')
np_arr_K=np.array(policy_K_file)
print np_arr_K.shape