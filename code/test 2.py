# -*- coding: utf-8 -*-

# DO NOT CHANGE
import pandas as pd
import numpy as np

def get_order(structure):
    
    new_list = list()
    
    for key, value in structure.items():
        if not value:
            new_list.append(key)
            
    for key, value in structure.items():
        for attribution in value:
            if attribution in new_list and key not in new_list:
                new_list.append(key)
                
    for key, value in structure.items():
        for attribution in value:
            if attribution not in new_list:
                new_list.append(attribution)
                
    for key, value in structure.items():
        if key not in new_list:
            new_list.append(key)
    
    return new_list
    

def learn_parms(data,structure,var_order):
       
    dict_ = {}
    for var in var_order:
        matrix = []
        key_list = []
        
        if len(structure[var]) == 0:
            df1 =  pd.DataFrame(list(data[var].value_counts(normalize=True))).T
            df1.columns = np.unique(data[var])
            dict_[var] =  df1
            continue ;
        
        groups = data.groupby(structure[var])
        
        for key, group in groups:
            matrix.append(list(group[var].value_counts(normalize=True)))
            df2 = pd.DataFrame(matrix)
            df2.columns = np.unique(data[var])
            key_list.append(key)
        
        df2.index = key_list
        dict_[var] = df2
        
    return dict_
    
                
def print_parms(var_order,parms):
    
    for var in var_order:
        print('-------------------------')
        print('Variable Name=%s'%(var))
        print(parms[var])
        #TODO: print the trained paramters
        
    
data=pd.read_csv('https://drive.google.com/uc?export=download&id=1taoE9WlUUN4IbzDzHv7mxk_xSj07f-Zt', sep=' ')

str1={'A':[],'S':[],'E':['A','S'],'O':['E'],'R':['E'],'T':['O','R']}
order1=get_order(str1)
parms1=learn_parms(data,str1,get_order(str1))
print('-----First Structure------')
print_parms(order1,parms1)
print('')

str2={'A':['E'],'S':['A','E'],'E':['O','R'],'O':['R','T'],'R':['T'],'T':[]}
order2=get_order(str2)
parms2=learn_parms(data,str2,get_order(str2))
print('-----Second Structure-----')
print_parms(order2,parms2)
print('')
