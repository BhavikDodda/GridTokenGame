from math import *
def sumDecom(n):
    List=[]

    for i in range(0,ceil(n/2)+1):
        if(n-i>=i):
            List.append([i,n-i])
    return List

def insertCombInConfig(comb,config,index):
    possibility=config[:index]+comb+config[index+1:]
    possibility=sorted(list(filter(lambda x: x > 0, possibility)))
    return possibility

def nextPossibilities(config):
    possibilities=[]
    for index,component in enumerate(config):
        print("main",index)
        componentCombs=(sumDecom(component-1))
        for comb in componentCombs:
            possibility=insertCombInConfig(comb,config,index)
            possibilities.append(possibility)
        componentCombs=(sumDecom(component-2))
        for comb in componentCombs:
            possibility=insertCombInConfig(comb,config,index)
            possibilities.append(possibility)

    return possibilities

Fset=[]
NFset=[[]]
Fstrategies=[]
#print(nextPossibilities([3,2,1]))

def F_or_NF(config):
    if config in Fset:
        return "Fset"
    if config in NFset:
        return "NFset"
    foundalready=0
    for index,component in enumerate(config):
        componentCombs=(sumDecom(component-1))
        for comb in componentCombs:
            possibility=insertCombInConfig(comb,config,index)
            if(possibility not in Fset):
                if(possibility not in NFset):
                    F_or_NF(possibility)
            if(possibility in NFset):
                if(config not in Fset):
                    Fset.append(config)
                    Fstrategies.append((index,[comb[0]]))
                    foundalready=1
        componentCombs=(sumDecom(component-2))
        for comb in componentCombs:
            possibility=insertCombInConfig(comb,config,index)
            if(possibility not in Fset):
                if(possibility not in NFset):
                    F_or_NF(possibility)
            if(possibility in NFset):
                if(config not in Fset):
                    Fset.append(config)
                    Fstrategies.append((index,[comb[0],comb[0]+1]))
                    foundalready=1
    if foundalready:
        return "Fset"
    if (config not in NFset):
        NFset.append(config)
        return "NFset"

determine=F_or_NF([9])
print("ans",determine)
print("Fset")
print(Fset)
print("Fstrategies")
print(Fstrategies)
print("NFset")
print(NFset)