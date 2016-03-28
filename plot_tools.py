
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import datetime
import json
import itertools
import glob
import copy
from uncertainties import ufloat


# In[2]:

#create a class
class EspressoRun :
    def __init__(self,filename=None,name=None,normalizeOpenMP=False) :
        if filename != None :
            assert type(filename) == str
            self.filename=filename
            self.name= filename if name == None else name
            self.parseHeader()
            self.df = pd.read_csv(filename, comment='#', header=0, skiprows=self.skipLines)
            if self.header["threadsPerMPI"] != 1.0 and normalizeOpenMP :
                print "normalizing"
                self.df.cpuTime = self.df.cpuTime / self.header["threadsPerMPI"]
            #if Parent is null for BODY section it means the function was called in main
            self.df.ix[ pd.isnull(self.df.Parent) & (self.df.Section == "BODY"), 'Parent' ] = 'main'
            #pd.read_csv()
    #         self.df.cpuTime = self.df.cpuTime.astype(np.int64)
        else :
            self.filename = ""
            self.name = ""
            self.header=dict()
        
    def parseHeader(self) :
        #header file is a json
        headerStr=""
        theFile = open(self.filename,'r')
        self.skipLines=1
        for line in theFile :
            
            if ">>> BENCH BEGIN" in line :
                break
            headerStr += line
            self.skipLines +=1
        theFile.close()
        self.header=json.loads(headerStr)
        self.header=self.header['header']

            
    def plotIterationStats(self,fig = None):
        """Plot iterations cpuTime and their distribution distribution"""
        fig = plt.figure() if fig == None else fig
        title = "MPI: %s - OMP: %s" % (self.header["MPIProcs"],self.header["threadsPerMPI"])
        fig.suptitle=(title)
        
        ax1 = plt.subplot(2,2,1);
        ax1.set_title(title + " time per iteration")
        ax1.set_xlabel("Iteration number")
        ax1.set_ylabel("cpu time (s)")
        
        
        ax2 = plt.subplot(2,2,2);
        ax2.set_title(title + " time per iteration distribution")
        ax2.set_xlabel("cpu time (s)")
        
        
        left = range(1,len(self.header["iterations"]) +1)
        height = [i['timePerIter'] for i in self.header["iterations"]]
        ax1.bar(left,height);
        ax2.hist(height,len(height)/4)
        
    def getMeanIterationTime(self):
        return np.mean([i['timePerIter'] for i in self.header['iterations']])
        
    def getStdIterationTime(self):
        return np.std(map(lambda x : x['timePerIter'],self.header['iterations']))
        
        
        
        
            
    
    def computeParentPercent(self):
        asd = self.df.groupby(['Parent','Section']).aggregate(sum)
        asd = self.df[self.df.Section == "BODY"].groupby('Parent').aggregate(sum)
        asd.rename(columns={'cpuTime':'parentCpuTime','wallTime':'parentWallTime','Calls':'parentCalls'},inplace=True)

        self.df = pd.merge(self.df , asd , how='left', right_index=True, left_on='Parent')

        self.df['wallTime_parentPercent']=np.nan
        self.df['cpuTime_parentPercent']=np.nan
        self.df['calls_parentPercent']=np.nan

        self.df.cpuTime_parentPercent = self.df.cpuTime / self.df.parentCpuTime * 100
        self.df.wallTime_parentPercent = self.df.wallTime / self.df.parentWallTime * 100
        self.df.calls_parentPercent = self.df.Calls / self.df.parentCalls * 100
    
    def getParentDatas(self,parent) :
        return self.df[ self.df.Parent == parent]
        
    def getPieOfParent(self,parent,metric ='cpuTime_parentPercent'):
        toPlot = self.getParentDatas(parent)[['name',metric]].set_index('name') 
        plt.figure(figsize=(6,6))
        return plt.pie(toPlot,labels = toPlot.index,startangle=90)
    
    def getBodyFigure(self) :
        datas = self.df
        theParents = datas[datas.Section == "BODY"].Parent.unique()
        subplotGridHight = len(theParents)/2 + len(theParents)%2
        fig = plt.figure(figsize=(18,18))
        k = 0
        h = 0
        fig.suptitle("Profiling", fontsize=14, fontweight='bold')
        for parent in theParents:
            #print parent
            #print k,h
            ax = plt.subplot2grid((subplotGridHight,2), (k,h))
            h = (h+1)%2
            if h%2 == 0 : k = k+1

            toBarPlot = datas[(datas.Parent == parent) & (datas.Section == "BODY")][['name','Parent','cpuTime_parentPercent','wallTime_parentPercent','calls_parentPercent']]

            #sort by most cpu demanding
            toBarPlot.sort_values('cpuTime_parentPercent',ascending = False, inplace=True)

            cpuOffsets = [0]
            wallOffsets = [0]
            callsOffsets = [0]
            for i,j,z in zip(toBarPlot.cpuTime_parentPercent,toBarPlot.wallTime_parentPercent,toBarPlot.calls_parentPercent) :
                cpuOffsets.append(cpuOffsets[len(cpuOffsets)-1] + i)
                wallOffsets.append(wallOffsets[len(wallOffsets)-1] + j)
                callsOffsets.append(callsOffsets[len(callsOffsets)-1] + z)
            cpuOffsets = cpuOffsets[:-1]
            wallOffsets = wallOffsets[:-1]
            callsOffsets = callsOffsets[:-1]

            cmap = plt.cm.jet
            zipped = zip(toBarPlot['name'],
                         toBarPlot.cpuTime_parentPercent,toBarPlot.wallTime_parentPercent,toBarPlot.calls_parentPercent,
                         cpuOffsets,wallOffsets,callsOffsets,
                         range(0,len(toBarPlot['name'])))
            #print in reverse order because we wnat the bottom stack to be the least in the legend
            #unfortunately managing the legend order is not so easy
            for name,cpu,wall,calls,cpuO,wallO,callsO,nth in reversed(zipped):
                plt.bar([1,3,5], #position of the bar
                       [cpu,wall,calls],#height
                       width=1,
                       bottom=[cpuO,wallO,callsO],
                        color = cmap(1- (float(nth)/len(toBarPlot['name']))),
                        label = name 
                      )

            plt.xlim(0,10)
            plt.ylim(0,110)
            plt.title(parent)
            plt.xticks([1.5,3.5,5.5],['cpu time','wall time','calls'])
            plt.legend()
        return fig
    
    def getGlobalStackedPlot(self,savefile="") :
        #if False if 'Condensed' in self.header.keys() else self.header['Condensed'] :
         #   return self.df[['name','cpuTime','cpuTimeStd','wallTime','wallTimeStd']].set_index('name').plot(kind='bar',figsize=(10,6),yerr=['cpuTimeStd','wallTimeStd'])
        #else:
            toRet =  self.df[['name','cpuTime','wallTime']].set_index('name').plot(kind='bar',figsize=(10,6))
            if savefile != "" : 
                plt.savefig(savefile)
            return toRet;
    
    def barplot(self,dataframe,metrics=['cpuTime'],**kwargs) :
        return  dataframe[['name']+metrics].set_index('name').plot(kind='bar',**kwargs)
            


# In[3]:

"""
Organized collection of espresso runs
"""
def timeTicks(x, pos):                                                                                                                                                                                                                                                         
    d = datetime.timedelta(milliseconds=x)                                                                                                                                                                                                                                          
    return str(d)

def secondsTicks(x, pos):
    return '%.0e' % (x/1000)
    #return str(x/1000)

class Experiment :
    def __init__(self) :
        self.runs = []
    def addRun(self,espressoRun) :
        self.runs.append(espressoRun)
        
        
    def plotHeaderAttribute(self,attributeName='numOfIterations',orderBy='threadsPerMPI', perCore = False, figure=None ,axes= None,
                            labels = None , ylabel = None):
        fig = plt.figure() if figure == None else figure
        ax = fig.add_subplot(111) if axes == None else axes
        labels = attributeName if labels == None else labels;
        ylabel = attributeName if ylabel == None else ylabel
        
        #get attribute list
        attrList = [ i.header[attributeName] for i in self.runs ] if perCore == False else [ 1.0 * i.header[attributeName]/i.header["totCores"] for i in self.runs ]
        xlist = [ i.header[orderBy] for i in self.runs ]
        
        
        plt.plot(xlist,attrList,marker = 'o');
        
        
        
        #shrink axis 20% and put the legend outside
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    def plotIterationStats(self):
        for i in self.runs :
            i.plotIterationStats(fig=plt.figure(figsize=(15,10)))
        
            
    def plotFunction(self,functions=['PWSCF'],labels=None,metric='cpuTime',
                     ylabel=None,orderBy='totCores',figure=None, axes=None, 
                     ylog=False,ylogBase=10,returned=False, title=None,speedup=False,
                     rescaleIterations=False,
                     legend=True,
                     useTimeFormatter=True,
                     yformatter='time'
                    ):
        #data = [{'index':index,'value':run.df[run.df.name == functionName][metric].values[0]} for run,index in zip(self.runs,range(len(self.runs))) ]
        fig = plt.figure() if figure == None else figure
        ax = fig.add_subplot(111) if axes == None else axes
        labels = functions if labels == None else labels;
        
        ylabel = metric if ylabel == None else ylabel
        data=[]
        for run,index in zip(self.runs,range(len(self.runs))) :
            toAppend =  dict(index=index)
            values = []
            for functionName in functions :
                valToAppend = run.df[run.df.name == functionName][metric].values[0] if len(run.df[run.df.name == functionName][metric].values) > 0 else -1
                values.append(valToAppend)
            toAppend['values']=values
            data.append(toAppend)
            
        #order by could be or in the header or a column of the df
        if orderBy in self.runs[0].header.keys() :
            for i,pos in zip(data,range(len(data))) :
                data[pos]['orderBy']= self.runs[i['index']].header[orderBy]
                
        dataSorted = sorted(data,key=lambda x : x['orderBy'])
#         print dataSorted
        
        marker = itertools.cycle(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd'))
        
        for functionName,functionLabel,index in zip(functions,labels,range(len(functions))) :
            ys=[ i['values'][index] for i in dataSorted]
            if speedup :
                ys = [ float(ys[0])/i for i in ys]

            if speedup :
                left = [i['orderBy'] for i in dataSorted]
            else:
                left = range(1,2*len(dataSorted)+1,2)
                
            toDelete = [ pos for y,pos in zip(ys,range(len(ys))) if y < 0]
            
            ys = [y for y,pos in zip(ys,range(len(ys))) if pos not in toDelete]
            left = [l for l,pos in zip(left,range(len(left))) if pos not in toDelete]
            if (len(left) == 0) : continue

#             print toDelete , ys
            if ylog :
                ax.semilogy(left,ys,label=functionLabel, marker = marker.next(), basey=ylogBase)
            else :
                ax.plot(left,ys,label=functionLabel, marker = marker.next())
        
        xticks = range(1,2*len(dataSorted)+1,2)
        xticklabels = [ i['orderBy'] for i in dataSorted]
        
        if speedup :
            xticks = [ i['orderBy'] for i in dataSorted ]
#             ax.plot([0,xticks[-1]],[0,xticks[-1]])
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(orderBy)
        ax.set_xlim(0,xticks[-1]+1)
        #plt.xticks(xticks,xticklabels)
        #plt.xlabel(orderBy)
        #plt.xlim(0,xticks[-1]+1)
        if speedup :
            ax.set_ylabel('Speedup')
        else :
            ax.set_ylabel(metric)
        if title != None :
            ax.set_title(title)
        
        if not speedup and metric != 'calls' and yformatter != 'default':
            if yformatter == 'time' :
                formatter =  ticker.FuncFormatter(timeTicks)
            elif yformatter == 'seconds' :
                formatter =  ticker.FuncFormatter(secondsTicks)                
            ax.yaxis.set_major_formatter(formatter) 
            
        if legend :
        #shrink axis 20% and put the legend outside
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        if returned :
            return fig
        
        
    def plotParents(self,runIndex=0,**kwargs):
        return self.plotFunction(functions=self.getParentsToPlot())
    
    def getParentsToPlot(self):
        return [ 'PWSCF' if i == 'main' else i for i in self.runs[0].df.Parent.unique()  ]
    
    def condense(self) :
        union = pd.concat([ run.df for run in self.runs])
        grouped = union[['name','Section','cpuTime','wallTime','Calls']].groupby(['name','Section'],sort=False)
#         for name,group in grouped :             
#             print name ,np.std(group.cpuTime)
        #Applying different functions to DataFrame columns < CERCA STA ROBA
        newDf=grouped.aggregate(np.mean).reset_index()
        std=grouped.aggregate(np.std).reset_index()


        #print grouped.get_group(('init_run','BODY'))
        #print np.std(grouped.get_group(('init_run','BODY')).wallTime)
        toRet = EspressoRun()
        newDf['Parent'] = self.runs[0].df.Parent
        newDf = newDf[['name','cpuTime','wallTime','Calls','Parent','Section']]
        toRet.df = newDf
        toRet.computeParentPercent()
        toRet.df['cpuTimeStd'] = std['cpuTime']
        toRet.df['wallTimeStd'] = std['wallTime']
        toRet.df['CallsStd'] = std['Calls']
        toRet.headers=[ run.header for run in self.runs]
        toRet.header = dict(self.runs[0].header)
        toRet.header['Condensed'] = True
        
        return toRet
        
    def slowFunctions(self,metric='cpuTime'):
        union = pd.concat([ run.df for run in sorted(self.runs,key=lambda x : x.header['totCores']) ])
        grouped = union[['name','Section',metric]].groupby(['name','Section'],sort=False)
        toRet = []
        for (name,section), group in grouped:
#             print name
#             print group
#             print "asd"
            for cpuTime,pos in zip(group[metric][1:],range(len(group[metric][1:]))) :
                if cpuTime > group[metric].values[pos] :
                    toRet.append(name)
                    break
        
        return toRet
    
    def getRescaleRatios(self,metric = 'cpuTime', min_iters = 0 ):
        min_iters =  min(map(lambda x : len(x.header['iterations']) , self.runs )) if min_iters == 0 else min_iters

        ratios=[]
        for run in self.runs :
            realTime = float(run.df[run.df.name == 'PWSCF'][metric])
            #in realta' il tempo delle itarazioni e' in wallTime non in cpu time
            #quindi con i threads vai in scioltezza
            #l'hai controllata mille volte, fidati
            lastIterTime = 1000.* run.header['iterations'][min_iters - 1 ]['endCpuTime']
            ratio = (1. * lastIterTime / realTime)
            ratios.append(ratio)
            
        return ratios
    
    def rescale(self,**kwargs):
        rescaled = copy.deepcopy(self)
        for run,cpuRatio,wallRatio in zip(rescaled.runs,self.getRescaleRatios('cpuTime',**kwargs),
                                          self.getRescaleRatios('wallTime',**kwargs)) :
            run.df['cpuTime'] = run.df['cpuTime']*cpuRatio
            run.df['wallTime'] = run.df['wallTime']*wallRatio
            
        return rescaled
    
    def getIterationDF(self,metric='totCores'):
        sourceDict={'mean':[],'std':[],'stdPerc':[],'n':[] }
        index = []
        for run in self.runs :
            #tio64.getMeanIterationTime(),tio64.getStdIterationTime(),tio64.getStdIterationTime()/tio64.getMeanIterationTime(),len(tio64.header['iterations'])
            sourceDict['mean'].append(run.getMeanIterationTime())
            sourceDict['std'].append(run.getStdIterationTime())
            sourceDict['stdPerc'].append(run.getStdIterationTime()/run.getMeanIterationTime())
            sourceDict['n'].append(len(run.header['iterations']))
            index.append(str(run.header[metric]))
        
        
        return pd.DataFrame(sourceDict,index=index,columns=['mean','std','stdPerc','n'])

    def plotIterationTrend(self,figure=None,axes=None,ylog=False,speedup=False,zeroErrorOnOne=False,axesTitle=None,label=None,
                          metric = 'totCores',lineFormat='-o',errorColor='g'):
        xticks=list(self.getIterationDF(metric).index)
        xs = range (1, len(xticks)+1)
        
        #fig, ax = plt.subplots(nrows=1, ncols=1)
        fig = plt.figure() if figure == None else figure
        ax = fig.add_subplot(111) if axes == None else axes
        
        ys = list(self.getIterationDF(metric)['mean'])
        ye = list(self.getIterationDF(metric)['std'])
        
        if speedup:
            #for error algebra
            uncert=np.array([ ufloat(val,sig) for val,sig in zip(ys,ye) ])
            out = uncert[0]/uncert
            
            
            ys = [x.n for x in out]
            ye = [abs(x.s) for x in out]
            if not zeroErrorOnOne :
                ratio = uncert[0]/ufloat(uncert[0].n,uncert[0].s)
                ye[0] = ratio.s#uncert[0].s/uncert[0].n
            

        if ylog :
            ax.set_yscale('log')
            
        ax.set_xlim(left=xs[0]-0.5,right=xs[len(xs)-1]+0.5)
        plt.xticks(xs,xticks)
        
        ax.set_title(None if axesTitle == None else axesTitle)
        
        
        plotlabel = "mean iteration time" if label == None else label
        
        ax.set_ylabel("speedup" if speedup else "seconds")
        ax.set_xlabel(metric)
        
        #print plotlabel    
        ax.errorbar(x=xs,y=ys,yerr=ye,fmt=lineFormat,ecolor=errorColor,label=plotlabel)
        
        legendLocation = 'upper right' if not speedup else 'upper left'
        ax.legend(loc = legendLocation)
        
    def getIterationsNumPerRun(self) :
        return map(lambda x : (x.header['numOfIterations'],x.header['totCores']) , self.runs)
    
    def getMinIterations(self) :
        return min (map ( lambda x : x[0] ,self.getIterationsNumPerRun()))
    
    def getValuesForFunction(self,function = 'PWSCF', metric = 'cpuTime'):
        cpuTimes = map (lambda x : float(x.df[x.df.name == function][metric]),self.runs )
        ranks = map (lambda x : x.header['totCores'], self.runs)
        
        return (ranks,cpuTimes)
        
        
        
    


# In[4]:

def condenseFolder(folder,csvFileName,extraHeaderField=None):
    exp = Experiment()
    for folder in glob.glob(folder+'/try[0-9]*'):
            exp.addRun(EspressoRun(folder+'/'+csvFileName))

    toRet = exp.condense()
    if extraHeaderField != None : 
        toRet.header[extraHeaderField[0]] = extraHeaderField[1]
    return toRet

    


# In[ ]:



