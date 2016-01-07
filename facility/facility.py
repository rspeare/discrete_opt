#from collections import namedtuple
import numpy as np
import gurobipy as g
#Item = namedtuple("Item", ['index', 'value', 'weight'])
import pandas as pd
from haversine import haversine

class facility:
    def __init__(self):
        """
        Initializes the facility location instance. 
        Attributes:

        n :: Number of candidate facilities to be opened
        m :: Number of customers whose demands need to be met
        
        Functions:
        obj: returns the objective function,
              sum_{i,j} X_ij * D_ij+sum_{j} y_j s_j
              
            where D_ij is the translational distance matrix between
            facilities and customers and s_i is the start up cost for
            facility/factory j. The decision variables are the matrix X_ij
            and the y_j. Both of which are binary. 
        
        distance(x1,x2) :: return the euclidean distance between x1,x2
        write_outfile(file): Writes output_sol() to file
        output_sol() : writes the following facility/customer assignments
                to stdout. 
                
                objective_float optimal?_bool
                ass_0, ass_1, ... ass_M-1
                
            The assignment of all customers 1...M to one of the facilities 
            1...N. If a facility ID doesn't show up, it's not open!
        """
        self.n=0
        self.m=0
        self.cluster=False
        self.optimal=0
    def get_gifts(self):
        self.north_pole=[90.,0.]
        self.sleigh_weight=10.
        self.capacity=1000.-self.sleigh_weight
        path='/Users/rspeare/Dropbox/research/Machine Learning/kaggle/santa2015/data/gifts.csv'
        df=pd.read_csv(path)
        dfs=df.sort_values('Longitude')
        dfs.loc[:,'TripId']=dfs.loc[:,'GiftId']
        dfs.loc[:,'Capacity']=self.capacity
        return dfs
    def cluster_gifts(self,verbose,giftDF):
#       df=pd.read_csv(path)
       # Kaggle specific variables

       print('initializing santa data...')

       
       self.cluster=True
       # set demand as weight
       print("creating customer demands (gift weights)...")

#       self.d=df.Weight.values
#       self.xc=zip(df.Latitude.values,df.Longitude.values)
#       self.xf=self.xc
    #       np.vstack([df.Longitude.values,df.Latitude.values]).T
       
       #since we are clustering, any customer can be a factory
       self.n=len(giftDF)
       self.m=self.n
       
       # For memory efficiency, do not create data matrix
       print("creating distance matrix...")
       self.create_distance_matrix(x=giftDF[['Latitude','Longitude']].values)

       # Set start up -- fixed -- cost to be the distance 
       # to the centroid
       print("creating startup costs...")
       self.s=[haversine(xx,self.north_pole) for xx in giftDF[['Latitude','Longitude']].values]
       return self.gurobi_cluster(verbose,giftDF)
       # Uniform Capacity
#       self.c=np.ones(self.n)*self.capacity
    def gurobi_cluster(self,verbose,giftDF):
        self.model=g.Model("cluster")
        print('creating decision variables...')

        self.X = [[ self.model.addVar(vtype=g.GRB.BINARY,name='X_'+str(c)+'-'+str(f)) for f in np.arange(self.n)] for c in np.arange(self.m)]
        self.y = [ self.model.addVar(vtype=g.GRB.BINARY,name='y_'+str(f)) for f in np.arange(self.n)]
        self.model.update()
#        self.s=list(self.s)
#        self.c=list(self.c)
#        self.d=list(self.d)
        self.Dist=list(self.Dist)
        
        # objective
        # \sum_{c,f} X_[c,f]*D_[c,v]+ \sum_{f} s_[f]*y_[f]
        print('setting objective...')
#        dcosts=sum([self.X[c][f]*haversine(self.xc[c],self.xf[f]) for f in np.arange(self.n) for c in np.arange(self.m)])
        dcosts=g.quicksum([self.X[c][f]*self.Dist[c][f] for f in np.arange(self.n) for c in np.arange(self.m)])
#        dcosts=sum([self.X[c][f]*self.distance(self.xc[c],self.xf[f]) for f in np.arange(self.n) for c in np.arange(self.m)])
        print("cleaning up distance matrix")
        self.Dist=[]       

        scosts=g.quicksum([self.y[f]*self.s[f] for f in np.arange(self.n)])
#        print(dcosts)
#        print(scosts)
        self.model.setObjective(dcosts+scosts,g.GRB.MINIMIZE)

#        self.model.setObjective(sum([self.y[f]*self.s[f] for f in np.arange(self.n)]),g.GRB.MINIMIZE)
        print('setting constraints...')
        # Customer Must be served by SOMEONE
        for c in np.arange(self.m):
            self.model.addConstr(g.quicksum([self.X[c][f] for f in np.arange(self.n)])==1,str(c)+" served")
            
        # Entry in X matrix can never be greater than y
#        for f in np.arange(self.n):
#            for c in np.arange(self.m):
#                print('adding new constraint '+'X'+str(c)+'-'+str(f)+" valid")
#                self.model.addConstr(self.X[c][f] <= self.y[f],'X'+str(c)+'-'+str(f)+" valid")
            
        # Customer demand must not exceed assigned Facility capacity
        for f in np.arange(self.n):
#            self.model.addConstr(sum([self.X[c][f]*self.d[c] for c in np.arange(self.m)])<=self.c[f]*self.y[f],str(f)+" stocked")
            self.model.addConstr(g.quicksum([self.X[c][f]*giftDF.iloc[c]['Weight'] for c in np.arange(self.m)])<=giftDF.iloc[f]['Capacity']*self.y[f],str(f)+" stocked")

        
        #Quiet Mode
        self.model.setParam( 'OutputFlag', verbose )
        self.model.setParam(g.GRB.Param.MIPGap, 0.1)
        self.model.setParam(g.GRB.Param.TimeLimit, 20*60.0)
#        self.model.setParam(g.GRB.Param.Method,1)
        print('optimizing...')
#        self.model.setParam(g.GRB.Param.Threads,4)
        self.model.optimize()
        self.optimal=1
        # Variables are stored in row major order. First X, then y
        vars=self.model.getVars()
        self.ass=np.ones(self.m)*-1

        # set final values for activation variables
        for i in np.arange(self.n):
            self.y[i]=int(vars[self.m*self.n+i].x)
        self.y=np.array(self.y)

        # set final values for assignment variables
        i=0
        self.X=np.zeros((self.m,self.n))
        for c in np.arange(self.m):
            for f in np.arange(self.n):
                self.X[c,f]=int(vars[i].x)
                i+=1

#        self.ass=np.zeros(self.m)
#        for c in np.arange(self.m):
#            self.ass[c]=int(np.where(self.X[c]>0)[0][0])
#            giftDF.iloc[c]['TripId']=int(np.where(self.X[c]>0)[0][0])
#        giftDF.loc[:,'TripId']=[giftDF.iloc[int(np.where(self.X[c]>0)[0][0])]['GiftId'] for c in np.arange(self.m)]
        return [giftDF.iloc[int(np.where(self.X[c]>0)[0][0])]['GiftId'] for c in np.arange(self.m)]       
#        return giftDF

    def solve(self):
        self.model.optimize(my_callback)
    def obj(self):
        """
        return the objective function
        """
        return np.sum(self.Dist*self.X)+np.dot(self.y,np.array(self.s))

    def write_outfile(self,outfile):
        """
        writes the following facility/customer assignments
                to file. 
                
                objective_float optimal?_bool
                ass_0, ass_1, ... ass_M-1
                
            The assignment of all customers 1...M to one of the facilities 
            1...N. If a facility ID doesn't show up, it's not open!
        """
        
        file=open(outfile,'w')
        file.write(self.output_sol())
        file.close()

    def output_sol(self):
        """
        writes the following facility/customer assignments
                to file. 
                
                objective_float optimal?_bool
                ass_0, ass_1, ... ass_M-1
                
            The assignment of all customers 1...M to one of the facilities 
            1...N. If a facility ID doesn't show up, it's not open!
        """
        output = str(self.model.ObjVal) + ' ' + str(self.optimal) + '\n'
        for i in np.arange(self.m):
            output+=str(int(np.where(self.X[i,:]>0)[0][0]))+' '
        return output
    def hav(self,x1,x2):
        return haversine(x1,x2)
    def distance(self,x1,x2):
        """
        returns Euclidean distance between x1 and x2, both of which 
        can live in R^n. 
        """
        return np.linalg.norm(x1-x2)

    def create_distance_matrix(self,**kwargs):
        """
        Greedily creates a customer-->facility distance matrix.
        This code is not vectorized and could take a while. 
        """
        xx = kwargs.get('x',None)
        if (self.cluster):
            if (xx == None):
                self.Dist=np.zeros((self.n,self.n)).astype(np.float32)
#               xx=np.array(self.xc)
                for i in np.arange(self.n):
                    self.Dist[i,:]=np.linalg.norm(np.array(self.xc)[:,np.newaxis]-np.array(self.xc[i]),axis=-1).flatten()
#                for j in np.arange(self.n):
##                    self.Dist[i,j]=haversine(self.xc[i],self.xc[j])
#                    self.Dist[i,j]=np.linalg.norm(xx[i]-xx[j])
            else:
                print('passed x vector')
                self.Dist=np.zeros((len(xx),len(xx))).astype(np.float32)
                for i in np.arange(self.n):
                    self.Dist[i,:]=np.linalg.norm(xx[:,np.newaxis]-xx[i],axis=-1).flatten()
                
        else:
            self.Dist=np.zeros((self.m,self.n))
            for i in np.arange(self.m):
                for j in np.arange(self.n):
#                    self.Dist[i,j]=self.distance(self.xc[i],self.xf[j])
                    self.Dist[i,j]=np.linalg.norm(self.xc[i]-self.xf[j])

    def read_infile(self,infile):
        """
        read infile of the following format:
        
        N M 
        s_1 c_1 x_1 y_1
        ...
        s_N c_N x_N y_N
        d_1 x_1 y_1
        ...
        d_M x_M y_M
        
        Where s_i is the startup cost, c_i the capacity, <x_i,y_i> the 
        euclidean position of facility i. 
        """
        file=open(infile,'r')
        input_data = ''.join(file.readlines())
        self.read_input(input_data)
        file.close()

    def read_input(self,lines):
        """
        Read input of Type:
        
        N M
        s_i c_i x_i y_i
        ...
        s_N c_N x_N y_N
        d_j x_j y_j
        ...
        d_M x_M y_M
        
        So we have N factories and M customers to be served.
        s_i specifies the start up cost of factory i
        c_i specifies the capacity of factory i
        d_j is the demand of cusomer j
        x_i,y_i are cartesian coordinates
        """
        lines=lines.split('\n')
       
        header=lines[0].split()
        self.n,self.m=[int(x) for x in header]

        # Startup cost, capacity, euclidean position
        self.s=np.zeros(self.n)
        self.c=np.zeros(self.n)
        self.xf=np.zeros((self.n,2))
        # Demand, euclidean poition
        self.d=np.zeros(self.m)
        self.xc=np.zeros((self.m,2))
        # Decision Variables
        self.X = np.zeros((self.m,self.n))

        for i in np.arange(self.n):
            line=lines[i+1].split()
#            print('facility line: ',line)
            self.s[i],self.c[i],x,y=line
            self.xf[i]=[x,y]

        for i in np.arange(self.m):
            line=lines[i+1+self.n].split()
#            print('customer line: ',line)
            self.d[i],x,y=line
            self.xc[i]=[x,y]
        print(str(self.n)+' facilities, '+str(self.m)+' customers: '+str(self.m*self.n+self.n)+' VARS')
        print('creating distance matrix...')
        self.create_distance_matrix()

    def gurobi_solve(self):
        self.model=g.Model("facility")
        print('creating decision variables...')

        self.X = [[ self.model.addVar(vtype=g.GRB.BINARY,name='X_'+str(c)+'-'+str(f)) for f in np.arange(self.n)] for c in np.arange(self.m)]
        self.y = [ self.model.addVar(vtype=g.GRB.BINARY,name='y_'+str(f)) for f in np.arange(self.n)]
        self.model.update()
        self.s=list(self.s)
        self.c=list(self.c)
        self.d=list(self.d)
        self.Dist=list(self.Dist)

        # objective
        # \sum_{c,f} X_[c,f]*D_[c,v]+ \sum_{f} s_[f]*y_[f]
        print('setting objective...')
#        dcosts=sum([self.X[c][f]*haversine(self.xc[c],self.xf[f]) for f in np.arange(self.n) for c in np.arange(self.m)])
        dcosts=g.quicksum([self.X[c][f]*self.Dist[c][f] for f in np.arange(self.n) for c in np.arange(self.m)])
#        dcosts=sum([self.X[c][f]*self.distance(self.xc[c],self.xf[f]) for f in np.arange(self.n) for c in np.arange(self.m)])
        print("cleaning up distance matrix")
        self.Dist=[]

        scosts=g.quicksum([self.y[f]*self.s[f] for f in np.arange(self.n)])
        self.model.setObjective(dcosts+scosts,g.GRB.MINIMIZE)

#        self.model.setObjective(sum([self.y[f]*self.s[f] for f in np.arange(self.n)]),g.GRB.MINIMIZE)
        print('setting constraints...')
        # Customer Must be served by SOMEONE
        for c in np.arange(self.m):
            self.model.addConstr(sum([self.X[c][f] for f in np.arange(self.n)])==1,str(c)+" served")
        # Entry in X matrix can never be greater than y
        for f in np.arange(self.n):
            for c in np.arange(self.m):
#                print('adding new constraint '+'X'+str(c)+'-'+str(f)+" valid")
                self.model.addConstr(self.X[c][f] <= self.y[f],'X'+str(c)+'-'+str(f)+" valid")
        # Customer demand must not exceed assigned Facility capacity
        for f in np.arange(self.n):
            self.model.addConstr(sum([self.X[c][f]*self.d[c] for c in np.arange(self.m)])<=self.c[f]*self.y[f],str(f)+" stocked")
        
        #Quiet Mode
#        self.model.setParam( 'OutputFlag', False )
        print('optimizing...')
        self.model.setParam(g.GRB.Param.Threads,1)

#0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent.
        self.model.setParam(g.GRB.Param.Method,1) #dual simplex
#        self.model.setParam(g.GRB.param.MIPGap,10**(-6))
        self.model.optimize(my_callback)
#        self.solve()
        self.optimal=1
        # Variables are stored in row major order. First X, then y
        vars=self.model.getVars()
        self.ass=np.ones(self.m)*-1

        # set final values for activation variables
        for i in np.arange(self.n):
            self.y[i]=int(vars[self.m*self.n+i].x)
        self.y=np.array(self.y)

        # set final values for assignment variables
        i=0
        self.X=np.zeros((self.m,self.n))
        for c in np.arange(self.m):
            for f in np.arange(self.n):
                self.X[c,f]=int(vars[i].x)
                i+=1

#        self.ass=np.zeros(self.m)
        for c in np.arange(self.m):
            self.ass[c]=np.where(self.X[c]>0)[0][0]
def get_X_from_vars(vars,m,n):
    i=0
    X=np.zeros((m,n))
    for c in np.arange(m):
        for f in np.arange(n):
            X[c,f]=int(vars[i].x)
            i+=1
    return X
    
def get_y_from_vars(vars,m,n):
    y=np.zeros(n)
    for i in np.arange(n):
        y[i]=int(vars[m*n+i].x)
    y=np.array(y)
    return y    

def my_callback(model,where):
    """
    template callback function
    """
    if ((where == g.GRB.Callback.MIPNODE) or (where == g.GRB.Callback.MIPSOL)):
#    if where == g.GRB.Callback.MIPSOL:
        X = model.cbGetSolution(model._vars[:m*n])
        y = model.cbGetSolution(model._vars[m*n:])
        print(y)
#        time=model.cbGet(g.GRB.RUNTIME)
#        if (time > 100):
#            print(time)
#            break
        pass


#        print(vars)    
        
def my_callback2(model,where):
    """
    template callback function
    """
    if where == g.GRB.Callback.POLLING:
        pass
    elif where == g.GRB.Callback.MIPSOL:
        vars=model.cbGetSolution(model.getVars())
#        print(vars)

import sys
#import knapsack
if __name__ == '__main__':
    if len(sys.argv)>1:
        print('reading infile')
        f=facility()
        infile = sys.argv[1].strip()
        f.read_infile(infile)
        f.gurobi_solve()
        f.output_sol()
    else:
        print('This test rquires an input file on the command line')
        
