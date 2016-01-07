from collections import namedtuple
import numpy as np
import gurobipy as g
Item = namedtuple("Item", ['index', 'value', 'weight'])

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
        output = str(self.obj()) + ' ' + str(0) + '\n'
        for i in np.arange(self.m):
            output+=str(int(np.where(self.X[i,:]>0)[0][0]))+' '
        return output

    def distance(self,x1,x2):
        """
        returns Euclidean distance between x1 and x2, both of which 
        can live in R^n. 
        """
        return np.linalg.norm(x1-x2)

    def create_distance_matrix(self):
        """
        Greedily creates a customer-->facility distance matrix.
        This code is not vectorized and could take a while. 
        """
        self.Dist=np.zeros((self.m,self.n))
        for i in np.arange(self.m):
            for j in np.arange(self.n):
                self.Dist[i,j]=self.distance(self.xc[i],self.xf[j])

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
        dcosts=sum([self.X[c][f]*self.Dist[c][f] for f in np.arange(self.n) for c in np.arange(self.m)])
        scosts=sum([self.y[f]*self.s[f] for f in np.arange(self.n)])
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
        self.model.setParam( 'OutputFlag', False )
        print('optimizing...')
#        self.model.setParam(g.GRB.Param.Threads,4)
        self.model.optimize(my_callback)

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
def my_callback(model,where):
    if where == g.GRB.Callback.POLLING:
        pass
    elif where == g.GRB.Callback.MIPSOL:
        vars=model.cbGetSolution(model.getVars())
        print(vars)

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
        
