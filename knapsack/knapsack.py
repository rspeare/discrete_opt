from collections import namedtuple
import numpy as np
import gurobipy as g
Item = namedtuple("Item", ['index', 'value', 'weight'])

class knapsack:
    def __init__(self):
        """
        Initializes the knapsack instance
        """
        self.n=0
        self.K=0
        self.tuple_data=False
        self.optimal=0
    def obj(self):
        return np.dot(self.v,self.x)

    def write_outfile(self,outfile):
        obj=np.dot(self.v,self.x)
        file=open(outfile,'w')
        file.write(str(obj)+' '+str(self.optimal)+'\n')
        for i in np.arange(self.n):
            file.write(str(int(self.x[i]))+' ')
        file.close()

    def output_sol(self):
        obj=int(np.dot(self.v,self.x))
        output=str(obj)+' '+str(self.optimal)+'\n'
        for i in np.arange(self.n):
            output+=str(int(self.x[i]))+' '
        output+='\n'
        return output

    def read_infile(self,infile):
#        data=np.loadtxt(infile)
#        self.n,self.K=data[0,:]
#        self.v=data[1:,0]
#        self.w=data[1:,0]
        file=open(infile,'r')
        header=file.readline().split()
        self.n,self.K=[int(x) for x in header]
        if (self.tuple_data):
            self.items=[]
            for i in np.arange(self.n):
                line=file.readline().split()
                self.items.append(Item(i,int(line[0]),int(line[1])))
        else:
            self.w = np.zeros(self.n)
            self.v = np.zeros(self.n)
            self.x = np.zeros(self.n)
            for i in np.arange(self.n):
                line=file.readline().split()
                self.v[i],self.w[i]=float(line[0]),float(line[1])
    def greedy1(self,infile):
        """
        Store items in the bag, without sorting until reached capacity.
        """
        self.read_infile(infile)
        value=0
        weight=0
        i=0
        while (weight <= self.K):
            # propose add
            weight+=self.w[i]
            if (weight <= self.K):
                self.x[i]=1
                value+=self.v[i]
            else:
                break
            i+=1
        self.write_outfile(infile+'_greedy1_sol')

    def greedy2(self,infile):
        """
        Store items in the bag, after sorting by value,
        until reached capacity.
        """
        self.read_infile(infile)
        value=0
        weight=0
        i=0
        index=np.argsort(self.v)
        for i in index:
            # propose add
            weight+=self.w[i]
            if (weight <= self.K):
                self.x[i]=1
                value+=self.v[i]
            else:
                break
        self.write_outfile(infile+'_greedy2_sol')
    def gurobi_solve(self):
        self.m=g.Model("knapsack")
        self.x =[self.m.addVar(vtype=g.GRB.BINARY, name="x"+str(i)) for i in np.arange(self.n)]
        self.m.update()
        self.w = list(self.w)
        self.v = list(self.v)
        self.m.setObjective(g.quicksum([self.v[i]*self.x[i] for i in np.arange(self.n)]), g.GRB.MAXIMIZE)
        self.m.addConstr(g.quicksum([self.w[i]*self.x[i] for i in np.arange(self.n)]) <= self.K, "c0")
#        self.m.setParam(g.GRB.param.MIPGap,10**(-8))

        self.m.optimize(my_callback)
        self.optimal=1

        v=self.m.getVars()
        for i in np.arange(self.n):
            self.x[i]=int(v[i].x)

    def gurobi(self,infile):
        self.read_infile(infile)
        self.m=g.Model("knapsack")
        self.x =[self.m.addVar(vtype=g.GRB.BINARY, name="x"+str(i)) for i in np.arange(self.n)]
        self.m.update()
        self.w = list(self.w)
        self.v = list(self.v)
        self.m.setObjective(sum([self.v[i]*self.x[i] for i in np.arange(self.n)]), g.GRB.MAXIMIZE)
        self.m.addConstr(sum([self.w[i]*self.x[i] for i in np.arange(self.n)]) <= self.K, "c0")
        self.m.optimize()
        self.optimal=1

        v=self.m.getVars()
        for i in np.arange(self.n):
            self.x[i]=int(v[i].x)
        self.write_outfile(infile+'_gurobi_sol')
def my_callback(model,where):
    """
    template callback function
    """
    if where == g.GRB.Callback.POLLING:
        pass
    elif where == g.GRB.Callback.MIPSOL:
        #        vars=model.cbGetSolution(model.getVars())
        optimal=1
        return optimal
    elif where == g.GRB.Callback.MESSAGE:
        optimal=0
        return optimal

import sys
#import knapsack
if __name__ == '__main__':
    if len(sys.argv)>1:
        print('reading infile')
        sack=knapsack()
        infile = sys.argv[1].strip()
        sack.greedy2(infile)
        print('solved')

    else:
        print('This test rquires an input file on the command line')
        
