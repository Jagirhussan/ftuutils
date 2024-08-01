import numpy as np
import networkx as nx
import sympy

from ftuutils.compositionutils import SymbolicPHS,Composer
'''
Logic to edit the topology of a FTU. 
'''


class ReducePHS():
    """
    Instance to reduce the size of the FTU based on clustering some of its nodes together
    """
    
    def __init__(self,compositePHSWithStructure):
        """Setup a composite PHS for reduction

        Args:
            compositePHSWithStructure (dict): Dictionary containing the composite SymbolicPHS as json string
            and its structural organisation in terms of subcomponents
        """
        if isinstance(compositePHSWithStructure,str):
            self.compositePHS = SymbolicPHS.loadPHSDefinition(compositePHSWithStructure['compositePHS'])
            self.phsStructure = compositePHSWithStructure['compositePHSStructure']
        elif isinstance(compositePHSWithStructure,Composer):
            self.compositePHS = compositePHSWithStructure.compositePHS
            self.phsStructure = {'type':compositePHSWithStructure.phstypes,
                                 'rowidxs':compositePHSWithStructure.ridxs,
                                 'colidxs':compositePHSWithStructure.cidxs,
                                 'phsstructure':compositePHSWithStructure.phsclassstructure}
        else:
            raise Exception(f"Input from {type(compositePHSWithStructure)} not supported!")
        
    def getPHSAssignments(self):
        return self.phsStructure['type']
        
    def setClusters(self,nodegroups):
        """Set the cluster/grouping of nodes in the phs
        {clusternum:[phs nums]}
        Preliminary checks are done to ensure the group elements are of the same PHS type
        and the cardinality and unitary cluster membership are accurate
        Args:
            nodegroups (dict): Cluster assignment of nodes
        """
        #Check membership
        phstypes  = self.phsStructure['type']
        phstructure=self.phsStructure['phsstructure']
        maxelements = len(phstypes)
        nelem = 0
        elementids = []
        for n,v in nodegroups.items():
            nelem += len(v)
            pt = phstypes[v[0]]
            for vv in v[1:]:
                if pt != phstypes[vv]:
                    raise Exception(f"cluster {n} has multiple PHS types {pt} and {phstypes[vv]}")                    
            elementids.extend(v)
            
        if nelem!=maxelements:
            raise Exception(f"cluster assignment uses more elements {nelem} than in PHS {maxelements}")

        for n,v in nodegroups.items():
            for n1,v1 in nodegroups.items():
                if n != n1:
                    if bool(set(v) & set(v1)):
                        raise Exception(f"cluster {n} and {n1} have common elements")
        
        self.nodegroups = nodegroups
        #create the partition matrix
        elementids.sort()
        pcols = 0 #maximum subcomponents phs dof dim
        rowindexs = dict()
        rowoffset = 0
        for i,el in enumerate(elementids):
            rowindexs[el] = rowoffset 
            rowoffset = rowindexs[el]+ phstructure[phstypes[i]]['rows']
            if phstructure[phstypes[i]]['cols'] > pcols:
                pcols = phstructure[phstypes[i]]['cols']
        #rowoffset = numrows of compositePHS            
        Pc = sympy.zeros(rowoffset,pcols*len(nodegroups))
        pcolix = 0
        
        for n,v in nodegroups.items():
            phs_ = phstructure[phstypes[v[0]]]
            phsel = sympy.eye(phs_['rows'])
            for vv in v:
                Pc[rowindexs[vv]:rowindexs[vv]+phs_['rows'],pcolix:pcolix+phs_['rows']] = phsel
            pcolix += pcols
        #Reduce the system
        '''
            ^J = PT J P is a skewsymmetric matrix.
            ^K = PT K P is skewsymmetric matrix.
            ^R = PT R P is a positive definite matrix.
            ^B = PT B P is a positive definite matrix.
            ^Bhat = PT B .
            ^Q= PT Q P is a positive definite matrix.        
            ^E= PT E P is a positive definite matrix.        
        '''
                    
        PT = Pc.T
        #K = self.compositePHS.B * self.compositePHS.C * (self.compositePHS.B.T)
        #Kr = PT*K*Pc
        Cr = PT*self.compositePHS.C*Pc
        Jr = PT*self.compositePHS.J*Pc
        Rr = PT*self.compositePHS.R*Pc
        Qr = PT*self.compositePHS.Q*Pc
        Er = PT*self.compositePHS.E*Pc
        Br = PT*self.compositePHS.B*Pc
        Bhatr = PT*self.compositePHS.Bhat
        statesr = PT*sympy.Matrix(self.compositePHS.states)
        ham = sympy.simplify(0.5*statesr.T*Qr*statesr)
        statesr = statesr[:]
        vars = self.compositePHS.variables
        parameters = self.compositePHS.parameters
        statevalues = self.compositePHS.parameters        
        
        
        phs =compositePHS = SymbolicPHS(
            Jr,
            Rr,
            Br,
            Bhatr,
            Qr,
            Er,
            Cr,
            ham,
            statesr,
            self.compositePHS.u,
            self.compositePHS.usplit,
            vars,
            parameters,
            statevalues,
        )
        
        sympy.pprint(ham)

import pickle        
if __name__ == '__main__':
    with open(r'D:\12Labours\GithubRepositories\FTUUtils\tests\data\Temp\compositephs.pkl','rb') as js:
        phsr = pickle.load(js)
        rphs = ReducePHS(phsr)
        ng = {'N1':[0,1],'N2':[2,3,4]}
        rphs.setClusters(ng)