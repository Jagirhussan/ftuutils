import numpy as np
import networkx as nx
import sympy, os
from copy import deepcopy
from collections import OrderedDict
import json

from ftuutils.compositionutils import SymbolicPHS,Composer
from ftuutils import codegenerationutils, compositionutils
'''
Logic to edit the topology of a FTU. 
'''
import time
class perf():
    
    def __init__(self) -> None:
        self.starttic = 0
        self.stoptic = 0
        self.prefix = ""
    
    def start(self,msg=""):
        self.starttic = time.time()
        self.prefix = msg
        
    def stop(self,msg=""):
        self.stoptic = time.time()
        print(f"Completed {self.prefix} in {(self.stoptic-self.starttic)} seconds {msg}")

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
            compositePHSjson = json.loads(compositePHSWithStructure)
            self.compositePHS = SymbolicPHS.loadPHSDefinition(compositePHSWithStructure['compositePHS'])
            self.phsStructure = compositePHSjson['compositePHSStructure']
            self.phsinstances =  compositePHSjson['phsinstances']
        elif isinstance(compositePHSWithStructure,Composer):
            self.composer = compositePHSWithStructure
            self.compositePHS = compositePHSWithStructure.compositePHS
            self.phsStructure = {'type':compositePHSWithStructure.phstypes,
                                 'rowidxs':compositePHSWithStructure.ridxs,
                                 'colidxs':compositePHSWithStructure.cidxs,
                                 'phsstructure':compositePHSWithStructure.phsclassstructure}
            self.phsinstances = compositePHSWithStructure.cellHamiltonians
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
        pcols = 0
        for n,v in nodegroups.items():
            nelem += len(v)
            pt = phstructure[phstypes[v[0]]]
            pcols += pt['cols']
            for vv in v[1:]:
                if pt['rows'] != phstructure[phstypes[vv]]['rows']:
                    raise Exception(f"cluster {n} has multiple PHS types {pt} and {phstypes[vv]} and dofs do not match")                    
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
        
        rowindexs = dict()
        rowoffset = 0
        for i,el in enumerate(elementids):
            rowindexs[el] = rowoffset 
            rowoffset = rowindexs[el]+ phstructure[phstypes[i]]['rows']
        
        #Update new hamiltonians
        cellHamiltonians = dict()
        for n,v in nodegroups.items():
            ham = 0
            uset = []
            for vv in v:
                #Node labels start at 1
                ham = ham + self.phsinstances[vv+1].hamiltonian
                uset.extend(self.phsinstances[vv+1].u[:])
            cellHamiltonians[n]= {'hamiltonian':ham,'u':set(uset)}
        self.cellHamiltonians = cellHamiltonians
        
        Pc = [sympy.zeros(rowoffset,pcols) for mts in range(9)] #Eight matrices of PHS
        ##Couling constant per phs instance dof based logic
        ccprefix = ['Cc','Jc','Rc','Qc','Ec','Bc','Bh','Sc']
        self.couplingconstants = []
        pcolix = 0        
        for n,v in nodegroups.items():
            phs_ = phstructure[phstypes[v[0]]]
            dofs = phs_['rows']            
            for vv in v:
                for mts in range(8):
                    consts = [sympy.Symbol(f"{ccprefix[mts]}_{di}_{vv}") for di in range(1,dofs+1)]
                    self.couplingconstants.extend(consts)
                    phsel = sympy.diag(*consts)
                    Pc[mts][rowindexs[vv]:rowindexs[vv]+phs_['rows'],pcolix:pcolix+phs_['rows']] = phsel               
                #For states to reduced states     
                Pc[-1][rowindexs[vv]:rowindexs[vv]+phs_['rows'],pcolix:pcolix+phs_['rows']] = sympy.eye(phs_['rows'])                    
            pcolix += phs_['cols']

        self.partitionMatrix = Pc    
        ##Per cluster weights logic
        # ccprefix = ['Cc','Jc','Rc','Qc','Ec','Bc','Bh','Sc']
        # pcolix = 0
        # self.couplingconstants = []
        # ccc = 0
        # for n,v in nodegroups.items():
        #     phs_ = phstructure[phstypes[v[0]]]
        #     for mts in range(8):
        #         phsel = sympy.eye(phs_['rows'])*sympy.Symbol(f"{ccprefix[mts]}_{ccc+1}")
        #         self.couplingconstants.append(sympy.Symbol(f"{ccprefix[mts]}_{ccc+1}"))
        #         for vv in v:
        #             Pc[mts][rowindexs[vv]:rowindexs[vv]+phs_['rows'],pcolix:pcolix+phs_['rows']] = phsel
        #     ccc +=1
        #     pcolix += phs_['cols']
            
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
                    
        #PT = Pc.T
        #K = self.compositePHS.B * self.compositePHS.C * (self.compositePHS.B.T)
        #Kr = PT*K*Pc
        Cr = Pc[0].T*self.compositePHS.C*Pc[0]
        Jr = Pc[1].T*self.compositePHS.J*Pc[1]
        Rr = Pc[2].T*self.compositePHS.R*Pc[2]
        Qr = Pc[3].T*self.compositePHS.Q*Pc[3]
        Er = Pc[4].T*self.compositePHS.E*Pc[4]
        Br = Pc[5].T*self.compositePHS.B*Pc[5]
        Bhatr = Pc[6].T*self.compositePHS.Bhat
        statesr = Pc[-1].T*sympy.Matrix(self.compositePHS.states)
        
        statesr = statesr[:]
        vars = self.compositePHS.variables
        parameters = self.compositePHS.parameters
        statevalues = self.compositePHS.statevalues        
        
        stateVec = sympy.zeros(len(statesr),1)
        reducedStateMap = dict()
        for i,s in enumerate(statesr):
            allstates = [x.name for x in s.free_symbols]
            #When there is common prefix use it, else the first states name
            cpre = os.path.commonprefix(allstates)
            if len(cpre)==0:
                cpre = allstates[0].split('_')[0]+'_'
            stateVec[i,0] = sympy.Symbol(f"r{cpre}{i}") #sympy.Symbol(f"{s}".replace("+","_P_").replace(" ",""))
            for s in allstates:
                reducedStateMap[sympy.Symbol(s)] = stateVec[i,0]
        self.stateVec = stateVec
        self.reducedStateMap = reducedStateMap
        ham = sympy.simplify(self.compositePHS.hamiltonian.subs(reducedStateMap))
                
        self.reducedPHS = SymbolicPHS(
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

    def exportForFitting(self):
        """
        Export the matrices as is and compose the rhs at runtime
        from the matrices. Reduces the code generation time
        """
        stateVec = sympy.Matrix(self.compositePHS.states)
       
        C = self.compositePHS.C
        J = deepcopy(self.compositePHS.J)
        R = deepcopy(self.compositePHS.R)
        Q = self.compositePHS.Q
        # Since E^-1 can be expensive, we will scale by the rate diagonal value of E for that component       
        E = deepcopy(self.compositePHS.E)
        for i in range(E.shape[0]):
            E[i, i] = 1 / E[i, i]
        
        B = self.compositePHS.B
        Bhat = self.compositePHS.Bhat

        #Create variables for each nonlinear variable
        nonlinearrhsterms = dict()
        constants = dict()
        for c, t in self.compositePHS.parameters.items():
            fs = t["value"].free_symbols
            if len(fs) > 0: #This is a nonlinearterm
                nonlinearrhsterms[c] = t['value']
            else:
                constants[c] = t['value']
                
        #Handle operators in J and R. Op*state = Op(state)
        for r in range(J.shape[0]):
            for c in range(J.shape[1]):
                jfs = J[r,c].free_symbols
                if len(jfs)>0:
                    fs = set()
                    for x in jfs:
                        fs = fs.union(x.xreplace(nonlinearrhsterms).free_symbols)

                    #print(J[r,c],end=" -> ")
                    if stateVec[c] in fs:
                        if isinstance(J[r,c],sympy.Add):
                            term = 0
                            for elem in J[r,c].args:
                                eits = set()
                                for x in elem.free_symbols:
                                    eits = eits.union(x.xreplace(nonlinearrhsterms).free_symbols)
                                if stateVec[c] in eits:
                                    term = term + elem/stateVec[c]
                                else:
                                    term = term + elem
                            J[r,c] = sympy.simplify(term)
                        else:
                            J[r,c] = sympy.simplify(J[r,c]/stateVec[c])
                    #print(J[r,c])
        for r in range(R.shape[0]):
            for c in range(R.shape[1]):
                rfs = R[r,c].free_symbols
                if len(rfs)>0:
                    fs = set()
                    for x in rfs:
                        fs = fs.union(x.xreplace(nonlinearrhsterms).free_symbols)
                    #print(R[r,c],end=" -> ")
                    if stateVec[c] in fs:
                        if isinstance(R[r,c],sympy.Add):
                            term = 0
                            for elem in R[r,c].args:
                                eits = set()
                                for x in elem.free_symbols:
                                    eits = eits.union(x.xreplace(nonlinearrhsterms).free_symbols)
                                if stateVec[c] in eits:
                                    term = term + elem/stateVec[c]
                                else:
                                    term = term + elem
                            R[r,c] = sympy.simplify(term)
                        else:
                            R[r,c] = sympy.simplify(R[r,c]/stateVec[c])
                    #print(R[r,c])                
                
                
        arraymap = dict()
        for i in range(stateVec.shape[0]):
            arraymap[stateVec[i]] = sympy.Symbol(f"states[{i}]")
        statevalues = []
        for i in range(stateVec.shape[0]):
            statevalues.append(self.compositePHS.statevalues[stateVec[i]]['value'])
        
        #Collect arrayname and constant substituted and simplified nonlinear terms
        nic = 0 #First variables are inputs
        nonlineararraymap = dict()
        inputVec = dict()
        for v in self.compositePHS.u:
            nonlineararraymap[v] = sympy.Symbol(f"variables[{nic}]")
            inputVec[v] = nonlineararraymap[v]
            nic += 1                
        ivend = nic    
        solvedvars = dict()
        for k,v in nonlinearrhsterms.items():
            val = sympy.simplify(v.xreplace(constants).xreplace(arraymap))
            nonlineararraymap[k] = sympy.Symbol(f"variables[{nic}]")
            solvedvars[nonlineararraymap[k]] = val
            nic += 1
        Rarr = R.xreplace(nonlineararraymap).xreplace(arraymap)
        Jarr = J.xreplace(nonlineararraymap).xreplace(arraymap)
        Qarr = Q.xreplace(nonlineararraymap).xreplace(arraymap)
        Earr = E.xreplace(nonlineararraymap).xreplace(arraymap)
        Barr = B.xreplace(nonlineararraymap).xreplace(arraymap)
        Bhatarr = Bhat.xreplace(nonlineararraymap).xreplace(arraymap)
        Carr = C.xreplace(nonlineararraymap).xreplace(arraymap)

        pcm = self.partitionMatrix[-1]
        paramsetcode = ''
        #matnames = ['Jr','Rr','Qr','Er','Cr','Br','Bhatr','Ict']
        matnames = ['Jr','Rr','Qr','Er','Cr','Br','Bhatr']
        pix = 0
        for m in matnames:
            for i in range(pcm.shape[0]):
                for j in range(pcm.shape[1]):
                    if pcm[i,j] != 0:
                        paramsetcode += f"    {m}[{i},{j}] = parameters[{pix}]\n"
                        pix+=1            
                
        pycode = f'''
import numpy as np
from numpy import exp
        
STATE_COUNT = {stateVec.shape[0]}
VARIABLE_COUNT = {nic}
PARAMETER_COUNT = {pix}

# PHS matrices
J = np.zeros(({Jarr.shape[0]},{Jarr.shape[1]}))
R = np.zeros(({Rarr.shape[0]},{Rarr.shape[1]}))
Q = np.zeros(({Qarr.shape[0]},{Qarr.shape[1]}))
Einv = np.zeros(({Earr.shape[0]},{Earr.shape[1]}))
B = np.zeros(({Barr.shape[0]},{Barr.shape[1]}))
Bhat = np.zeros(({Bhatarr.shape[0]},{Bhatarr.shape[1]}))
C = np.zeros(({Carr.shape[0]},{Carr.shape[1]}))

#Partition matrices
Jr = np.zeros(({pcm.shape[0]},{pcm.shape[1]}))
Rr = np.zeros(({pcm.shape[0]},{pcm.shape[1]}))
Qr = np.zeros(({pcm.shape[0]},{pcm.shape[1]}))
Er = np.zeros(({pcm.shape[0]},{pcm.shape[1]}))
Br = np.zeros(({pcm.shape[0]},{pcm.shape[1]}))
Cr = np.zeros(({pcm.shape[0]},{pcm.shape[1]}))
Bhatr = np.zeros(({pcm.shape[0]},{pcm.shape[1]}))
Ict = np.zeros(({pcm.shape[0]},{pcm.shape[1]})) #For initial condition
        
def Heaviside(x):
    if x > 0:
        return 1.0
    return 0.0
    
def Abs(x):
    return np.fabs(x)    
    
def create_states_array():
    return np.zeros(STATE_COUNT)

def create_variables_array():
    return np.zeros(VARIABLE_COUNT)

def setReductionParameters(parameters):
{paramsetcode}
        
def initialise_variables(states, variables):\n  
'''
        pycode +="    #State values\n"
        for i in range(stateVec.shape[0]):
            if statevalues[i] != 0.0:
                pycode +=f"    states[{i}] = {statevalues[i]}\n"
            else:
                pycode +=f"    states[{i}] = 1e-12\n"
        for k,v in solvedvars.items():            
            pycode +=f"    {k} = {v}\n"
        #previous state values
        # pycode +="    #State values at t - 1\n"
        # for i in range(stateVec.shape[0]):
        #     pycode +=f"\tvariables[{ivend+nic+i}] = {statevalues[i]}\n"
        pycode +="\n    #Ict matrix entries\n"
        for i in range(pcm.shape[0]):
            for j in range(pcm.shape[1]):
                if pcm[i,j]!=0.0:
                    pycode +=f"    Ict[{i},{j}] = 1\n"            
                
        pycode +="\n    #PHS matrix - constant entries\n"
        #Matrix constants
        def getConstantMatrixEntries(mat,matname):
            code = ""
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):        
                    if mat[i,j]!=0.0:
                        mexp = f"{mat[i,j]}"
                        if 'variables' not in mexp and 'states' not in mexp:
                            code +=f"    {matname}[{i},{j}]={mat[i,j]}\n"
            return code        
        pycode += getConstantMatrixEntries(Jarr,"J")
        pycode += getConstantMatrixEntries(Rarr,"R")
        pycode += getConstantMatrixEntries(Qarr,"Q")
        pycode += getConstantMatrixEntries(Earr,"Einv")
        pycode += getConstantMatrixEntries(Barr,"B")
        pycode += getConstantMatrixEntries(Bhatarr,"Bhat")
        pycode += getConstantMatrixEntries(Carr,"C")        
        
        pycode +="\n#Setup rhs\ndef setup_rhs(voi,states,variables):"
        pycode +="\n    process_timesensitive_inputs(voi,states,variables)"
        pycode +="\n    #Computing nonlinear variables\n"
        #First compute nonlinear variables
        for k,v in solvedvars.items():            
            pycode +=f"    {k} = {v}\n"
        pycode +="\n    #Variable PHS matrix entries\n"
        #Now set the matrices
        def getVariableMatrixEntries(mat,matname):
            code = ""
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):        
                    if mat[i,j]!=0.0:
                        mexp = f"{mat[i,j]}"
                        if 'variables' in mexp or 'states' in mexp:
                            code +=f"    {matname}[{i},{j}]={mat[i,j]}\n"
            return code
        pycode += getVariableMatrixEntries(Jarr,"J")
        pycode += getVariableMatrixEntries(Rarr,"R")
        pycode += getVariableMatrixEntries(Qarr,"Q")
        pycode += getVariableMatrixEntries(Earr,"Einv")
        pycode += getVariableMatrixEntries(Barr,"B")
        pycode += getVariableMatrixEntries(Bhatarr,"Bhat")
        pycode += getVariableMatrixEntries(Carr,"C")
        #pycode +=f"\n    #variables[{nic+ivend}:-1] are state values at time t-1, updated after each successful step"
        #pycode +=f"\n\treturn Einv@((J-R)@Q@states - B@C@(B.T)@variables[{nic+ivend}:] + Bhat@variables[:{ivend}])"
        pycode +="\n\n#Compute rhs\ndef compute_rhs(voi,states,variables):"
        pycode +=f"\n    setup_rhs(voi,states,variables)"
        pycode +=f"\n    return Einv@((J-R)@Q@states - B@C@(B.T)@states + Bhat@variables[:{ivend}])"
        pycode +="\n#solve a step rhs\ndef process_timesensitive_inputs(voi,states,variables):"
        if ivend>0:
            pycode +="\n    #Input variables"                
            for k,v in inputVec.items():
                pycode +=f"\n    {v} = 0.0 #{k}"
        else:
            pycode +="\n    pass"
        pycode +="\n"
        pycode +='''
    if voi > 100.0 and voi < 120.0:
        variables[0] = 0.5

def compute_reduced_rhs(voi,states,variables):
    #Map reduced states to full states
    full_states = Ict@states
    setup_rhs(voi,full_states,variables)
    #reduced matrices
    Erd = (Er.T)@Einv@Er
    Qrd = (Qr.T)@Q@Qr
    Jrd = (Jr.T)@J@Jr
    Rrd = (Rr.T)@R@Rr
    Crd = (Cr.T)@C@Cr
    Brd = (Br.T)@B@Br
    Bhrd = (Bhatr.T)@Bhat
    return Erd@((Jrd-Rrd)@Qrd@states - Brd@Crd@(Brd.T)@states + Bhrd@variables[:1])


def solve_model(starttime=0,stoptime=300,steps=300):
    """Solve model with ODE solver"""
    from scipy.integrate import ode
    import numpy as np
    # Initialise constants and state variables
    states = create_states_array()
    variables = create_variables_array()
    initialise_variables(states,variables)
    # Set timespan to solve over
    voi = np.linspace(starttime, stoptime, steps)

    # Construct ODE object to solve
    r = ode(compute_rhs)
    r.set_integrator('vode', method='bdf', atol=1e-06, rtol=1e-06, max_step=1)
    r.set_initial_value(states, voi[0])
    r.set_f_params(variables)

    # Solve model
    result = np.zeros((STATE_COUNT,steps))
    result[:,0] = states    
    for (i,t) in enumerate(voi[1:]):
        if r.successful():
            r.integrate(t)
            result[:,i+1] = r.y
            states = r.y
        else:
            break

    return voi, result

def reduced_solve_model(starttime=0,stoptime=300,steps=300):
    """Solve model with ODE solver"""
    from scipy.integrate import ode
    import numpy as np
    # Initialise constants and state variables
    full_states = create_states_array()
    variables = create_variables_array()
    initialise_variables(full_states,variables)
    states = (Ict.T)@full_states
    # Set timespan to solve over
    voi = np.linspace(starttime, stoptime, steps)

    # Construct ODE object to solve
    r = ode(compute_reduced_rhs)
    r.set_integrator('vode', method='bdf', atol=1e-06, rtol=1e-06, max_step=1)
    r.set_initial_value(states, voi[0])
    r.set_f_params(variables)

    # Solve model
    result = np.zeros((states.shape[0],steps))
    result[:,0] = states    
    for (i,t) in enumerate(voi[1:]):
        if r.successful():
            r.integrate(t)
            result[:,i+1] = r.y
            states = r.y
        else:
            break

    return voi, result


import matplotlib.pyplot as plt
def full_model():
    t,r = solve_model()
    U = r.reshape((25,4,-1))[:,2,:].squeeze()
    grid = plt.GridSpec(5, 5, wspace=0.2, hspace=0.5)

    ix = 0
    for i in range(5):
        for j in range(5):
            ax = plt.subplot(grid[i, j])
            ax.plot(U[ix,:])
            ax.title.set_text(f'{ix+1}')
            ix += 1
            if ix+1 > U.shape[0]:
                break
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.3)
    plt.show()      
    
def reduced_model():
    #Initialize reduction parameters
    np.random.seed(0)
    params = np.random.rand(PARAMETER_COUNT)
    setReductionParameters(params)
    t,r = reduced_solve_model()
    U = r.reshape((5,4,-1))[:,2,:].squeeze()
    grid = plt.GridSpec(5, 1, wspace=0.2, hspace=0.5)

    ix = 0
    for i in range(5):
        ax = plt.subplot(grid[i, 0])
        ax.plot(U[ix,:])
        ax.title.set_text(f'{ix+1}')
        ix += 1
        if ix+1 > U.shape[0]:
            break
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.3)
    plt.show()         
    
if __name__ == '__main__':
    reduced_model()
    
        '''                    
        print(pycode)
        return pycode
                   
    def generatePythonIntermediates(self,substituteParameters=False):
        r"""Generate numerically solvable PHS
            d/dx (Ex) = (J-R) Q x - \hat{B}\hat{C}\hat{B}^T \hat{u} + \bar{B} \bar{u}
            \hat{y} = \hat{B}^T Q x
            \bar{y} = \bar{B}^T Q x
        """

        
        reducedStateMap = self.reducedStateMap
        stateVec = self.stateVec
        ucapVec = sympy.Matrix([f"u_{s}" for s in stateVec])
        self.ucapVec = ucapVec
        Ccap = self.reducedPHS.C.xreplace(reducedStateMap)
        Delx = self.reducedPHS.Q.xreplace(reducedStateMap) * stateVec  # Potential
        # Since E^-1 can be expensive, we will scale by the rate diagonal value of E for that component
        Einv = sympy.eye(self.reducedPHS.E.shape[0])
        for i in range(self.reducedPHS.E.shape[0]):
            Einv[i, i] = 1 / self.reducedPHS.E[i, i]
        JRQx = (self.reducedPHS.J - self.reducedPHS.R) * Delx
        interioru = self.reducedPHS.B * Ccap * (self.reducedPHS.B.T) * ucapVec
        exterioru = self.reducedPHS.Bhat * self.reducedPHS.u.T
        rhs = sympy.SparseMatrix(Einv * (JRQx - interioru + exterioru)).xreplace(reducedStateMap)
        inputs = sympy.SparseMatrix(Einv * (-interioru + exterioru)).xreplace(reducedStateMap)
        rhsfreesymbols = rhs.free_symbols
        rhsnumericconstants = []
        self.raw_rhs = deepcopy(rhs)
        for elem in rhs:
            nc = compositionutils.getNumerics(elem)
            rhsnumericconstants.extend(
                        [
                            term
                            for term in nc
                            if term not in rhsfreesymbols
                            and isinstance(term, sympy.Number)
                            and np.fabs(float(term)) != 1.0
                        ]
            )


        # Constants in uCapVec
        for i, s in enumerate(stateVec):
            elem = interioru[i]
            nc = compositionutils.getNumerics(elem)
            rhsnumericconstants.extend(
                        [
                            term
                            for term in nc
                            if term not in rhsfreesymbols
                            and isinstance(term, sympy.Number)
                            and np.fabs(float(term)) != 1.0
                        ]
            )
            
        nc = compositionutils.getNumerics(self.reducedPHS.hamiltonian)
        rhsnumericconstants.extend(
                    [
                        term
                        for term in nc
                        if term not in rhsfreesymbols
                        and isinstance(term, sympy.Number)
                        and np.fabs(float(term)) != 1.0
                    ]
        )            
        #Find constants in composite parameters and get the list of nonlinear terms as well
        nonlinearrhsterms = dict()
        for c, t in self.reducedPHS.parameters.items():
            fs = t["value"].free_symbols
            cvdict = dict()
            if len(fs) > 0: #This is a nonlinearterm
                nc = compositionutils.getNumerics(t["value"])
                rhsnumericconstants.extend(
                        [
                            term
                            for term in nc
                            if term not in rhsfreesymbols
                            and isinstance(term, sympy.Number)
                            and np.fabs(float(term)) != 1.0
                        ])                
                nonlinearrhsterms[c] = sympy.simplify(t["value"].xreplace(reducedStateMap))
            else:
                rhsnumericconstants.append(np.fabs(float(t["value"])))

        constantsubs = dict()
        constCtr = 1
        for c in set(rhsnumericconstants):
            constantsubs[np.abs(c)] = f"c_{constCtr}"
            constCtr += 1

        #Convert to sympy Symbols for substitution
        for k in constantsubs:
            constantsubs[k] = sympy.Symbol(constantsubs[k])

        for c in nonlinearrhsterms:
            #Sympy handle Heaviside wierdly - so rename here and 
            vs = f"{nonlinearrhsterms[c]}"
            if "Heaviside" in vs:
                vn = sympy.sympify(vs.replace("Heaviside(","heaviside(")).xreplace(constantsubs)
                vk = f"{vn}"
                nonlinearrhsterms[c] = sympy.sympify(vk)
            else: 
                v = nonlinearrhsterms[c].xreplace(constantsubs)
                nonlinearrhsterms[c] = v
            
        # Xreplace is faster than subs - no deep mathematical reasoning, ok for constant replacement
        cleanrhs = rhs.xreplace(constantsubs)
        cleaninputs = inputs.xreplace(constantsubs)

        # Find all symbolic constants in nonlinearrhsterms
        for k, v in nonlinearrhsterms.items():
            fs = v.free_symbols
            cvdict = dict()
            for f in fs:
                if f in self.reducedPHS.parameters:
                    if self.reducedPHS.parameters[f]["value"] in constantsubs:
                        cvdict[f] = constantsubs[self.reducedPHS.parameters[f]["value"]]
                    elif np.fabs(float(self.reducedPHS.parameters[f]["value"])) != 1.0:
                        constantsubs[self.reducedPHS.parameters[f]["value"]] = f
                        constCtr += 1
            if len(cvdict) > 0:
                v = v.xreplace(cvdict)
        # Remove constant entries that are same to given precision
        constCtr = 1
        constantstoprecision = dict()
        newkeys = dict()
        skippedkeys = dict()
        phsconstants = dict()
        #Get all phs constants from composite parameters (all with numeric values)
        #if not self.substituteParameters:
        for k,v in self.reducedPHS.parameters.items():
            if len(v['value'].free_symbols)==0:
                phsconstants[k] = v #float(v['value'])
        
        for k, v in constantsubs.items():
            pk = f"{float(k):6f}"
            if pk not in constantstoprecision:
                if v.name.startswith('c_'):
                    constantstoprecision[pk] = sympy.Symbol(f"c_{constCtr}")
                else:
                    constantstoprecision[pk] = v
                constCtr += 1
            #Only for constant defined by the code and not phs constants
            if v != constantstoprecision[pk] and not v in phsconstants: #v.name.startswith('c_'):
                skippedkeys[v] = constantstoprecision[pk]
            newkeys[k] = constantstoprecision[pk]

        for k, v in newkeys.items():
            constantsubs[k] = v

        # Handle elements that are functions, the multiplication operation with a state should be composition
        cleanedrhs = []

        for relem in cleanrhs:
            expandedelem = sympy.expand(relem)
            #Look into each product term of a sum or product
            # Sum need to be summed, product need to be multiplied
            if isinstance(expandedelem,sympy.Add):
                reducedelem = 0
                for elem in expandedelem.args:
                    #Each element is a product or free
                    estates = []
                    enterms = []
                    for f in elem.free_symbols:
                        if f in stateVec.free_symbols:
                            estates.append(f)
                        if f in nonlinearrhsterms:
                            enterms.append(f)
                    if len(estates)>0 and len(enterms)>0:
                        denom = 1
                        for nt in enterms:
                            entf = nonlinearrhsterms[nt].free_symbols
                            for s in estates:
                                if s in entf:
                                    denom *= s
                        reducedelem += sympy.simplify(elem/denom)
                    else:
                        reducedelem += elem
                cleanedrhs.append(reducedelem)
            elif isinstance(expandedelem,sympy.Mul): # if its a product
                reducedelem = 1
                estates = []
                enterms = []
                for f in expandedelem.free_symbols:
                    if f in stateVec.free_symbols:
                        estates.append(f)
                    if f in nonlinearrhsterms:
                        enterms.append(f)
                if len(estates)>0 and len(enterms)>0:
                    denom = 1
                    for nt in enterms:
                        entf = nonlinearrhsterms[nt].free_symbols
                        for s in estates:
                            if s in entf:
                                denom *= s
                    reducedelem *= sympy.simplify(expandedelem/denom)
                else:
                    reducedelem *= expandedelem
                cleanedrhs.append(reducedelem)
            else:
                cleanedrhs.append(relem)

        # Constants also contain u vector, however they are updated after each step
        # Do the update in compute_variables method, and initialise them in initialise_variables method
        # Compute rhs contains, nonlinearrhsterms and cleanedrhs
        # All parameters are in compositeparameters
        # Translate to use constants and states arrays
        arraymapping = OrderedDict()
        invarraymapping = OrderedDict()
        arraysubs = dict()
        for i, s in enumerate(stateVec):
            arraysubs[s] = sympy.Symbol(f"states[{i}]")
            arraymapping[s] = f"states[{i}]"
            invarraymapping[f"states[{i}]"] = s.name


        numconstants = 0
        # Do ubar first as numconstants change due to precision selection
        # ubar entries
        ubaridxmap = dict()
        for s in self.reducedPHS.u.free_symbols:
            arraysubs[s] = sympy.Symbol(f"variables[{numconstants}]")
            arraymapping[s.name] = f"variables[{numconstants}]"
            invarraymapping[f"variables[{numconstants}]"] = s.name
            ubaridxmap[s.name] = f"variables[{numconstants}]"
            numconstants += 1
        #Find any connectivity related symbols
        ftuidmap = dict()
        for s in interioru.free_symbols:
            if not s.name.startswith("u_"):
                arraysubs[s] = sympy.Symbol(f"variables[{numconstants}]")
                arraymapping[s.name] = f"variables[{numconstants}]"
                invarraymapping[f"variables[{numconstants}]"] = s.name
                ftuidmap[s.name] = f"variables[{numconstants}]"
                numconstants += 1
        self.couplingconstantsmap = {}
        if len(self.couplingconstants)>0:
            for s in self.couplingconstants:
                arraysubs[s] = sympy.Symbol(f"variables[{numconstants}]")
                self.couplingconstantsmap[s] = arraysubs[s]
                arraymapping[s.name] = f"variables[{numconstants}]"
                invarraymapping[f"variables[{numconstants}]"] = s.name
                ftuidmap[s.name] = f"variables[{numconstants}]"
                numconstants += 1
                

        # Multiple k's will have same v due to defined precision
        definedConstants = []
        for k, v in constantsubs.items():
            if v not in definedConstants:
                arraysubs[v] = sympy.Symbol(f"variables[{numconstants}]")
                arraymapping[str(v)] = f"variables[{numconstants}]"
                invarraymapping[f"variables[{numconstants}]"] = str(v)
                numconstants += 1
                definedConstants.append(v)
        if not substituteParameters:
            #Insert phs constants
            for v,k in phsconstants.items():
                arraysubs[v] = sympy.Symbol(f"variables[{numconstants}]")
                arraymapping[str(v)] = f"variables[{numconstants}]"
                invarraymapping[f"variables[{numconstants}]"] = str(v)
                numconstants += 1
        else:
            #Reduce repeats
            existingvalues = {}
            newphsconstants = {}
            for k,vdict in phsconstants.items():
                v = vdict['value']
                if float(v) in existingvalues:
                    arraysubs[k] = existingvalues[float(v)]
                    arraymapping[str(k)] = str(arraysubs[k])
                    invarraymapping[arraymapping[str(k)] ] = str(k)
                else:
                    arraysubs[k] = sympy.Symbol(f"variables[{numconstants}]")
                    arraymapping[str(k)] = f"variables[{numconstants}]"
                    invarraymapping[f"variables[{numconstants}]"] = str(k)
                    numconstants += 1    
                    existingvalues[v] = arraysubs[k]  
                    newphsconstants[k] = vdict            
            phsconstants = newphsconstants
            
        # uCap entries
        for s in stateVec:
            arraysubs[sympy.Symbol(f"u_{s}")] = sympy.Symbol(
                f"variables[{numconstants}]"
            )
            arraymapping[f"u_{s}"] = f"variables[{numconstants}]"
            invarraymapping[f"variables[{numconstants}]"] = fr"\hat{{u}}_{s}"
            numconstants += 1

        # Non linear rhs terms
        for s, v in nonlinearrhsterms.items():
            arraysubs[s] = sympy.Symbol(f"variables[{numconstants}]")
            arraymapping[s.name] = f"variables[{numconstants}]"
            invarraymapping[f"variables[{numconstants}]"] = s.name
            numconstants += 1

        uCapterms = dict()
        ucapdescriptive = dict()
        for i, s in enumerate(stateVec):
            consu = Delx[i].xreplace(skippedkeys).xreplace(constantsubs)
            res = consu.xreplace(arraysubs)
            uCapterms[arraymapping[f"u_{s}"]] = res
            # Done this was as sympy printing changes the order of printed expr
            ucapdescriptive[
                arraymapping[f"u_{s}"]
            ] = fr"\hat{{u}}_{s} = {codegenerationutils._stringsubs(res.__str__(),invarraymapping)}"

        nonlineararrayedrhsterms = dict()
        nonlinearrhstermsdescriptive = dict()

        for s, v in nonlinearrhsterms.items():
            consv = v.xreplace(skippedkeys).xreplace(constantsubs)
            res = consv.xreplace(arraysubs)
            nonlineararrayedrhsterms[arraymapping[s.name]] = res
            # Done this was as sympy printing changes the order of printed expr
            nonlinearrhstermsdescriptive[
                arraymapping[s.name]
            ] = f"{s} = {codegenerationutils._stringsubs(res.__str__(),invarraymapping)}"

        arrayedrhs = []
        arrayedinputs = []
        # Use cleanedrhs and not cleanrhs - nonlinear terms with functions are transformed to compositions and not multiplication
        for elem in cleanedrhs:
            arrayedrhs.append(elem.xreplace(skippedkeys).xreplace(arraysubs))
        for elem in cleaninputs:
            arrayedinputs.append(elem.xreplace(skippedkeys).xreplace(arraysubs))
        
        return numconstants,phsconstants,constantsubs,nonlinearrhsterms,inputs,arrayedinputs,arraymapping,uCapterms,ucapdescriptive,nonlineararrayedrhsterms,nonlinearrhstermsdescriptive,arrayedrhs,invarraymapping,rhs,ubaridxmap,ftuidmap,cleaninputs
            
    def exportAsPython(self):
        """Export the FTU description in the composer as python code

        Args:
            composer (compositionutils.Composer): Composer instance that has the resolved FTU 
        """
        numconstants,phsconstants,constantsubs,nonlinearrhsterms,inputs,arrayedinputs,arraymapping,uCapterms,ucapdescriptive,nonlineararrayedrhsterms,nonlinearrhstermsdescriptive,arrayedrhs,invarraymapping,rhs,ubaridxmap,ftuidmap,cleaninputs = self.generatePythonIntermediates()
        stateVec = sympy.Matrix(self.stateVec)
        perfctr = perf()
        # Generate metedata
        variabledescription = 'VOI_INFO = {"name": "t", "units": "second", "component": "main", "type": VariableType.VARIABLE_OF_INTEGRATION}\n'
        variabledescription += "STATE_INFO = [\n"
        for k, v in self.reducedPHS.statevalues.items():
            variabledescription += f'    {{"name": "{k}", "units": "{v["units"]}", "component": "main", "type": VariableType.STATE}},\n'
        variabledescription += "]\n\nVARIABLE_INFO = [\n"
        for k, v in ubaridxmap.items():
            #TODO get the dimension
            variabledescription += f'    {{"name": "{k}", "units": "dimensionless", "component": "main", "type": VariableType.EXTERNAL_INPUT}},\n'        
        if len(ftuidmap)>0:
            for k,v in ftuidmap.items():
                variabledescription += f'    {{"name": "{k}", "units": "dimensionless", "component": "main", "type": VariableType.CONSTANT}},\n'                    
        
        # Maintain this order when creating variables
        # Do constant subs, constant subs will have multiple values for the same constant due to precision
        definedNames = []
        for k, v in constantsubs.items():
            if v not in definedNames:
                if not v.name.startswith("-"):
                    variabledescription += f'    {{"name": "{v}", "units": "dimensionless", "component": "main", "type": VariableType.CONSTANT}},\n'
                    definedNames.append(v)
        for k,v in phsconstants.items():
            if k.name in arraymapping:
                vunit = v['units']
                variabledescription += f'    {{"name": "{k}", "units": "{vunit}", "component": "main", "type": VariableType.CONSTANT}},\n'                    


        # TODO compute the units of calculated terms
        # Do uCap terms
        for v in stateVec:
            variabledescription += f'    {{"name": "u_{v}", "units": "dimensionless", "component": "main", "type": VariableType.INTERNAL_INPUT}},\n'
        for s in self.reducedPHS.u.free_symbols:
            variabledescription += f'    {{"name": "{s}", "units": "dimensionless", "component": "main", "type": VariableType.EXTERNAL_INPUT}},\n'

        # Do nonlinear terms
        for k, v in nonlinearrhsterms.items():
            variabledescription += f'    {{"name": "{k}", "units": "dimensionless", "component": "main", "type": VariableType.ALGEBRAIC}},\n'
        variabledescription += "]\n"

        # ucap and nonlinearrhs terms go into def compute_variables(voi, states, rates, variables)
        # nonlineararrayedrhsterms go into compute_rates(voi, states, rates, variables)
        # def compute_computed_constants(variables) is ucap and nonlinearrhs
        # def initialise_variables(states, variables) contains initialisations

        pycode = f"""
# The content of this file was generated using the FTUWeaver

from enum import Enum
import numpy as np


__version__ = "0.0.1"

STATE_COUNT = {len(self.stateVec)}
VARIABLE_COUNT = {numconstants}
COUPLING_PARAMETERS = {len(self.couplingconstants)}

def heaviside(x):
    if x > 0:
        return 1.0
    return 0.0
    
def Abs(x):
    return np.fabs(x)

class VariableType(Enum):
    VARIABLE_OF_INTEGRATION = 0
    STATE = 1
    CONSTANT = 2
    COMPUTED_CONSTANT = 3
    ALGEBRAIC = 4
    INTERNAL_INPUT = 5
    EXTERNAL_INPUT = 6

{variabledescription}

def create_states_array():
    return np.zeros(STATE_COUNT)

def create_variables_array():
    return np.zeros(VARIABLE_COUNT)
        
def initialise_variables(states, variables):\n"""
        # Do states first
        #As states are reduced find the combined values
        stateinitialvalues = dict()
        for k, v in self.reducedPHS.statevalues.items():
            stateinitialvalues[k] = v['value']
                    
        # for k, v in self.reducedPHS.statevalues.items():
        #     try:
        #         stmt = f"    {arraymapping[k.name]} = {float(v['value']):6f}  #{k}\n"
        #     except:
        #         stmt = f"    {arraymapping[k.name]} = {v['value']}  #{k}\n"
        #     pycode += stmt
        for k, v in ubaridxmap.items():
            pycode += f"    {v} = 0.0 #{k} External input\n"
                    
        if len(ftuidmap)>0:
            for k,v in ftuidmap.items():
                pycode += f"    {v} = 1.0  #{k} This needs to be set for accurate simulation\n"

        # Do constant subs
        definedVariables = []
        for k, v in constantsubs.items():
            if v not in definedVariables:
                if v.name in arraymapping:
                    try:
                        stmt = f"    {arraymapping[v.name]} = {float(k):6f}  #{v}\n"
                    except:
                        stmt = f"    {arraymapping[v.name]} = {k}  #{v}\n"
                    pycode += stmt
                    definedVariables.append(v)
        for v, k in phsconstants.items():
            if v not in definedVariables:
                if v.name in arraymapping:
                    try:
                        stmt = f"    {arraymapping[v.name]} = {float(k['value']):6f}  #{v}\n"
                    except:
                        stmt = f"    {arraymapping[v.name]} = {k['value']}  #{v}\n"
                    pycode += stmt
                    definedVariables.append(v)                
        #Do states here as they may use use variables
        for i,s in enumerate(self.reducedPHS.states):
            rv = sympy.simplify(s.subs(stateinitialvalues)).xreplace(self.couplingconstantsmap)
            try:
                stmt = f"    {arraymapping[stateVec[i,0]]} = {float(rv.name):6f}  #{s.name}\n"
            except:
                stmt = f"    {arraymapping[stateVec[i,0]]} = {rv}  #{s}\n"
            pycode += stmt
                    
        pycode += "\ndef compute_computed_constants(variables):\n\tpass\n\n"
        pycode += "def compute_variables(voi, states, rates, variables):\n\tt=voi #mapping to t\n"
        # Do uCap terms
        for k, v in uCapterms.items():
            pycode += f"    #{ucapdescriptive[k]}\n"
            pycode += f"    {k} = {v}\n"

        # Do rhs
        pycode += "\ndef compute_rates(voi, states, rates, variables):\n\tt=voi #mapping to t\n"
        # Do nonlinear terms - these depend on state values and therefore step size, os here instead of compute variables
        for k, v in nonlineararrayedrhsterms.items():
            pycode += f"    #{nonlinearrhstermsdescriptive[k]}\n"
            pycode += f"    {k} = {v}\n"
        for i, v in enumerate(arrayedrhs):
            pycode += f"    #\dot{{{self.stateVec[i]}}} = {codegenerationutils._stringsubs(str(v),invarraymapping)} # {sympy.simplify(rhs[i,0])}\n"
            pycode += f"\trates[{i}] = {v}\n"

        # Do inputs
        pycode += "\ndef compute_inputs(voi,inputs,states,variables):\n"
        for i, v in enumerate(arrayedinputs):
            pycode += f"    # cell[{i}] = {cleaninputs[i]}\n"
            pycode += f"\tinputs[{i}] = {v}\n"
        
        #Provide input hook for coupling parameters optimisation
        pycode += "\ndef set_coupling_parameters(variables,parameters):\n"
        for i, (k,v) in enumerate(self.couplingconstantsmap.items()):
            pycode += f"    {v} = parameters[{i}] # {k}\n"
        pycode += f"\treturn variables\n\n"
        # Provide external input variable names in comment to help support
        ubarcomment = ""
        for k, v in ubaridxmap.items():
            ubarcomment += f"    #    {k} -> {v}\n"
        pycode += f'''
from math import exp

def process_time_sensitive_events(voi, states, rates, variables):
    """Method to process events such as (re)setting inputs, updating switches etc
        Unline process_events, this method is called in rhs calculation
        Useful to ensure that time sensitive inputs are set espcially if ode integrator timestep spans over the 
        input time. Note that this should be re-entrant i.e. not modify states, else this will
        lead to solver dependent behaviour, esp. solvers that use multiple steps
        The method is called before each rhs evaluation
    Args:
        voi (int) : Current value of the variable of integration (time)
        states (np.array): A vectors of model states
        variables (_type_): A vector of model variables
    """
    #External input variables - listed to help code event processing logic
{ubarcomment}
    #Comment the line below (and uncomment the line after) to solve the model without event processing!    
    #raise("Process time sensitive events not implemented")
    #pass
    variables[0] = 0.0
    if voi > 100 and voi < 110:
        variables[0] = 0.5
    #Following needs to be performed to set internal inputs from current state values
    compute_variables(voi,states,rates,variables)    
    
def process_events(voi, states,variables):
    """Method to process events such as (re)setting inputs, updating switches etc
        The method is called after each successful ode step
    Args:
        voi (int) : Current value of the variable of integration (time)
        states (np.array): A vectors of model states
        variables (_type_): A vector of model variables
    """
    #External input variables - listed to help code event processing logic
{ubarcomment}

    #Comment the line below (and uncomment the line after) to solve the model without event processing!    
    #raise("Process events not implemented")

    pass

rates = np.zeros(STATE_COUNT)

def rhs(voi, states, variables):
    #Perform (re)setting of inputs, time sensitive event processing etc
    process_time_sensitive_events(voi,states,rates,variables)    
    #Compute rates
    compute_rates(voi,states,rates,variables)
    return rates

def solve_model(starttime=0,stoptime=300,steps=300):
    """Solve model with ODE solver"""
    from scipy.integrate import ode
    import numpy as np
    # Initialise constants and state variables
    states = create_states_array()
    variables = create_variables_array()
    initialise_variables(states,variables)
    # Set timespan to solve over
    voi = np.linspace(starttime, stoptime, steps)

    # Construct ODE object to solve
    r = ode(rhs)
    r.set_integrator('vode', method='bdf', atol=1e-06, rtol=1e-06, max_step=1)
    r.set_initial_value(states, voi[0])
    r.set_f_params(variables)

    # Solve model
    result = np.zeros((STATE_COUNT,steps))
    result[:,0] = states    
    for (i,t) in enumerate(voi[1:]):
        if r.successful():
            r.integrate(t)
            result[:,i+1] = r.y
            states = r.y
            #Perform event processing etc
            process_events(t,states,variables)
        else:
            break

    return (voi, result, variables)

import matplotlib.pyplot as plt
if __name__ == '__main__':
    t,r,v = solve_model()
    grid = plt.GridSpec({(len(self.nodegroups))}, 1, wspace=0.2, hspace=0.5)

    ix = 0
    for i in range({(len(self.nodegroups))}):
        for j in range(1):
            ax = plt.subplot(grid[i, j])
            ax.plot(r[ix,:])
            ax.title.set_text(f'{{ix//{(len(self.stateVec)//len(self.nodegroups))}+1}}')
            ix += {(len(self.stateVec)//len(self.nodegroups))}
            if ix+{(len(self.stateVec)//len(self.nodegroups))} > r.shape[0]:
                break
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.3)
    plt.show() 
'''
        return pycode.replace("1**2*", "").replace("-1.0*","-")

    def exportAsODEStepper(self,modelName):
        r"""Generate numerically solvable PHS
            d/dx (Ex) = (J-R) Q x - \hat{B}\hat{C}\hat{B}^T \hat{u} + \bar{B} \bar{u}
            \hat{y} = \hat{B}^T Q x
            \bar{y} = \bar{B}^T Q x
        
            Setup the code as a Python class, with the ability to step through time and
            set inputs
        """
        numconstants,phsconstants,constantsubs,nonlinearrhsterms,inputs,arrayedinputs,arraymapping,uCapterms,ucapdescriptive,nonlineararrayedrhsterms,nonlinearrhstermsdescriptive,arrayedrhs,invarraymapping,rhs_,ubaridxmap,ftuidmap,cleaninputs = self.generatePythonIntermediates()

        #Also handle constant subs
        revconstantsubs = dict()
        for k,v in constantsubs.items():
            revconstantsubs[v] = k
        nonlinearrhstermssub = dict()
        for k,v in nonlinearrhsterms.items():
            nonlinearrhstermssub[k] = sympy.simplify(v.xreplace(revconstantsubs))

        rhs = self.raw_rhs.xreplace(nonlinearrhstermssub)
    
        #Replace all symbolic parameters in hamiltonian with their values from composer.compositeparameters and constants
        parvals_ = dict()
        for k,v in self.reducedPHS.parameters.items():
            parvals_[k] = v['value'].xreplace(nonlinearrhstermssub)
        #some composite parameters also use symbols listed in composite parameters
        parvals = dict()
        for k,v in parvals_.items():
            parvals[k] = v.xreplace(parvals_)
        
        #Subtitute in cellHamiltonians - requires symbols
        arraymappingsym = {k:sympy.Symbol(v) for k,v in arraymapping.items()}
        cellhams = OrderedDict()
        externalInputEnergy = OrderedDict()
        totalInputEnergy = OrderedDict()
        ubaridxmapsym = {sympy.Symbol(k):sympy.Symbol(v) for k,v in ubaridxmap.items()}
        #For setting up energy calculations
        supportEnergyCalculations = OrderedDict()
        
        #Used for calculating total input energy contribution, sets all internal inputs to zero
        ubaridxmapzero = {sympy.Symbol(f"u_{s}"):0 for s in self.stateVec}
        inputStateSymbolMap = {sympy.Symbol(f"u_{k}"):v for k,v in arraymapping.items()}
        
        inputsidx = 0
        #Determine the hamiltonian without exterior inputs - compare this with one having the
        #external inputs included to determing the energetic contribution from the external inputs
        cix = 0
        for k,v in self.cellHamiltonians.items():
            cham = v['hamiltonian'].xreplace(self.reducedStateMap)
            #Subtitute for inputs
            noisubs = dict()
            for ui in v['u']:
                if inputs[inputsidx]!=0:
                    noisubs[ui] = 0.0

            noicham = cham.xreplace(noisubs) #Calculate hamiltonian without external inputs            
            notcham = noicham.xreplace(ubaridxmapzero) #Calculate the hamiltonian without any internal and external input contributions
            cdiff = sympy.simplify(cham - noicham)
            tdiff = sympy.simplify(cham - notcham)
                
            cellhams[k] = cham.xreplace(arraymappingsym).xreplace(ubaridxmapsym) 
            externalInputEnergy[k] = cdiff.xreplace(arraymappingsym).xreplace(ubaridxmapsym) 
            totalInputEnergy[k] = tdiff.xreplace(arraymappingsym).xreplace(ubaridxmapsym) 
            cix +=1
        
    
        pycode = f"""
# The content of this file was generated using the FTUWeaver

import numpy as np
from numpy import exp
from scipy.integrate import ode

__version__ = "0.0.1"

def heaviside(x):
    if x > 0:
        return 1.0
    return 0.0

def Abs(x):
    return np.fabs(x)


class {modelName}():
    STATE_COUNT = {len(self.stateVec)}
    VARIABLE_COUNT = {numconstants}
    CELL_COUNT  = {len(self.nodegroups)}
    COUPLING_PARAMETERS = {len(self.couplingconstants)}
    
    def __init__(self):
        self.states = np.zeros(self.STATE_COUNT)
        self.rates = np.zeros(self.STATE_COUNT)
        self.variables = np.zeros(self.VARIABLE_COUNT)
        self.time = 0.0
        self.odeintegrator = ode(lambda t,x : self.rhs(t,x))
        self.odeintegrator.set_integrator('vode', method='bdf', atol=1e-06, rtol=1e-06, max_step=1)
        self.odeintegrator.set_initial_value(self.states, self.time)       
        #Initialize variables
        states, variables = self.states, self.variables
"""
        # Do states first
        stateinitialvalues = dict()
        for k, v in self.reducedPHS.statevalues.items():
            stateinitialvalues[k] = v['value']
        
        for i,s in enumerate(self.reducedPHS.states):
            rv = sympy.simplify(s.subs(stateinitialvalues))
            try:
                stmt = f"        {arraymapping[self.stateVec[i,0]]} = {float(rv.name):6f}  #{rv.name}\n"
            except:
                stmt = f"        {arraymapping[self.stateVec[i,0]]} = {rv}  #{rv}\n"
            pycode += stmt
            
            
        for k, v in ubaridxmap.items():
            pycode += f"        {v} = 0.0 #{k} External input\n"
                    
        if len(ftuidmap)>0:
            for k,v in ftuidmap.items():
                pycode += f"        {v} = 1.0  #{k} This needs to be set for accurate simulation\n"
        # Do constant subs
        definedVariables = []
        for k, v in constantsubs.items():
            if v not in definedVariables:
                if v.name in arraymapping:
                    try:
                        stmt = f"        {arraymapping[v.name]} = {float(k):6f}  #{v}\n"
                    except:
                        stmt = f"        {arraymapping[v.name]} = {k}  #{v}\n"
                    pycode += stmt
                    definedVariables.append(v)
        for k, v in phsconstants.items():
            if k not in definedVariables:
                if k.name in arraymapping:
                    try:
                        stmt = f"        {arraymapping[k.name]} = {float(v['value']):6f}  #{k}\n"
                    except:
                        stmt = f"        {arraymapping[k.name]} = {v['value']}  #{k}\n"
                    pycode += stmt
                    definedVariables.append(k)


        pycode += "\n    def compute_variables(self,voi):\n        t=voi #mapping to t\n        states, rates, variables = self.states,self.rates,self.variables\n"
        # Do uCap terms
        for k, v in uCapterms.items():
            pycode += f"        #{ucapdescriptive[k]}\n"
            pycode += f"        {k} = {v}\n"

        # Do rhs
        pycode += "\n    def compute_rates(self,voi):\n        t=voi #mapping to t\n        states, rates, variables = self.states,self.rates,self.variables\n"
        # Do nonlinear terms - these depend on state values and therefore step size, os here instead of compute variables
        for k, v in nonlineararrayedrhsterms.items():
            pycode += f"        #{nonlinearrhstermsdescriptive[k]}\n"
            pycode += f"        {k} = {v}\n"
        for i, v in enumerate(arrayedrhs):
            pycode += f"        #\\dot{{{self.stateVec[i]}}} = {codegenerationutils._stringsubs(str(v),invarraymapping)} # {sympy.simplify(rhs[i,0])}\n"
            pycode += f"        rates[{i}] = {v}\n"

        # Do inputs
        pycode += f"\n    def compute_inputs(self,voi,inputs):\n        t,states,variables=voi,self.states,self.variables\n        #inputs size {len(self.stateVec)}\n"
        for i, v in enumerate(arrayedinputs):
            pycode += f"        # forstate[{i}] = {cleaninputs[i]}\n"
            pycode += f"        inputs[{i}] = {v}\n"

        # Compute Hamiltonian's for each cells
        # TODO Code has unresolved symbols
        pycode += f"\n    def compute_hamiltonian(self,cellHam):\n        t,states,variables=self.time,self.states,self.variables\n        #cellHam = np.zeros({len(self.nodegroups)})\n"
        cix = 0
        for k,v in cellhams.items():
            pycode += f"        cellHam[{cix}] = {v}\n"
            cix +=1
        pycode += f"\n        return cellHam\n"

        # Compute energy input from external inputs
        pycode += f"\n    def compute_external_energy(self,inputEnergy):\n        t,states,variables=self.time,self.states,self.variables\n        #inputEnergy = np.zeros({len(self.nodegroups)})\n"
        cix = 0
        for k,v in externalInputEnergy.items():
            pycode += f"        inputEnergy[{cix}] = {v}\n"
            cix +=1
        pycode += f"\n        return inputEnergy\n"

        # Compute energy input from internal and external inputs
        pycode += f"\n    def compute_total_input_energy(self,totalInputEnergy):\n        t,states,variables=self.time,self.states,self.variables\n        #totalInputEnergy = np.zeros({len(self.nodegroups)})\n"
        cix = 0
        for k,v in totalInputEnergy.items():
            pycode += f"        totalInputEnergy[{cix}] = {v}\n"
            cix +=1
        pycode += f"\n        return totalInputEnergy\n"
                
        # Provide external input variable names in comment to help support
        ubarcomment = ""
        for k, v in ubaridxmap.items():
            ubarcomment += f"        #    {k} -> {v}\n"

        pycode += f'''
    def process_time_sensitive_events(self,voi):
        """Method to process events such as (re)setting inputs, updating switches etc
        Unline process_events, this method is called in rhs calculation
        Useful to ensure that time sensitive inputs are set espcially if ode integrator timestep spans over the 
        input time. Note that this should be re-entrant i.e. not modify states, else this will
        lead to solver dependent behaviour, esp. solvers that use multiple steps
        The method is called before each rhs evaluation
        Args:
            voi (int) : Current value of the variable of integration (time)
            states (np.array): A vectors of model states
            variables (_type_): A vector of model variables
        """
        states, rates, variables = self.states,self.rates,self.variables
        #External input variables - listed to help code event processing logic
{ubarcomment}
        #Comment the line below (and uncomment the line after) to solve the model without event processing!    
        #raise("Process time sensitive events not implemented")
        variables[0] = 0.0
        if voi > 100 and voi < 110:
            variables[0] = 0.5        
        #Following needs to be performed to set internal inputs from current state values
        self.compute_variables(voi)    
            
    def process_events(self,voi):
        """Method to process events such as (re)setting inputs, updating switches etc
        The method is called after each successful ode step
        Args:
            voi (int) : Current value of the variable of integration (time)
        """
        #External input variables - listed to help code event processing logic
        states, rates, variables = self.states,self.rates,self.variables
{ubarcomment}
        #Comment the line below (and uncomment the line after) to solve the model without event processing!    
        #raise("Process events not implemented")


    def rhs(self, voi, states):
        self.states = states    
        #Perform (re)setting of inputs, time sensitive event processing etc
        self.process_time_sensitive_events(voi)    
        #Compute rates
        self.compute_rates(voi)
        return self.rates

    def step(self,step=1.0):
        if self.odeintegrator.successful():
            self.odeintegrator.integrate(step)
            self.time = self.odeintegrator.t
            self.states = self.odeintegrator.y
            #Perform event processing etc
            self.process_events(self.time)
        else:
            raise Exception("ODE integrator in failed state!")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    starttime=0
    stoptime=300
    steps=300
    finst = FTUStepper()
    voi = np.linspace(starttime, stoptime, steps)
    result = np.zeros((finst.STATE_COUNT,steps))
    result[:,0] = finst.states    
    for (i,t) in enumerate(voi[1:]):
        finst.step(t)
        result[:,i+1] = finst.states      
    fig = plt.figure(figsize=(50, 50))
    grid = plt.GridSpec({(len(self.nodegroups)+1)//3}, 3, wspace=0.2, hspace=0.5)

    ix = 0
    for i in range({(len(self.nodegroups)+1)//3}):
        for j in range(3):
            ax = plt.subplot(grid[i, j])
            ax.plot(result[ix,:])
            ax.title.set_text(f'{{ix//{(len(self.stateVec)//len(self.nodegroups))}+1}}')
            ix += {(len(self.stateVec)//len(self.nodegroups))}
            if ix+{(len(self.stateVec)//len(self.nodegroups))} > result.shape[0]:
                break
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.3)
    fig.savefig(f"FTUStepper_results.png",dpi=300)
    plt.show()         
'''

        return pycode.replace("1**2*", "").replace("-1.0*","-")

    def compose(self, substituteParameters=True):
        """
        Create the composite PHS matrices for compostion description that has been loaded
        Requires the phs for the different cell types to be loaded see loadPHSDefinition, loadCellTypePHSMap
        Requires the phs interconnection to be loaded see loadCompositionFile
        Ideally all interconnection u's and y's to be of the same size, if some phs have more u's,
        then select the ones that need to be used in the interconnection see setCellTypeUSplit
        """
        # calculate matrix dimensions
        ptype = [] #Store the phs type
        ridxs = [] #To store the row index of phs matrix on full matrix
        cidxs = [] #To store the col index of phs matrix on full matrix
        jmat = []
        qmat = []
        emat = []
        rmat = []
        bcapmat = []
        nbcapmat = []
        lashmat = []
        bdasmat = []
        lashzeroflag = []
        jrsize = 0
        jcsize = 0
        brsize = 0
        bcsize = 0
        networkLap = dict()
        networkAdj = dict()
        self.compositeparameters = dict()

        #self.substituteParameters = substituteParameters
        #self.requiredFTUparameters = False
        for n, g in self.composer.networkGraphs.items():
            try:
                networkLap[n] = nx.laplacian_matrix(g.to_undirected()).todense()
            except:
                #self.composer.requiredFTUparameters = True
                ug = g.to_undirected()
                lap = sympy.zeros(ug.number_of_nodes(),ug.number_of_nodes())
                nodes = list(ug.nodes())
                for i,nd in enumerate(nodes):
                    nedges = list(ug.edges(nd,data='weight'))
                    ndeg = len(nedges)
                    lap[i,i] = ndeg 
                    for ed in nedges:
                        if ed[0]==nd:
                            tar = nodes.index(ed[1])
                        else:
                            tar = nodes.index(ed[0])
                        lap[i,tar] = -sympy.sympify(ed[2])
                networkLap[n] = lap
                
            # Construct node connectivity matrix, should be consistent with KCL
            nadj = np.zeros((g.number_of_nodes(), g.number_of_nodes()))
            for nd, nbrdict in g.adjacency():
                nadj[
                    nd, nd
                ] = (
                    -1
                )  # -len(nbrdict) # src (-1) -> target (+1) , not using in-degree as src is associated with a quantity
                for ni, wtd in nbrdict.items():
                    nadj[nd, ni] = +1
            networkAdj[n] = nadj

        hamiltonian = None
        ucapVec = []
        ycapVec = []
        ucapVectm1 = []  # At t-1
        ycapVectm1 = []  # At t-1
        stateVec = []
        rhsvec = []
        nodePHS = dict()
        statevalues = dict()

        self.cellHamiltonians = OrderedDict() #Ensure insertion order is maintained
        self.nodePHSData = dict()
        for k in self.composer.inodeIndexes:
            v = self.composer.nodeData[k]
            phs = self.composer.phsInstances[self.composer.cellTypePHS[k]]
            self.nodePHSData[k] = phs #Used for generating input hooks
            statevalues.update(phs.statevalues)
            nodePHS[k] = phs
            #self.compositeparameters.update(phs.parameters)
            r, c = phs.J.shape
            jrsize += r
            jcsize += c
            r, c = phs.B.shape
            brsize += r
            bcsize += c
            if hamiltonian is not None:
                hamiltonian = sympy.Add(hamiltonian, phs.hamiltonian)
            else:
                hamiltonian = phs.hamiltonian
            self.cellHamiltonians[k] = phs
            ptype.append(self.composer.cellTypePHS[k])
            jmat.append(phs.J)
            qmat.append(phs.Q)
            emat.append(phs.E)
            rmat.append(phs.R)
            for i in range(phs.u.rows):
                stateVec.append(f"{phs.states[i,0]}")
                # k in encoded in phs states
                rhsvec.append(f"Del(H_{phs.states[i,0]})")
                if not v["bsplit"][i]:
                    # k in encoded in u during instantiation
                    usym = f"{phs.u[i,0]}_{i}"
                    ysym = "y" + usym[1:]

                    ucapVec.append(sympy.Symbol(usym))
                    ycapVec.append(sympy.Symbol(ysym))
                    ucapVectm1.append(sympy.sympify(usym + "(t-1)"))
                    ycapVectm1.append(sympy.sympify(ysym + "(t-1)"))
                else:
                    ucapVec.append(sympy.Symbol("0"))
                    ycapVec.append(sympy.Symbol("0"))
                    ucapVectm1.append(sympy.Symbol("0"))
                    ycapVectm1.append(sympy.Symbol("0"))

            bcap, bdas = phs.split(v["bsplit"])
            # Unlike the paper, C matrix is a real matrix and not connectivity matrix, therefore 
            # Metric contributions from bcap is determined by C matrix
            # Only active/inactive contribution from component 
            # information is necessary - set all non zero entries in Bcap to 1
            for ix in range(bcap.shape[0]):
                for iy in range(bcap.shape[1]):
                    if bcap[ix, iy] != 0:
                        bcap[ix, iy] = 1  # sympy.Symbol("1")

            # Include composite's Bhat to bcap
            if (
                phs.Bhat.shape[0] == bcap.shape[0]
                and phs.Bhat.shape[1] == bcap.shape[1]
            ):
                # Set the Bhat's nonzeros to bcap
                for i in range(phs.Bhat.shape[0]):
                    for j in range(phs.Bhat.shape[1]):
                        if not sympy.is_zero(phs.Bhat[i, j]):
                            bcap[i, j] = phs.Bhat[i, j]
            elif phs.Bhat.shape[0] != 0 and phs.Bhat.shape[1] != 0:
                raise ("Composite PHS Bhat and Bcap are not of the same shape!")

            bcapmat.append(
                bcap
            )  # Since C is scaled by E, Bcap does not need to be scaled interior u = B C B^T Q x
            nbcapmat.append((-bcap).T)
            # The dimensions need to match the JMatrix as Bdash*udash is summed with (J-R) Q x
            bash = sympy.zeros(phs.B.shape[0], self.composer.uVec.shape[1])
            lash = sympy.zeros(phs.B.shape[0], self.composer.uVec.shape[1])  # laplacian

            # boundaryinputs has the inputs that the nodes receives
            # boundaryinputs[k][0] - component names
            # boundaryinputs[k][1] - name index into u vector
            # boundaryinputs[k][2] - network is dissipative or not

            lashupdated = False
            if k in self.composer.boundaryinputs:
                for (i, uin) in enumerate(self.composer.boundaryinputs[k][0]):
                    comp = self.composer.boundaryinputs[k][1][i]
                    # Subscript of uin contains the node to which k is connected
                    # prefix is given by phs.u[comp]
                    # if dissipative get the laplacian weight using k, uin_subscript
                    # us = Symbolics.variable(uin)
                    uix = -1
                    u_ix = -1  # Get the index in uVec
                    for j in range(self.composer.uVec[0, :].cols):
                        if self.composer.uVec[0, j] == uin:
                            uix = self.composer.uVec[1, j]
                            u_ix = j
                            break
                    # uVec[1,:] has the names, uVec[2,:] has the component indexes

                    if self.composer.boundaryinputs[k][2][i] == False:
                        bash[uix, u_ix] = phs.B[comp, comp]
                    else:
                        nid = f"{phs.usplit[comp]}"
                        if nid in self.composer.dissipativeNets:  # If dissipative
                            # uin is the node that provides the input - the suffix provides the node id and the index into the lapacian/adjacency matrix
                            ulc = phs.u[comp]
                            ulc = str(phs.u[comp]).find("_")
                            suffix = int(str(uin)[ulc + 1:])
                            wt = networkLap[nid][k, suffix]
                            lash[comp, u_ix] = wt
                            lashupdated = True
                        else:  # Has input in bash
                            # Check uix and comp
                            bash[uix, u_ix] = phs.B[comp, comp]

            bdasmat.append(bash)
            lashmat.append(lash)
            lashzeroflag.append(lashupdated)
        # Set the vector elements in the correct order
        if self.composer.uVec.cols > 1:
            self.uVecSymbols = self.composer.uVec[0, self.composer.uVec[1, :]]
        else:
            # When there is a single input symbol
            self.uVecSymbols = sympy.Matrix([self.composer.uVec[0]])
        self.stateVec = stateVec
        self.statevalues = statevalues
        self.rhsVec = rhsvec
        self.xVec = sympy.Matrix.vstack(
            sympy.Matrix(stateVec), sympy.Matrix(ucapVec), sympy.Matrix(ycapVec))
        self.rVec = sympy.Matrix.vstack(
            sympy.Matrix(rhsvec), sympy.Matrix(ucapVectm1), sympy.Matrix(ycapVectm1)
        )
        self.hamiltonian = hamiltonian
        """
            Create full connection matrix - get the asymmetric part for C and the other for R

            Dissipative networks are expected to solve Reaction diffusion equations of the form

            dx/dt = phs(x) + ∇x , where ∇x is the diffusive contributions computed from the weighted graph laplacian

            When the PHS is of the form

            E dx/dt = (J-R)Qx + Bu then the above equation requires a scaling factor for the ∇x term

            E dx/dt = phs(x) + E ∇x

            all terms of the lapacian matrix for the component are scaled by E
        """
        Cx = sympy.zeros(brsize, bcsize)
        roffset = 0
        for (i, b) in enumerate(self.composer.inodeIndexes):
            # phs = self.phsInstances[self.cellTypePHS[b]]
            phs = nodePHS[b]
            # Emat = phs.E
            usplit = phs.usplit
            if phs.C.shape[0] != 0 and phs.C.shape[1] != 0:
                # Scaling by compostite's E done at the time of construction
                Cx[
                    roffset: roffset + phs.C.shape[0],
                    coffset: coffset + phs.C.shape[1],
                ] = phs.C

            for (j, n) in enumerate(usplit):
                nid = n
                if (
                    nid in networkAdj
                ):  # if u has zeros, networks for them will not be present
                    lap = networkAdj[nid]
                    if self.composer.dissipativeNets[nid]:
                        lap = networkLap[nid]
                    coffset = 0
                    for (k, x) in enumerate(self.composer.inodeIndexes):
                        phsx = nodePHS[x]
                        lc = phsx.B.shape[1]
                        if self.composer.dissipativeNets[nid]:
                            try:
                                Cx[roffset + j, coffset + j] = lap[b, x]
                            except:  # DomainError
                                continue
                        else:
                            # Setting up a symmetrix matrix for connection
                            if k > i:
                                try:
                                    Cx[roffset + j, coffset + j] = lap[b, x]
                                    Cx[coffset + j, roffset + j] = -lap[b, x]
                                except:  # DomainError
                                    continue
                            elif k == i:
                                try:
                                    Cx[roffset + j, coffset + j] = lap[b, x]
                                except:  # DomainError
                                    continue
                        coffset += lc
            roffset += phs.B.shape[0]

        self.uyConnectionMatrix = Cx

        sym = (Cx + Cx.T) / 2
        skewsym = (Cx - Cx.T) / -2  # We need -C
        self.uyConnectionMatrixComputed = True

        # calculate the full matrix size
        jr = 2 * jrsize + skewsym.shape[0]
        jc = 2 * jcsize + skewsym.shape[1]

        self.Jcap = sympy.zeros(jrsize, jcsize)
        self.Rcap = sympy.zeros(jrsize, jcsize)
        self.Ecap = sympy.zeros(jrsize, jcsize)
        self.Qcap = sympy.zeros(jrsize, jcsize)
        self.Bcap = sympy.zeros(jrsize, jcsize)
        self.nBcapT = sympy.zeros(jrsize, jcsize)
        self.Bdas = sympy.zeros(jrsize, self.composer.uVec.shape[1])
        self.Cmatrix = skewsym
        self.Lmatrix = sym

        jr, jc = 0, 0
        for (i, jx) in enumerate(jmat):
            ridxs.append(jr)
            cidxs.append(jc)
            self.Jcap[jr: jr + jx.shape[0], jc: jc + jx.shape[1]] = jx
            self.Rcap[jr: jr + jx.shape[0], jc: jc + jx.shape[1]] = rmat[i]
            self.Ecap[jr: jr + jx.shape[0], jc: jc + jx.shape[1]] = emat[i]
            self.Qcap[jr: jr + jx.shape[0], jc: jc + jx.shape[1]] = qmat[i]
            self.Bcap[jr: jr + jx.shape[0], jc: jc + jx.shape[1]] = bcapmat[i]
            self.nBcapT[jr: jr + jx.shape[0],
                        jc: jc + jx.shape[1]] = nbcapmat[i]
            # if laplacian is nonzero then Bdas should be lashmat
            if not lashzeroflag[i]:
                self.Bdas[jr: jr + bdasmat[i].shape[0], :] = bdasmat[i]
            else:
                self.Bdas[jr: jr + bdasmat[i].shape[0], :] = lashmat[i]

            jr += jx.shape[0]
            jc += jx.shape[1]

        freevars = self.Jcap.free_symbols
        freevars = freevars.union(self.Rcap.free_symbols)
        freevars = freevars.union(self.Bcap.free_symbols)
        freevars = freevars.union(self.Bdas.free_symbols)
        freevars = freevars.union(self.Qcap.free_symbols)
        freevars = freevars.union(self.hamiltonian.free_symbols)
        freevars = freevars.union(self.uVecSymbols.free_symbols)

        for k, v in self.compositeparameters.items():
            freevars = freevars.union(k.free_symbols)
            freevars = freevars.union(v["value"].free_symbols)

        self.phstypes = ptype
        self.ridxs = ridxs
        self.cidxs = cidxs
        self.phsclassstructure = dict()
        for k,v in self.composer.phsInstances.items():
            self.phsclassstructure[k] = {'rows':v.J.rows,'cols':v.J.cols}

        self.compositePHS = SymbolicPHS(
            self.Jcap,
            self.Rcap,
            self.Bcap,
            self.Bdas,
            self.Qcap,
            self.Ecap,
            self.uyConnectionMatrix,
            self.hamiltonian,
            stateVec,
            self.uVecSymbols,
            sympy.zeros(self.composer.uVec.shape[0], 1),
            freevars,
            self.compositeparameters,
            self.statevalues,
        )


import pickle        
if __name__ == '__main__':
    with open(r'D:\12Labours\GithubRepositories\FTUUtils\tests\data\Temp\compositephs.pkl','rb') as js:
        phsr = pickle.load(js)
        rphs = ReducePHS(phsr)
        ng = {0: np.array([ 0,  5, 10, 15, 20]), 
              1: np.array([ 1,  6, 11, 16, 21]), 
              2: np.array([ 2,  7, 12, 17, 22]), 
              3: np.array([ 3,  8, 13, 18, 23]), 
              4: np.array([ 4,  9, 14, 19, 24])
              }
        
        rphs.setClusters(ng)
        rphs.exportForFitting()
        #pys = rphs.exportAsPython()
        #pys = rphs.exportAsODEStepper('test')
        #with open(r'D:\12Labours\GithubRepositories\FTUUtils\tests\data\Temp\reduced.py','w') as rpy:
        #    print(pys,file=rpy)
        #rphs.compose()