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
        
        Pc = sympy.zeros(rowoffset,pcols)
        pcolix = 0
        
        for n,v in nodegroups.items():
            phs_ = phstructure[phstypes[v[0]]]
            phsel = sympy.eye(phs_['rows'])
            for vv in v:
                Pc[rowindexs[vv]:rowindexs[vv]+phs_['rows'],pcolix:pcolix+phs_['rows']] = phsel
            pcolix += phs_['cols']
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
        print(sympy.latex(Pc))
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
        # Generate metedata
        variabledescription = 'VOI_INFO = {"name": "t", "units": "second", "component": "main", "type": VariableType.VARIABLE_OF_INTEGRATION}\n'
        variabledescription += "STATE_INFO = [\n"
        for k, v in self.reducedPHS.statevalues.items():
            variabledescription += f'\t{{"name": "{k}", "units": "{v["units"]}", "component": "main", "type": VariableType.STATE}},\n'
        variabledescription += "]\n\nVARIABLE_INFO = [\n"
        for k, v in ubaridxmap.items():
            #TODO get the dimension
            variabledescription += f'\t{{"name": "{k}", "units": "dimensionless", "component": "main", "type": VariableType.EXTERNAL_INPUT}},\n'        
        if len(ftuidmap)>0:
            for k,v in ftuidmap.items():
                variabledescription += f'\t{{"name": "{k}", "units": "dimensionless", "component": "main", "type": VariableType.CONSTANT}},\n'                    
        
        # Maintain this order when creating variables
        # Do constant subs, constant subs will have multiple values for the same constant due to precision
        definedNames = []
        for k, v in constantsubs.items():
            if v not in definedNames:
                if not v.name.startswith("-"):
                    variabledescription += f'\t{{"name": "{v}", "units": "dimensionless", "component": "main", "type": VariableType.CONSTANT}},\n'
                    definedNames.append(v)
        for k,v in phsconstants.items():
            if k.name in arraymapping:
                vunit = v['units']
                variabledescription += f'\t{{"name": "{k}", "units": "{vunit}", "component": "main", "type": VariableType.CONSTANT}},\n'                    


        # TODO compute the units of calculated terms
        # Do uCap terms
        for v in stateVec:
            variabledescription += f'\t{{"name": "u_{v}", "units": "dimensionless", "component": "main", "type": VariableType.INTERNAL_INPUT}},\n'
        for s in self.reducedPHS.u.free_symbols:
            variabledescription += f'\t{{"name": "{s}", "units": "dimensionless", "component": "main", "type": VariableType.EXTERNAL_INPUT}},\n'

        # Do nonlinear terms
        for k, v in nonlinearrhsterms.items():
            variabledescription += f'\t{{"name": "{k}", "units": "dimensionless", "component": "main", "type": VariableType.ALGEBRAIC}},\n'
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
        
        for i,s in enumerate(self.reducedPHS.states):
            rv = sympy.simplify(s.subs(stateinitialvalues))
            try:
                stmt = f"\t{arraymapping[stateVec[i,0]]} = {float(rv.name):6f}  #{rv.name}\n"
            except:
                stmt = f"\t{arraymapping[stateVec[i,0]]} = {rv}  #{rv}\n"
            pycode += stmt
            
        # for k, v in self.reducedPHS.statevalues.items():
        #     try:
        #         stmt = f"\t{arraymapping[k.name]} = {float(v['value']):6f}  #{k}\n"
        #     except:
        #         stmt = f"\t{arraymapping[k.name]} = {v['value']}  #{k}\n"
        #     pycode += stmt
        for k, v in ubaridxmap.items():
            pycode += f"\t{v} = 0.0 #{k} External input\n"
                    
        if len(ftuidmap)>0:
            for k,v in ftuidmap.items():
                pycode += f"\t{v} = 1.0  #{k} This needs to be set for accurate simulation\n"

        # Do constant subs
        definedVariables = []
        for k, v in constantsubs.items():
            if v not in definedVariables:
                if v.name in arraymapping:
                    try:
                        stmt = f"\t{arraymapping[v.name]} = {float(k):6f}  #{v}\n"
                    except:
                        stmt = f"\t{arraymapping[v.name]} = {k}  #{v}\n"
                    pycode += stmt
                    definedVariables.append(v)
        for v, k in phsconstants.items():
            if v not in definedVariables:
                if v.name in arraymapping:
                    try:
                        stmt = f"\t{arraymapping[v.name]} = {float(k['value']):6f}  #{v}\n"
                    except:
                        stmt = f"\t{arraymapping[v.name]} = {k['value']}  #{v}\n"
                    pycode += stmt
                    definedVariables.append(v)                
                    
        pycode += "\ndef compute_computed_constants(variables):\n\tpass\n\n"
        pycode += "def compute_variables(voi, states, rates, variables):\n\tt=voi #mapping to t\n"
        # Do uCap terms
        for k, v in uCapterms.items():
            pycode += f"\t#{ucapdescriptive[k]}\n"
            pycode += f"\t{k} = {v}\n"

        # Do rhs
        pycode += "\ndef compute_rates(voi, states, rates, variables):\n\tt=voi #mapping to t\n"
        # Do nonlinear terms - these depend on state values and therefore step size, os here instead of compute variables
        for k, v in nonlineararrayedrhsterms.items():
            pycode += f"\t#{nonlinearrhstermsdescriptive[k]}\n"
            pycode += f"\t{k} = {v}\n"
        for i, v in enumerate(arrayedrhs):
            pycode += f"\t#\dot{{{self.stateVec[i]}}} = {codegenerationutils._stringsubs(str(v),invarraymapping)} # {sympy.simplify(rhs[i,0])}\n"
            pycode += f"\trates[{i}] = {v}\n"

        # Do inputs
        pycode += "\ndef compute_inputs(voi,inputs,states,variables):\n"
        for i, v in enumerate(arrayedinputs):
            pycode += f"\t# cell[{i}] = {cleaninputs[i]}\n"
            pycode += f"\tinputs[{i}] = {v}\n"
        # Provide external input variable names in comment to help support
        ubarcomment = ""
        for k, v in ubaridxmap.items():
            ubarcomment += f"    #\t{k} -> {v}\n"
        pycode += f'''
from math import exp

def process_time_sensitive_events(voi, states, rates, variables):
    """Method to process events such as (re)setting inputs, updating switches etc
        Unline process_events, this method is called in rhs calculation
        Useful to ensure that time sensitive inputs are set espcially if ode integrator timestep spans over the 
        input time. Note that this should be re-entrant i.e. not modify states, else this will
        lead to solver dependent behaviour, esp. solvers that use multiple steps
        The method is called before each rhs evelauation
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
    grid = plt.GridSpec({(len(self.nodegroups)+1)//3}, 3, wspace=0.2, hspace=0.5)

    ix = 0
    for i in range({(len(self.nodegroups)+1)//3}):
        for j in range(3):
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
            ubarcomment += f"        #\t{k} -> {v}\n"

        pycode += f'''
    def process_time_sensitive_events(self,voi):
        """Method to process events such as (re)setting inputs, updating switches etc
        Unline process_events, this method is called in rhs calculation
        Useful to ensure that time sensitive inputs are set espcially if ode integrator timestep spans over the 
        input time. Note that this should be re-entrant i.e. not modify states, else this will
        lead to solver dependent behaviour, esp. solvers that use multiple steps
        The method is called before each rhs evelauation
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
        #pys = rphs.exportAsPython()
        # pys = rphs.exportAsODEStepper('test')
        # with open(r'D:\12Labours\GithubRepositories\FTUUtils\tests\data\Temp\reduced.py','w') as rpy:
        #     print(pys,file=rpy)