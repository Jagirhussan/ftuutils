#Package import for handling json data
import json
#Package imports to support FTU generation
from ftuutils import graphutils, phsutils
from copy import deepcopy
import networkx as nx
#Define the list of phs to be used by the FTU


phsval = '{"parameter_values":{"mu":{"value":"0.02","units":"dimensionless"},"a":{"value":"0.15","units":"dimensionless"},"M":{"value":"v*v*v/3-v","units":"dimensionless"},"g":{"value":"1.3","units":"dimensionless"}},"Hderivatives":{"cols":1,"rows":2,"elements":["v/2","w/2"]},"hamiltonianLatex":"v**2/2 + w**2/2","hamiltonian":"1/2*v**2 + 1/2*w**2","portHamiltonianMatrices":{"matJ":{"cols":2,"rows":2,"elements":["0","-1","1","0"]},"matR":{"cols":2,"rows":2,"elements":["M","0","0","g"]},"matB":{"cols":2,"rows":2,"elements":["a","0","0","1"]},"matBhat":{"cols":0,"rows":0,"elements":[]},"matQ":{"cols":2,"rows":2,"elements":["1","0","0","1"]},"matE":{"cols":2,"rows":2,"elements":["1/a","0","0","1/mu"]},"matC":{"cols":0,"rows":0,"elements":[]},"u":{"cols":1,"rows":2,"elements":["istim","beta"]},"u_connect2boundary":{"cols":1,"elements":[false,false],"rows":2}},"stateVector":{"cols":1,"rows":2,"elements":["v","w"]},"state_values":{"v":{"value":-1.2,"units":"dimensionless"},"w":{"value":0.0,"units":"dimensionless"}},"isphenomenological":false,"success":true}'  
phsinstance1 = json.loads(phsval)
phsinstance2 = deepcopy(phsinstance1) #Do a deepcopy else same instance is shared and issues with network splits will crop up

phstypes = {'FHN1':phsinstance1,'FHN2':phsinstance2}

#To use the phs instances loaded in the window/PHS tab use
#phstypes = phsutils.getAllPHSDefinitions()

#Here we use the first phs as the default phs for all cells

g = graphutils.Lattice2D(3,3,'FHN1')
g.setFibreConductivity(1.5)
g.setSheetConductivity(0.9)    

#Setup FHN2 types
nxg = g.getGraph()
ctypes = nx.get_node_attributes(nxg,"phs")
for ein in [4,7,8,9]:
    ctypes[ein] = 'FHN2'
nx.set_node_attributes(nxg,ctypes,"phs")

#Provide a dictionary to store connection information
phsdata = {}

#Specify for each PHS class, for each input component the network on which it connects
phsdata = phsutils.connect(phsdata , 'FHN1','istim',1) #Connection on u
phsdata = phsutils.connect(phsdata , 'FHN2','istim',1) #Connection on u

#Boundary connections can be specified as below. As a convention, boundary networks are negatively numbered
phsdata = phsutils.connectToBoundary(phsdata, 'FHN1','beta',-1) #Boundary connection for beta
phsdata = phsutils.connectToBoundary(phsdata, 'FHN2','beta',-2) #Boundary connection for beta

for ein in [49]:
    phsdata = phsutils.addExternalInput(phsdata,ein,'istim',-3)

#Set which networks are dissipative and add the information to the phsdata dictionary

networkDissipation = {1:True}
networkNames = {1:"ucap",-1:"threshold",-2:"autonodes",-3:"ubar"}

phsdata["networkNames"] = networkNames
phsdata["networkDissipation"] = networkDissipation

composer = g.composeCompositePHS(nxg,phstypes,phsdata,substituteParameters=False)
#Load the python code into the composition pythonic phs element
pcode = composer.exportAsPython()
with open(r'D:\12Labours\GithubRepositories\FTUUtils\tests\FHNtest\testcontrol.py','w') as pyf:
    print(pcode,file=pyf)