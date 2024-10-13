import numpy as np
import json, os

from ftuutils import phsutils 

points = None
datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"../examples/data")
with open(os.path.join(datadir,"positions.npy"), 'rb') as f:
    earr = np.load(f)
    xarr = np.load(f)
    points = np.load(f)

np.random.seed(0)
pids = np.arange(points.shape[0])
np.random.shuffle(pids)
maxpoints = pids.shape[0]//12
print(f"Selecting {maxpoints} of {pids.shape[0]}")
spids = pids[:maxpoints]
points = points[spids,:]

from ftuutils.base import FTUDelaunayGraph 
#Set edge weights based on orientation with respect to conductivity tensor
conductivity_tensor = np.array([1.2,0.9,0.5]) #Fibre sheet normal
g = FTUDelaunayGraph(points,"APN",conductivity_tensor)

# Determine nodes that will be on the boundary i.e. nodes that will communicate with the environment. These nodes need not necessarily be on the physical boundary of the discrete cell network

#Stimulus block is along the left wall
#Find nodes close to x start
left = np.min(points,axis=1)
leftpoints = np.isclose(points[:,0],left[0])
simb = []
#Get the nodes that are on the left edge
for i,v in enumerate(leftpoints):
    if v:
        simb.append(i)       
print("Selected input nodes",simb)

# Given a graph describing the cellular interconnections and a set of nodes that exchange energy (input nodes), we can start constructing an FTU (specifically the discrete part of it).
# 
# The example below uses a modified form of AlievPanfilov EC model that has membrane voltage and active tension as state variables. The model also has a approximate Hamiltonian description.

#Create a dictionary to store connection information
phsdata = {}
#FTUGraph type assigns a network id, we will use this network for membrane current exchange
defaultNetworkId = g.getDefaultNetworkID()
#Specify for each PHS class, for each input component the network on which it connects
phsdata = phsutils.connect(phsdata , 'APN','i_1',defaultNetworkId) #Connection on u - membrane voltage, network weight has conductivity encoded

# #Boundary connections are specified as below.
# #As a convention, boundary networks are negatively numbered
# 
# #Nodes that receive external input are specified as below
# # - these are boundary connections and the network number is negative
# #All external inputs with the same network number share the same input variable,
# #If different input variables are required for each 
# #external input, provide different network numbers for the nodes
# #this may be useful for much finer control of inputs.
# #In this example we assume that all the input nodes are excited by the same stimulus


for ein in simb:
    phsdata = phsutils.addExternalInput(phsdata,ein,'i_1',-2)

#Indicate which networks are dissipative and add the information to the phsdata dictionary
networkDissipation = {defaultNetworkId:True}
networkNames = {defaultNetworkId:"ucap",-2:"ubar"}
#The dictionary keys networkNames and networkDissipation are keywords for composition logic and must be adhered
phsdata["networkNames"] = networkNames
phsdata["networkDissipation"] = networkDissipation

# Provide Cell type descriptions, these can be constructed by hand, by FTUWeaver or through libbondgraph api
# Below we create a single celltype with the name `APN`

phsval = r'{"parameter_values":{"eps0":{"value":"0.002","units":"dimensionless"},"k":{"value":"8.0","units":"dimensionless"},"a":{"value":"0.13","units":"dimensionless"},"c0":{"value":"0.016602","units":"dimensionless"},"ct":{"value":"0.0775","units":"dimensionless"},"mu1":{"value":"0.2","units":"dimensionless"},"mu2":{"value":"0.3","units":"dimensionless"},"x1":{"value":"0.0001","units":"dimensionless"},"x2":{"value":"0.78","units":"dimensionless"},"x3":{"value":"0.2925","units":"dimensionless"},"eta1":{"value":"Tai*(Heaviside(u-x2)*Heaviside(x3-Tai)*(x1-c0)+c0)","units":"dimensionless"},"eta2":{"value":"Heaviside(u-x2)*Heaviside(x3-Tai)*(x1-c0)+c0","units":"dimensionless"},"sigma":{"value":"0.042969","units":"dimensionless"},"sqpi":{"value":"sqrt(2*3.141459265)","units":"dimensionless"},"kV":{"value":"exp(-0.5*((u-1)/sigma)**2)/(sigma*sqpi)","units":"dimensionless"},"U":{"value":"k*u*(u-a)*(1-u)-u*v","units":"dimensionless"},"V":{"value":"(eps0+(v*mu1)/(u+mu2))*(-v-k*u*(u-a-1))","units":"dimensionless"}},"Hderivatives":{"cols":1,"rows":4,"elements":["Tai","Ta","u","v"]},"hamiltonianLatex":"- Ta c_{0} kV + \\frac{Tai^{2} eta1}{2} - \\frac{eps0 k u^{3}}{3} + \\frac{eps0 k u^{2} \\left(a + 1\\right)}{2} - i_{1} v","hamiltonian":"eps0*k*((a+1)*u**2)/2 - eps0*k*u**3/3 + eta1*Tai**2/2 - c0*kV*Ta - (i_1)*v","portHamiltonianMatrices":{"matJ":{"cols":4,"rows":4,"elements":["0","- eta1/2","c0*kV/2","0","eta1/2","0","0","0","-c0*kV/2","0","0","0","0","0","0","0"]},"matR":{"cols":4,"rows":4,"elements":["c0","-eta1/2","-c0*kV/2","0","-eta1/2","eta2","0","0","-c0*kV/2","0","-U","0","0","0","0","-V"]},"matB":{"cols":4,"rows":4,"elements":["0","0","0","0","0","0","0","0","0","0","1/ct","0","0","0","0","0"]},"matBhat":{"cols":0,"rows":0,"elements":[]},"matQ":{"cols":4,"rows":4,"elements":["1","0","0","0","0","1","0","0","0","0","1","0","0","0","0","1"]},"matE":{"cols":4,"rows":4,"elements":["1","0","0","0","0","1","0","0","0","0","1/ct","0","0","0","0","1/ct"]},"matC":{"cols":0,"rows":0,"elements":[]},"u":{"cols":1,"rows":4,"elements":["0","0","i_1","0"]},"u_connect2boundary":{"cols":1,"elements":[false,false,false,false],"rows":4}},"stateVector":{"cols":1,"rows":4,"elements":["Tai","Ta","u","v"]},"state_values":{"Tai":{"value":0.000,"units":"volt"},"Ta":{"value":0.001,"units":"dimensionless"},"u":{"value":0,"units":"dimensionless"},"v":{"value":0.03604,"units":"dimensionless"}},"u_dims":{"i_1":{"units":"dimensionless"}},"isphenomenological":false,"success":true}'
phstypes = {'APN':json.loads(phsval)}

# With the above information and a graph with appropriate node and edge attributes, we can compose a FTU.


#Get the graph
G = g.getGraph()

#Call the FTU composition logic to create a FTU with above information
composer = g.composeCompositePHS(G,phstypes,phsdata)

cfile = composer.exportAsCellML()
print("\n\n")
print(cfile)
