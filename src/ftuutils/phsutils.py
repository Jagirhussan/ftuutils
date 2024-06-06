import json
htmlinstance = True
try:
    import js
except:
    htmlinstance = False
    
def getAllPHSDefinitions():
    if htmlinstance:
        result = {}
        phsrows = js.document.getElementById("phslist").rows
        for row in phsrows:
            phs = json.loads(row.cells[0].children[0].dataset.init)
            if 'phs' in phs:
                result[row.cells[0].children[0].id] = phs['phs']
            else:
                result[row.cells[0].children[0].id] = phs
            
        return result
    else:
        raise("Running on python instance. Not supported!!")

def setPHSComponentNetwork(phs,component,netid):
    if 'u_split' in phs['portHamiltonianMatrices']:
        phs['portHamiltonianMatrices']['u_split']['elements'][component] = netid
    else:
        ups = phs['portHamiltonianMatrices']['u']
        for x in range(ups['rows']):
            ups['elements'][x] = ''
        ups['elements'][component] = netid
        phs['portHamiltonianMatrices']['u_split'] = ups
    return phs


def connect(phsconnections,phs1,phs1comp,network):
    if "connections" not in  phsconnections:
        phsconnections["connections"] = dict()
        
    if phs1 not in phsconnections["connections"]:
        phsconnections["connections"][phs1] = dict()
    phsconnections["connections"][phs1][phs1comp] = network
    return phsconnections
    
def connectToBoundary(phsconnections,phs1,phs1comp,network):
    if "bdryconnections" not in  phsconnections:
        phsconnections["bdryconnections"] = dict()
    
    if phs1 not in phsconnections["bdryconnections"]:
        phsconnections["bdryconnections"][phs1] = dict()
    phsconnections["bdryconnections"][phs1][phs1comp] = network
    return phsconnections

def addExternalInput(phsconnections,node,component,network=-1):
    if "externalinputs" not in  phsconnections:
        phsconnections["externalinputs"] = dict()
    
    if node not in phsconnections["externalinputs"]:
        phsconnections["externalinputs"][node] = []
        
    phsconnections["externalinputs"][node].append([component,network])
    return phsconnections