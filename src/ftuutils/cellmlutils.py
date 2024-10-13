'''
Logic for translating a composite FTU to cellml
Module depends on llnl units library, used for infering physical dimensions from strings
the library is compiled to python through setup.py

'''

import numpy as np
import json, os, sympy, re
from libcellml import Analyser, Component, Model, Printer, Units, Validator, Variable, Issue
from ftuutils_units import getBaseSIUnits
#Units predefined in cellml
defaultcellmlunits = ["ampere","becquerel","candela","coulomb","dimensionless",
                      "farad","gram","gray","henry","hertz","joule","katal",
                      "kelvin","kilogram","litre","lumen","lux","metre","mole",
                      "newton","ohm","pascal","radian","second","siemens","sievert",
                      "steradian","tesla","volt","watt","weber"]
#CellML 2.0 headers and footers
math_header = '<math xmlns="http://www.w3.org/1998/Math/MathML" xmlns:cellml="http://www.cellml.org/cellml/2.0#">\n'
math_footer = '</math>'


def createCellMLUnit(bunits,unitname):
    '''
    Create a cellml physical unit description with `unitname` and user description as `bunits`
    '''
    baseunits = json.loads(getBaseSIUnits(bunits))
    cunit = Units(unitname)
    mult = baseunits['mult']
    lunits = ['m','kg','s','A','K','mol','cd']
    siunits= ['metre','kilogram','second','ampere','kelvin','mole','candela']
    for six,lx in enumerate(lunits):
        if baseunits[lx]!=0:
            if baseunits[lx]>1:
                cunit.addUnit(siunits[six],"",baseunits[lx],mult)
                mult = 1
            else:
                cunit.addUnit(siunits[six],"",baseunits[lx],mult)
    return cunit

def getUnitName(unitname):
    '''
    Get a unitname that does not contain white spaces
    '''
    re.sub(r'\W+', '', unitname)
    return unitname


#CellML specific printer to convert sympy expressions to a form of mathml supported by cellml
from sympy.printing.mathml import MathMLContentPrinter

class Sympy2CellMLContentPrinter(MathMLContentPrinter):
    def _print_Symbol(self, sym):
        ci = self.dom.createElement(self.mathml_tag(sym))
        ci.appendChild(self.dom.createTextNode(sym.name))
        return ci 

#CellML Errors
CellMLlevelDescription = {
    Issue.Level.ERROR: "Error",
    Issue.Level.WARNING: "Warning",
    Issue.Level.MESSAGE: "Message"
}

def serialiseToCellML(composer,modelName='FTUModel',timeunit='second'):
    '''
    Logic to translate composer instance to CellML
    Model variables are non-dimensionalised to reduce the 
    need to determining dimensions of intermediate variables
    that arise from composition. 
    This is done in a transparent manner where the names model
    state/parameter variables have physical units, intermediate variables
    that have the non-dimenionalised values are used in expressions
    '''
    #Get all the variables (states, parameters, inputs/outputs)
    #to determine their units and create their non-dimensionalised surrogates
    variables = dict()
    variablesvalue = dict()
    allunits = []
    for k,v in composer.statevalues.items():
        variables[k] = k
        if v['units'] != 'dimensionless':
            variables[k] = sympy.symbols(f"{k}_nd")
            variablesvalue[k] = v
            allunits.append(v['units'])
        else:
            variables[k] = k
            variablesvalue[k] = v

    #Handle inputs
    for inp in composer.uVecSymbols:
        if inp in composer.inputs:
            variables[inp] = inp
            v = composer.inputs[inp]
            if v['units'] != 'dimensionless':
                variables[inp] = sympy.symbols(f"{inp}_nd")
                variablesvalue[inp] = v
                allunits.append(v['units'])
            else:
                variables[inp] = inp
                variablesvalue[inp] = v            
    #CellML does not provide heavside implementation
    #Explicitly implement     
    h1 = 1
    heavisideterms = dict()
    nonlinearrhsterms = dict()
    for c, t in composer.compositeparameters.items():
        fs = t["value"].free_symbols
        if len(fs) > 0: #This is a nonlinearterm              
            trms = t["value"].free_symbols
            for tr in trms:
                if tr in variables:
                    continue

                if tr in composer.compositeparameters:
                    k = tr
                    v = composer.compositeparameters[tr]
                    if v['units'] != 'dimensionless':
                        variables[k] = sympy.symbols(f"{k}_nd")
                        variablesvalue[k] = v  
                        allunits.append(v['units'])
                    else:
                        variables[k] = k
                        variablesvalue[k] = v  
                else:
                    k = tr
                    v = composer.statevalues[tr]
                    if v['units'] != 'dimensionless':
                        variables[k] = sympy.symbols(f"{k}_nd")
                        variablesvalue[k] = v  
                    else:
                        variables[k] = k
                        variablesvalue[k] = v  
            #Handle heaviside functions                
            expr = t["value"]
            for fr in t["value"].atoms(sympy.Function):
                if f"{fr}".startswith("Heaviside"):
                    heavisideterms[f"Hv_{h1}"] = fr
                    expr = expr.xreplace({fr:sympy.symbols(f"Hv_{h1}")})
                    h1 += 1                
            nonlinearrhsterms[c] = expr

    nonlinearrhstermsnd = dict() #Surrogates for dimensionalised variables
    heavisidetermsnd = dict()
    for k,v in heavisideterms.items():
        heavisidetermsnd[k] = v.xreplace(variables)

    for k,v in nonlinearrhsterms.items():
        nonlinearrhstermsnd[k] = v.xreplace(variables)

    stateVec = sympy.Matrix(composer.stateVec)
    Ccap = composer.uyConnectionMatrix
    Delx = composer.Qcap * stateVec  # Potential
    # Since E^-1 can be expensive, we will scale by the rate diagonal value of E for that component
    Einv = sympy.eye(composer.Ecap.shape[0])
    for i in range(composer.Ecap.shape[0]):
        Einv[i, i] = 1 / composer.Ecap[i, i]
    JRQx = (composer.Jcap - composer.Rcap) * Delx
    interioru = composer.Bcap * Ccap * (composer.Bcap.T) * stateVec #ucapVec
    exterioru = composer.Bdas * sympy.Matrix(composer.uVecSymbols).T
    rhs = sympy.SparseMatrix(Einv * (JRQx - interioru + exterioru)).subs(variables)
    cellmlprinter = Sympy2CellMLContentPrinter() #For expression translations
    #CellML model instance
    model = Model()
    model.setName(modelName)
    ftu = Component('FTU')
    model.addComponent(ftu)
    
    #Create the model, variables and functions
    externalinputs = []
    eqns = []
    #Setup time
    ftu.addVariable(Variable('t')) #time
    ftu.variable('t').setUnits(getUnitName(timeunit))
    #Variables and suggrogates
    for v,vnd in variables.items():
        if v in composer.inputs:
            externalinputs.append(v)
            continue
        var = f"{v}"
        varib = Variable(var)
        varib.setUnits(variablesvalue[v]['units'])
        varib.setInitialValue(f"{variablesvalue[v]['value']}")
        ftu.addVariable(varib)
        if v != vnd:
            varibnd = Variable(f"{vnd}")
            varibnd.setUnits('dimensionless')
            ftu.addVariable(varibnd)
            mape = f'<apply><eq/><ci>{vnd}</ci><apply><divide/><ci>{v}</ci><cn cellml:units="{variablesvalue[v]["units"]}">1</cn></apply></apply>'
            eqns.append(mape)

    #Create input variables as well as components
    #Each input variable is assigned a component,
    #Users can extend the logic in the component to manifest the dynamics
    for eiv in externalinputs:
        v = composer.inputs[eiv]
        compx = Component(f'External_{eiv}')

        vv = Variable(f"{eiv}")
        vv.setUnits(v['units'])
        vv.setInitialValue(0.0)
        vv.setInterfaceType('public_and_private')
        compx.addVariable(vv)
        #Add in main component
        fvv = Variable(f"{eiv}")
        fvv.setUnits('dimensionless')
        fvv.setInterfaceType('public_and_private')
        ftu.addVariable(fvv)  
            
        if v['units'] != 'dimensionless':
            vvd = Variable(f"{eiv}_nd")
            vvd.setUnits('dimensionless')
            compx.addVariable(vvd)        
            mape = f'<apply><eq/><ci>{eiv}_nd</ci><apply><divide/><ci>{eiv}</ci><cn cellml:units="{v["units"]}">1</cn></apply></apply>'
            compx.appendMath(mape)
            model.addComponent(compx)
            Variable.addEquivalence(vvd,fvv)    
        else:
            model.addComponent(compx)
            Variable.addEquivalence(vv,fvv)    
        
    #Heaviside functions
    for hk,hv in heavisidetermsnd.items():      
        varhv = Variable(f"{hk}")
        varhv.setUnits('dimensionless')
        ftu.addVariable(varhv)      
        meq = f'<apply><eq/><ci>{hk}</ci><piecewise><piece><cn cellml:units="dimensionless">1.0</cn><apply><gt/>{cellmlprinter.doprint(hv.args[0])}<cn cellml:units="dimensionless">0.0</cn></apply></piece><otherwise><cn cellml:units="dimensionless">0.0</cn></otherwise></piecewise></apply>'
        eqns.append(meq)

    #Expressions to evaluate non linear terms
    for hk,hv in nonlinearrhstermsnd.items():      
        varhv = Variable(f"{hk}")
        varhv.setUnits('dimensionless')
        ftu.addVariable(varhv)      
        meq = f'<apply><eq/><ci>{hk}</ci>{cellmlprinter.doprint(hv)}</apply>'
        eqns.append(meq)

    #ode for each state variable
    for si in range(rhs.shape[0]):
        state = stateVec[si]
        rhst = rhs[si]
        statedim = composer.statevalues[state]['units']
        if statedim != 'dimensionless':
            odeq = f'''<apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>{state}</ci></apply>
                <apply>
                <times/>
                {cellmlprinter.doprint(rhst)}
                <apply>
                <divide/>
                <cn cellml:units="{getUnitName(statedim)}">1.0</cn>
                <cn cellml:units="{getUnitName(timeunit)}">1.0</cn>
                </apply>
                </apply>        
                </apply>'''
        else:
            odeq = f'''<apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>{state}</ci></apply>
                <apply>
                <divide/>
                {cellmlprinter.doprint(rhst)}
                <cn cellml:units="{getUnitName(timeunit)}">1.0</cn>
                </apply>        
                </apply>'''        
        eqns.append(odeq)        

    #Create the units after creating the variables - avoids the units are unlinked issue
    if timeunit not in defaultcellmlunits:
        timeunitDef = createCellMLUnit(timeunit,getUnitName(timeunit))
        model.addUnits(timeunitDef)
    else:
        timeunitDef = Units(timeunit)

    unitdefs = dict()
    unitdefs['dimensionless'] = Units('dimensionless')
    for ut in set(allunits):
        unitdefs[ut] = createCellMLUnit(ut,getUnitName(ut)) 
        if ut not in defaultcellmlunits:
            model.addUnits(unitdefs[ut])

    ftu.setMath(math_header)
    matheq = '\n'.join(eqns).replace('<cn>','<cn cellml:units="dimensionless">')
    ftu.appendMath(matheq)    
    ftu.appendMath(math_footer)

    model.fixVariableInterfaces()
    
    validator = Validator()
    analyser = None
    validator.validateModel(model)
    if validator.issueCount()==0:
        analyser = Analyser()
        analyser.analyseModel(model)
        if analyser.issueCount()==0:
            printer = Printer()
            return printer.printModel(model)
    if analyser is not None:
        analyser = validator
    number_of_issues = analyser.issueCount()
    erm = [f"\nThe {type(analyser).__name__} has found {number_of_issues} issues:"]
    for e in range(0, number_of_issues):
        # Retrieve the issue item.
        i = analyser.issue(e)
        # Within the issue are a level, a URL, a reference heading (if appropriate), and
        # the item to which the issue applies.
        level = i.level()
        erm.append(f"{CellMLlevelDescription[level]}[{e}]\tDesc : {i.description()} ")
    raise Exception('\n'.join(erm))

    