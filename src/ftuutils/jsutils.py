import js
import json

#Done this way as pyodide has issues with from compositionutils import FTUGraph
try:
    from ftuutils.base import FTUGraph
except:
    from base import FTUGraph


def loadResolutions(result):
    amb = js.document.getElementById("loadAPIModel")
    amb.style.display = 'block'
    amb.dataset.init = json.dumps(result)
    
def loadComposedLatex(self,report,targetElement):
    """
    Load the composed composite PHS on to html
    """
    try:
        amb = js.document.getElementById(targetElement)
        amb.style.display = 'block'
        amb.innerHTML = report                   
        js.typesetCompositePHS(targetElement)
        js.composedLatexLoaded()
    except Exception as ex:
        import traceback
        js.console.log(f"Failed to typeset composite PHS {ex}\n{traceback.print_exc()}")

def loadComposedPythonCode(self,code,targetElement):
    """
    Load the composed composite PHS on to html
    """
    try:
        amb = js.document.getElementById(targetElement)
        amb.style.display = 'block'
        amb.innerHTML = code                    
        js.composedPythonLoaded()
    except Exception as ex:
        js.console.log(f"Failed to load Pythonic code of PHS {ex}")
        
def composeCompositePHSFromHTMLElement(self,htmlelementid):
    """Generates the composite PHS from a json serialization of graph, phsdata and connection information
        Helper function to support calls from javascript or other non-python languages
    Args:
        comp (string): PHS composition description in json format
    """      
    try:
        jstring = js.document.getElementById(htmlelementid).value
        composition = json.loads(jstring)
        fgraph = FTUGraph()
        composer,composition = fgraph.composeCompositePHSFromGraphicalObject(composition)
            
        self.loadComposedLatex(composer,"compositemodelphs")
        self.loadComposedPythonCode(composer,"composedPHSPython")
        #Store the structure in composedPHS div
        js.document.getElementById("composedPHS").innerHTML = json.dumps(composition)

    except Exception as inst:
        import traceback
        js.console.log('Failed to find html element with `id`=`compositemodelphs` to load composition results')
        js.console.log(f'{traceback.print_exc()}')
        js.alert(f"Failed to complete composition\n{inst}")
