
# The content of this file was generated using the FTUWeaver

import numpy as np
from numpy import exp
from scipy.integrate import ode

def Heaviside(x):
    if x > 0:
        return 1.0
    return 0.0

__version__ = "0.0.1"

class FTUStepperHooks():
    #PHS Input name and variable entry
    inputhooks={
        'i_1':'variables[0]',
        }
    #Node name and state array entries for each associated state variable
    statehooks={
        '1':{"Tai": "states[0]", "Ta": "states[1]", "u": "states[2]", "v": "states[3]"},
        '2':{"Tai": "states[4]", "Ta": "states[5]", "u": "states[6]", "v": "states[7]"},
        '3':{"Tai": "states[8]", "Ta": "states[9]", "u": "states[10]", "v": "states[11]"},
        '4':{"Tai": "states[12]", "Ta": "states[13]", "u": "states[14]", "v": "states[15]"},
        '5':{"Tai": "states[16]", "Ta": "states[17]", "u": "states[18]", "v": "states[19]"},
        '6':{"Tai": "states[20]", "Ta": "states[21]", "u": "states[22]", "v": "states[23]"},
        '7':{"Tai": "states[24]", "Ta": "states[25]", "u": "states[26]", "v": "states[27]"},
        '8':{"Tai": "states[28]", "Ta": "states[29]", "u": "states[30]", "v": "states[31]"},
        '9':{"Tai": "states[32]", "Ta": "states[33]", "u": "states[34]", "v": "states[35]"},
        '10':{"Tai": "states[36]", "Ta": "states[37]", "u": "states[38]", "v": "states[39]"},
        '11':{"Tai": "states[40]", "Ta": "states[41]", "u": "states[42]", "v": "states[43]"},
        '12':{"Tai": "states[44]", "Ta": "states[45]", "u": "states[46]", "v": "states[47]"},
        '13':{"Tai": "states[48]", "Ta": "states[49]", "u": "states[50]", "v": "states[51]"},
        '14':{"Tai": "states[52]", "Ta": "states[53]", "u": "states[54]", "v": "states[55]"},
        '15':{"Tai": "states[56]", "Ta": "states[57]", "u": "states[58]", "v": "states[59]"},
        '16':{"Tai": "states[60]", "Ta": "states[61]", "u": "states[62]", "v": "states[63]"},
        '17':{"Tai": "states[64]", "Ta": "states[65]", "u": "states[66]", "v": "states[67]"},
        '18':{"Tai": "states[68]", "Ta": "states[69]", "u": "states[70]", "v": "states[71]"},
        '19':{"Tai": "states[72]", "Ta": "states[73]", "u": "states[74]", "v": "states[75]"},
        '20':{"Tai": "states[76]", "Ta": "states[77]", "u": "states[78]", "v": "states[79]"},
        '21':{"Tai": "states[80]", "Ta": "states[81]", "u": "states[82]", "v": "states[83]"},
        }
    #State name and solution vector indexes for each associated PHS state variable
    statenamehooks={
        'Tai':[0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80],
        'Ta':[1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69,73,77,81],
        'u':[2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74,78,82],
        'v':[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79,83],
    }
    #PHS name and hamiltonian solution vector indexes 
    phsnamehooks={
        'APN':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        }

class FTUStepperInputs():
    #Input Node mappings 
    nodes=[
            {
                'nodelabel':4,
                'hamiltonianIndex':3,
                'hamiltonian':'-0.154143227709786*Ta_4*exp(-270.806766140199*(u_4 - 1)**2) - 0.008251*Tai_4**3*Heaviside(0.2925 - Tai_4)*Heaviside(u_4 - 0.78) + 0.008301*Tai_4**3 - 1.0*i_1_2*v_4 - 0.00533333333333333*u_4**3 + 0.00904*u_4**2 - 0.0034875*u_u_1*v_4 - 0.00225306424965365*u_u_15*v_4 - 0.00432646210991617*u_u_19*v_4 + 0.0100670263595698*u_u_4*v_4',
                'states':{"Tai_4": {"value": "0.00000", "order": 0}, "Ta_4": {"value": "0.00100", "order": 1}, "u_4": {"value": "0.00000", "order": 2}, "v_4": {"value": "0.03604", "order": 3}},
                'inputs':['i_1_2'],
                'rhs':[
                    '-0.016602*Tai_4 + 0.154143227709786*u_4*exp(-270.806766140199*(u_4 - 1)**2)',
                    'Ta_4*(0.016502*Heaviside(0.2925 - Tai_4)*Heaviside(u_4 - 0.78) - 0.016602) + Tai_4**2*(-0.016502*Heaviside(0.2925 - Tai_4)*Heaviside(u_4 - 0.78) + 0.016602)',
                    '1.0*i_1_2 - u_4**2*(0.0775*v_4 + 0.62*(u_4 - 1)*(u_4 - 0.13)) + 0.0034875*u_u_1 + 0.00225306424965365*u_u_15 + 0.00432646210991617*u_u_19 - 0.0100670263595698*u_u_4',
                    '-0.0775*v_4*(8.0*u_4*(u_4 - 1.13) + v_4)*(0.002*u_4 + 0.2*v_4 + 0.0006)/(u_4 + 0.3)',
                ],
                'statevarmap':{
                        'u_u_1':2,
                        'u_u_19':74,
                        'u_u_4':14,
                        'u_u_15':58,
                },
                'inputExpressions':{
                        'u_4':{'statevecindex':14,'expr':'0.0034875*u_u_1 + 0.00225306424965365*u_u_15 + 0.00432646210991617*u_u_19 - 0.0100670263595698*u_u_4'},
                }
            },
            {
                'nodelabel':5,
                'hamiltonianIndex':4,
                'hamiltonian':'(-0.308286455419572*Ta_5 + (Tai_5**3*(-0.016502*Heaviside(0.2925 - Tai_5)*Heaviside(u_5 - 0.78) + 0.016602) - 0.0106666666666667*u_5**3 + 0.01808*u_5**2 - 2*v_5*(1.0*i_1_2 + 0.00181486232217075*u_u_11 + 0.0019375*u_u_12 + 0.0033558484396647*u_u_18 - 0.00908557896398634*u_u_5 + 0.00197736820215088*u_u_8))*exp(270.806766140199*(u_5 - 1)**2))*exp(-270.806766140199*(u_5 - 1)**2)/2',
                'states':{"Tai_5": {"value": "0.00000", "order": 0}, "Ta_5": {"value": "0.00100", "order": 1}, "u_5": {"value": "0.00000", "order": 2}, "v_5": {"value": "0.03604", "order": 3}},
                'inputs':['i_1_2'],
                'rhs':[
                    '-0.016602*Tai_5 + 0.154143227709786*u_5*exp(-270.806766140199*(u_5 - 1)**2)',
                    'Ta_5*(0.016502*Heaviside(0.2925 - Tai_5)*Heaviside(u_5 - 0.78) - 0.016602) + Tai_5**2*(-0.016502*Heaviside(0.2925 - Tai_5)*Heaviside(u_5 - 0.78) + 0.016602)',
                    '1.0*i_1_2 - u_5**2*(0.0775*v_5 + 0.62*(u_5 - 1)*(u_5 - 0.13)) + 0.00181486232217075*u_u_11 + 0.0019375*u_u_12 + 0.0033558484396647*u_u_18 - 0.00908557896398634*u_u_5 + 0.00197736820215088*u_u_8',
                    '-0.0775*v_5*(8.0*u_5*(u_5 - 1.13) + v_5)*(0.002*u_5 + 0.2*v_5 + 0.0006)/(u_5 + 0.3)',
                ],
                'statevarmap':{
                        'u_u_18':70,
                        'u_u_5':18,
                        'u_u_12':46,
                        'u_u_8':30,
                        'u_u_11':42,
                },
                'inputExpressions':{
                        'u_5':{'statevecindex':18,'expr':'0.00181486232217075*u_u_11 + 0.0019375*u_u_12 + 0.0033558484396647*u_u_18 - 0.00908557896398634*u_u_5 + 0.00197736820215088*u_u_8'},
                }
            },
            {
                'nodelabel':11,
                'hamiltonianIndex':10,
                'hamiltonian':'(-0.308286455419572*Ta_11 + (Tai_11**3*(-0.016502*Heaviside(0.2925 - Tai_11)*Heaviside(u_11 - 0.78) + 0.016602) - 0.0106666666666667*u_11**3 + 0.01808*u_11**2 - 2*v_11*(1.0*i_1_2 - 0.0209989121379338*u_u_11 + 0.00462913406379882*u_u_12 + 0.000266912464258687*u_u_13 + 0.0019375*u_u_15 + 0.00129147836225876*u_u_17 + 5.70067978491158e-5*u_u_19 + 0.00181486232217075*u_u_5 + 0.0110020181275976*u_u_9))*exp(270.806766140199*(u_11 - 1)**2))*exp(-270.806766140199*(u_11 - 1)**2)/2',
                'states':{"Tai_11": {"value": "0.00000", "order": 0}, "Ta_11": {"value": "0.00100", "order": 1}, "u_11": {"value": "0.00000", "order": 2}, "v_11": {"value": "0.03604", "order": 3}},
                'inputs':['i_1_2'],
                'rhs':[
                    '-0.016602*Tai_11 + 0.154143227709786*u_11*exp(-270.806766140199*(u_11 - 1)**2)',
                    'Ta_11*(0.016502*Heaviside(0.2925 - Tai_11)*Heaviside(u_11 - 0.78) - 0.016602) + Tai_1(-0.016502*Heaviside(0.2925 - Tai_11)*Heaviside(u_11 - 0.78) + 0.016602)',
                    '1.0*i_1_2 - u_1(0.0775*v_11 + 0.62*(u_11 - 1)*(u_11 - 0.13)) - 0.0209989121379338*u_u_11 + 0.00462913406379882*u_u_12 + 0.000266912464258687*u_u_13 + 0.0019375*u_u_15 + 0.00129147836225876*u_u_17 + 5.70067978491158e-5*u_u_19 + 0.00181486232217075*u_u_5 + 0.0110020181275976*u_u_9',
                    '-0.0775*v_11*(8.0*u_11*(u_11 - 1.13) + v_11)*(0.002*u_11 + 0.2*v_11 + 0.0006)/(u_11 + 0.3)',
                ],
                'statevarmap':{
                        'u_u_13':50,
                        'u_u_17':66,
                        'u_u_9':34,
                        'u_u_15':58,
                        'u_u_5':18,
                        'u_u_12':46,
                        'u_u_19':74,
                        'u_u_11':42,
                },
                'inputExpressions':{
                        'u_11':{'statevecindex':42,'expr':'-0.0209989121379338*u_u_11 + 0.00462913406379882*u_u_12 + 0.000266912464258687*u_u_13 + 0.0019375*u_u_15 + 0.00129147836225876*u_u_17 + 5.70067978491158e-5*u_u_19 + 0.00181486232217075*u_u_5 + 0.0110020181275976*u_u_9'},
                }
            },
            {
                'nodelabel':17,
                'hamiltonianIndex':16,
                'hamiltonian':'(-0.308286455419572*Ta_17 + (Tai_17**3*(-0.016502*Heaviside(0.2925 - Tai_17)*Heaviside(u_17 - 0.78) + 0.016602) - 0.0106666666666667*u_17**3 + 0.01808*u_17**2 - 2*v_17*(1.0*i_1_2 + 0.00129147836225876*u_u_11 + 0.00812293911287484*u_u_14 + 0.00777166709324443*u_u_16 - 0.0303833148058918*u_u_17 + 0.00624309617371499*u_u_2 + 0.00346663406379882*u_u_20 + 0.0034875*u_u_9))*exp(270.806766140199*(u_17 - 1)**2))*exp(-270.806766140199*(u_17 - 1)**2)/2',
                'states':{"Tai_17": {"value": "0.00000", "order": 0}, "Ta_17": {"value": "0.00100", "order": 1}, "u_17": {"value": "0.00000", "order": 2}, "v_17": {"value": "0.03604", "order": 3}},
                'inputs':['i_1_2'],
                'rhs':[
                    '-0.016602*Tai_17 + 0.154143227709786*u_17*exp(-270.806766140199*(u_17 - 1)**2)',
                    'Ta_17*(0.016502*Heaviside(0.2925 - Tai_17)*Heaviside(u_17 - 0.78) - 0.016602) + Tai_17**2*(-0.016502*Heaviside(0.2925 - Tai_17)*Heaviside(u_17 - 0.78) + 0.016602)',
                    '1.0*i_1_2 - u_17**2*(0.0775*v_17 + 0.62*(u_17 - 1)*(u_17 - 0.13)) + 0.00129147836225876*u_u_11 + 0.00812293911287484*u_u_14 + 0.00777166709324443*u_u_16 - 0.0303833148058918*u_u_17 + 0.00624309617371499*u_u_2 + 0.00346663406379882*u_u_20 + 0.0034875*u_u_9',
                    '-0.0775*v_17*(8.0*u_17*(u_17 - 1.13) + v_17)*(0.002*u_17 + 0.2*v_17 + 0.0006)/(u_17 + 0.3)',
                ],
                'statevarmap':{
                        'u_u_9':34,
                        'u_u_17':66,
                        'u_u_20':78,
                        'u_u_2':6,
                        'u_u_14':54,
                        'u_u_11':42,
                        'u_u_16':62,
                },
                'inputExpressions':{
                        'u_17':{'statevecindex':66,'expr':'0.00129147836225876*u_u_11 + 0.00812293911287484*u_u_14 + 0.00777166709324443*u_u_16 - 0.0303833148058918*u_u_17 + 0.00624309617371499*u_u_2 + 0.00346663406379882*u_u_20 + 0.0034875*u_u_9'},
                }
            },
        ]

class FTUStepper():
    STATE_COUNT = 84
    VARIABLE_COUNT = 330
    CELL_COUNT  = 21

    stateIndexes = {"Tai": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80], "Ta": [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81], "u": [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82], "v": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83]}

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
        states[0] = 0.000000  #Tai_1
        states[1] = 0.001000  #Ta_1
        states[2] = 0.000000  #u_1
        states[3] = 0.036040  #v_1
        states[4] = 0.000000  #Tai_2
        states[5] = 0.001000  #Ta_2
        states[6] = 0.000000  #u_2
        states[7] = 0.036040  #v_2
        states[8] = 0.000000  #Tai_3
        states[9] = 0.001000  #Ta_3
        states[10] = 0.000000  #u_3
        states[11] = 0.036040  #v_3
        states[12] = 0.000000  #Tai_4
        states[13] = 0.001000  #Ta_4
        states[14] = 0.000000  #u_4
        states[15] = 0.036040  #v_4
        states[16] = 0.000000  #Tai_5
        states[17] = 0.001000  #Ta_5
        states[18] = 0.000000  #u_5
        states[19] = 0.036040  #v_5
        states[20] = 0.000000  #Tai_6
        states[21] = 0.001000  #Ta_6
        states[22] = 0.000000  #u_6
        states[23] = 0.036040  #v_6
        states[24] = 0.000000  #Tai_7
        states[25] = 0.001000  #Ta_7
        states[26] = 0.000000  #u_7
        states[27] = 0.036040  #v_7
        states[28] = 0.000000  #Tai_8
        states[29] = 0.001000  #Ta_8
        states[30] = 0.000000  #u_8
        states[31] = 0.036040  #v_8
        states[32] = 0.000000  #Tai_9
        states[33] = 0.001000  #Ta_9
        states[34] = 0.000000  #u_9
        states[35] = 0.036040  #v_9
        states[36] = 0.000000  #Tai_10
        states[37] = 0.001000  #Ta_10
        states[38] = 0.000000  #u_10
        states[39] = 0.036040  #v_10
        states[40] = 0.000000  #Tai_11
        states[41] = 0.001000  #Ta_11
        states[42] = 0.000000  #u_11
        states[43] = 0.036040  #v_11
        states[44] = 0.000000  #Tai_12
        states[45] = 0.001000  #Ta_12
        states[46] = 0.000000  #u_12
        states[47] = 0.036040  #v_12
        states[48] = 0.000000  #Tai_13
        states[49] = 0.001000  #Ta_13
        states[50] = 0.000000  #u_13
        states[51] = 0.036040  #v_13
        states[52] = 0.000000  #Tai_14
        states[53] = 0.001000  #Ta_14
        states[54] = 0.000000  #u_14
        states[55] = 0.036040  #v_14
        states[56] = 0.000000  #Tai_15
        states[57] = 0.001000  #Ta_15
        states[58] = 0.000000  #u_15
        states[59] = 0.036040  #v_15
        states[60] = 0.000000  #Tai_16
        states[61] = 0.001000  #Ta_16
        states[62] = 0.000000  #u_16
        states[63] = 0.036040  #v_16
        states[64] = 0.000000  #Tai_17
        states[65] = 0.001000  #Ta_17
        states[66] = 0.000000  #u_17
        states[67] = 0.036040  #v_17
        states[68] = 0.000000  #Tai_18
        states[69] = 0.001000  #Ta_18
        states[70] = 0.000000  #u_18
        states[71] = 0.036040  #v_18
        states[72] = 0.000000  #Tai_19
        states[73] = 0.001000  #Ta_19
        states[74] = 0.000000  #u_19
        states[75] = 0.036040  #v_19
        states[76] = 0.000000  #Tai_20
        states[77] = 0.001000  #Ta_20
        states[78] = 0.000000  #u_20
        states[79] = 0.036040  #v_20
        states[80] = 0.000000  #Tai_21
        states[81] = 0.001000  #Ta_21
        states[82] = 0.000000  #u_21
        states[83] = 0.036040  #v_21
        variables[1] = 0.011900  #c_1
        variables[2] = 0.195871  #c_2
        variables[3] = 0.022188  #c_3
        variables[4] = 0.017321  #c_4
        variables[5] = 0.006041  #c_5
        variables[6] = 0.029072  #c_6
        variables[7] = 0.003356  #c_7
        variables[8] = 0.012990  #c_8
        variables[9] = 0.021454  #c_9
        variables[10] = 0.001342  #c_10
        variables[11] = 0.013687  #c_11
        variables[12] = 0.003444  #c_12
        variables[13] = 0.166178  #c_13
        variables[14] = 0.001977  #c_14
        variables[15] = 0.000057  #c_15
        variables[16] = 0.352216  #c_16
        variables[17] = 0.055825  #c_17
        variables[18] = 0.073612  #c_18
        variables[19] = 0.270954  #c_19
        variables[20] = 0.206773  #c_20
        variables[21] = 0.016664  #c_21
        variables[22] = 0.041942  #c_22
        variables[23] = 0.002331  #c_23
        variables[24] = 0.005705  #c_24
        variables[25] = 0.064486  #c_25
        variables[26] = 0.147224  #c_26
        variables[27] = 0.004629  #c_27
        variables[28] = 0.077500  #c_28
        variables[29] = 0.005074  #c_29
        variables[30] = 0.276829  #c_30
        variables[31] = 0.005532  #c_31
        variables[32] = 0.045000  #c_32
        variables[33] = 0.016602  #c_33
        variables[34] = 0.291731  #c_34
        variables[35] = 0.007772  #c_35
        variables[36] = 0.002333  #c_36
        variables[37] = 0.077811  #c_37
        variables[38] = 0.071386  #c_38
        variables[39] = 0.001777  #c_39
        variables[40] = 0.043301  #c_40
        variables[41] = 0.141962  #c_41
        variables[42] = 0.039124  #c_42
        variables[43] = 0.085243  #c_43
        variables[44] = 0.008123  #c_44
        variables[45] = 0.077942  #c_45
        variables[46] = 0.286301  #c_46
        variables[47] = 0.007887  #c_47
        variables[48] = 0.006843  #c_48
        variables[49] = 0.000393  #c_49
        variables[50] = 0.000267  #c_50
        variables[51] = 0.010535  #c_51
        variables[52] = 0.002649  #c_52
        variables[53] = 0.002253  #c_53
        variables[54] = 0.023094  #c_54
        variables[55] = 0.017267  #c_55
        variables[56] = 0.030311  #c_56
        variables[57] = 0.060000  #c_57
        variables[58] = 0.001815  #c_58
        variables[59] = 0.001790  #c_59
        variables[60] = 0.023660  #c_60
        variables[61] = 0.367619  #c_61
        variables[62] = 0.000305  #c_62
        variables[63] = 0.001007  #c_63
        variables[64] = 0.017421  #c_64
        variables[65] = 0.042727  #c_65
        variables[66] = 0.153546  #c_66
        variables[67] = 0.088301  #c_67
        variables[68] = 0.010067  #c_68
        variables[69] = 0.001938  #c_69
        variables[70] = 0.176603  #c_70
        variables[71] = 0.006606  #c_71
        variables[72] = 0.018755  #c_72
        variables[73] = 0.047563  #c_73
        variables[74] = 0.000024  #c_74
        variables[75] = 0.281371  #c_75
        variables[76] = 0.030383  #c_76
        variables[77] = 0.011410  #c_77
        variables[78] = 0.022925  #c_78
        variables[79] = 0.128321  #c_79
        variables[80] = 0.003311  #c_80
        variables[81] = 0.030108  #c_81
        variables[82] = 0.046405  #c_82
        variables[83] = 0.006030  #c_83
        variables[84] = 0.001834  #c_84
        variables[85] = 0.020999  #c_85
        variables[86] = 0.104812  #c_86
        variables[87] = 0.016025  #c_87
        variables[88] = 0.001350  #c_88
        variables[89] = 0.004650  #c_89
        variables[90] = 0.004326  #c_90
        variables[91] = 0.100280  #c_91
        variables[92] = 0.003250  #c_92
        variables[93] = 0.009086  #c_93
        variables[94] = 0.003467  #c_94
        variables[95] = 0.080556  #c_95
        variables[96] = 0.392043  #c_96
        variables[97] = 0.117233  #c_97
        variables[98] = 0.025514  #c_98
        variables[99] = 0.011002  #c_99
        variables[100] = 0.003032  #c_100
        variables[101] = 0.135933  #c_101
        variables[102] = 0.129897  #c_102
        variables[103] = 0.000736  #c_103
        variables[104] = 0.009945  #c_104
        variables[105] = 0.241997  #c_105
        variables[106] = 0.004998  #c_106
        variables[107] = 0.002349  #c_107
        variables[108] = 0.002695  #c_108
        variables[109] = 0.028490  #c_109
        variables[110] = 0.021806  #c_110
        variables[111] = 0.059731  #c_111
        variables[112] = 0.006243  #c_112
        variables[113] = 0.003487  #c_113
        variables[114] = 0.003596  #c_114
        variables[115] = 0.023418  #c_115
        variables[116] = 0.044731  #c_116
        variables[117] = 0.000611  #c_117
        variables[118] = 0.030080  #c_118
        variables[119] = 0.003686  #c_119
        variables[120] = 0.025000  #c_120
        variables[121] = 0.222801  #c_121
        variables[122] = 0.034175  #c_122
        variables[123] = 0.034770  #c_123
        variables[124] = 0.008054  #c_124
        variables[125] = 0.027297  #c_125
        variables[126] = 0.022609  #c_126
        variables[127] = 0.015180  #c_127
        variables[128] = 0.012879  #c_128
        variables[129] = 0.103923  #c_129
        variables[130] = 0.001291  #c_130
        variables[131] = 0.000100  #c_131
        variables[132] = 0.780000  #c_132
        variables[133] = 0.292500  #c_133
        variables[134] = 0.042969  #c_134
        variables[135] = 2.506575  #c_135
        variables[136] = 8.000000  #c_136
        variables[137] = 0.130000  #c_137
        variables[138] = 0.200000  #c_138
        variables[139] = 0.002000  #c_139
        variables[140] = 0.300000  #c_140

    def compute_variables(self,voi):
        t=voi #mapping to t
        states, rates, variables = self.states,self.rates,self.variables
        #\hat{u}_Tai_1 = Tai_1
        variables[141] = states[0]
        #\hat{u}_Ta_1 = Ta_1
        variables[142] = states[1]
        #\hat{u}_u_1 = u_1
        variables[143] = states[2]
        #\hat{u}_v_1 = v_1
        variables[144] = states[3]
        #\hat{u}_Tai_2 = Tai_2
        variables[145] = states[4]
        #\hat{u}_Ta_2 = Ta_2
        variables[146] = states[5]
        #\hat{u}_u_2 = u_2
        variables[147] = states[6]
        #\hat{u}_v_2 = v_2
        variables[148] = states[7]
        #\hat{u}_Tai_3 = Tai_3
        variables[149] = states[8]
        #\hat{u}_Ta_3 = Ta_3
        variables[150] = states[9]
        #\hat{u}_u_3 = u_3
        variables[151] = states[10]
        #\hat{u}_v_3 = v_3
        variables[152] = states[11]
        #\hat{u}_Tai_4 = Tai_4
        variables[153] = states[12]
        #\hat{u}_Ta_4 = Ta_4
        variables[154] = states[13]
        #\hat{u}_u_4 = u_4
        variables[155] = states[14]
        #\hat{u}_v_4 = v_4
        variables[156] = states[15]
        #\hat{u}_Tai_5 = Tai_5
        variables[157] = states[16]
        #\hat{u}_Ta_5 = Ta_5
        variables[158] = states[17]
        #\hat{u}_u_5 = u_5
        variables[159] = states[18]
        #\hat{u}_v_5 = v_5
        variables[160] = states[19]
        #\hat{u}_Tai_6 = Tai_6
        variables[161] = states[20]
        #\hat{u}_Ta_6 = Ta_6
        variables[162] = states[21]
        #\hat{u}_u_6 = u_6
        variables[163] = states[22]
        #\hat{u}_v_6 = v_6
        variables[164] = states[23]
        #\hat{u}_Tai_7 = Tai_7
        variables[165] = states[24]
        #\hat{u}_Ta_7 = Ta_7
        variables[166] = states[25]
        #\hat{u}_u_7 = u_7
        variables[167] = states[26]
        #\hat{u}_v_7 = v_7
        variables[168] = states[27]
        #\hat{u}_Tai_8 = Tai_8
        variables[169] = states[28]
        #\hat{u}_Ta_8 = Ta_8
        variables[170] = states[29]
        #\hat{u}_u_8 = u_8
        variables[171] = states[30]
        #\hat{u}_v_8 = v_8
        variables[172] = states[31]
        #\hat{u}_Tai_9 = Tai_9
        variables[173] = states[32]
        #\hat{u}_Ta_9 = Ta_9
        variables[174] = states[33]
        #\hat{u}_u_9 = u_9
        variables[175] = states[34]
        #\hat{u}_v_9 = v_9
        variables[176] = states[35]
        #\hat{u}_Tai_10 = Tai_10
        variables[177] = states[36]
        #\hat{u}_Ta_10 = Ta_10
        variables[178] = states[37]
        #\hat{u}_u_10 = u_10
        variables[179] = states[38]
        #\hat{u}_v_10 = v_10
        variables[180] = states[39]
        #\hat{u}_Tai_11 = Tai_11
        variables[181] = states[40]
        #\hat{u}_Ta_11 = Ta_11
        variables[182] = states[41]
        #\hat{u}_u_11 = u_11
        variables[183] = states[42]
        #\hat{u}_v_11 = v_11
        variables[184] = states[43]
        #\hat{u}_Tai_12 = Tai_12
        variables[185] = states[44]
        #\hat{u}_Ta_12 = Ta_12
        variables[186] = states[45]
        #\hat{u}_u_12 = u_12
        variables[187] = states[46]
        #\hat{u}_v_12 = v_12
        variables[188] = states[47]
        #\hat{u}_Tai_13 = Tai_13
        variables[189] = states[48]
        #\hat{u}_Ta_13 = Ta_13
        variables[190] = states[49]
        #\hat{u}_u_13 = u_13
        variables[191] = states[50]
        #\hat{u}_v_13 = v_13
        variables[192] = states[51]
        #\hat{u}_Tai_14 = Tai_14
        variables[193] = states[52]
        #\hat{u}_Ta_14 = Ta_14
        variables[194] = states[53]
        #\hat{u}_u_14 = u_14
        variables[195] = states[54]
        #\hat{u}_v_14 = v_14
        variables[196] = states[55]
        #\hat{u}_Tai_15 = Tai_15
        variables[197] = states[56]
        #\hat{u}_Ta_15 = Ta_15
        variables[198] = states[57]
        #\hat{u}_u_15 = u_15
        variables[199] = states[58]
        #\hat{u}_v_15 = v_15
        variables[200] = states[59]
        #\hat{u}_Tai_16 = Tai_16
        variables[201] = states[60]
        #\hat{u}_Ta_16 = Ta_16
        variables[202] = states[61]
        #\hat{u}_u_16 = u_16
        variables[203] = states[62]
        #\hat{u}_v_16 = v_16
        variables[204] = states[63]
        #\hat{u}_Tai_17 = Tai_17
        variables[205] = states[64]
        #\hat{u}_Ta_17 = Ta_17
        variables[206] = states[65]
        #\hat{u}_u_17 = u_17
        variables[207] = states[66]
        #\hat{u}_v_17 = v_17
        variables[208] = states[67]
        #\hat{u}_Tai_18 = Tai_18
        variables[209] = states[68]
        #\hat{u}_Ta_18 = Ta_18
        variables[210] = states[69]
        #\hat{u}_u_18 = u_18
        variables[211] = states[70]
        #\hat{u}_v_18 = v_18
        variables[212] = states[71]
        #\hat{u}_Tai_19 = Tai_19
        variables[213] = states[72]
        #\hat{u}_Ta_19 = Ta_19
        variables[214] = states[73]
        #\hat{u}_u_19 = u_19
        variables[215] = states[74]
        #\hat{u}_v_19 = v_19
        variables[216] = states[75]
        #\hat{u}_Tai_20 = Tai_20
        variables[217] = states[76]
        #\hat{u}_Ta_20 = Ta_20
        variables[218] = states[77]
        #\hat{u}_u_20 = u_20
        variables[219] = states[78]
        #\hat{u}_v_20 = v_20
        variables[220] = states[79]
        #\hat{u}_Tai_21 = Tai_21
        variables[221] = states[80]
        #\hat{u}_Ta_21 = Ta_21
        variables[222] = states[81]
        #\hat{u}_u_21 = u_21
        variables[223] = states[82]
        #\hat{u}_v_21 = v_21
        variables[224] = states[83]

    def compute_rates(self,voi):
        t=voi #mapping to t
        states, rates, variables = self.states,self.rates,self.variables
        #eta1_1 = Tai_1*(c_33 + (c_131 - c_33)*Heaviside(-Tai_1 + c_133)*Heaviside(u_1 - c_132))
        variables[225] = states[0]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[0] + variables[133])*Heaviside(states[2] - variables[132]))
        #eta2_1 = c_33 + (c_131 - c_33)*Heaviside(-Tai_1 + c_133)*Heaviside(u_1 - c_132)
        variables[226] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[0] + variables[133])*Heaviside(states[2] - variables[132])
        #kV_1 = exp(-0.5*(u_1 - 1)**2/c_134**2)/(c_134*c_135)
        variables[227] = exp(-0.5*(states[2] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_1 = -u_1*v_1 + u_1*c_136*(1 - u_1)*(u_1 - c_137)
        variables[228] = -states[2]*states[3] + states[2]*variables[136]*(1 - states[2])*(states[2] - variables[137])
        #V_1 = (-u_1*c_136*(u_1 - c_137 - 1) - v_1)*(v_1*c_138/(u_1 + c_140) + c_139)
        variables[229] = (-states[2]*variables[136]*(states[2] - variables[137] - 1) - states[3])*(states[3]*variables[138]/(states[2] + variables[140]) + variables[139])
        #eta1_2 = Tai_2*(c_33 + (c_131 - c_33)*Heaviside(-Tai_2 + c_133)*Heaviside(u_2 - c_132))
        variables[230] = states[4]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[4] + variables[133])*Heaviside(states[6] - variables[132]))
        #eta2_2 = c_33 + (c_131 - c_33)*Heaviside(-Tai_2 + c_133)*Heaviside(u_2 - c_132)
        variables[231] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[4] + variables[133])*Heaviside(states[6] - variables[132])
        #kV_2 = exp(-0.5*(u_2 - 1)**2/c_134**2)/(c_134*c_135)
        variables[232] = exp(-0.5*(states[6] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_2 = -u_2*v_2 + u_2*c_136*(1 - u_2)*(u_2 - c_137)
        variables[233] = -states[6]*states[7] + states[6]*variables[136]*(1 - states[6])*(states[6] - variables[137])
        #V_2 = (-u_2*c_136*(u_2 - c_137 - 1) - v_2)*(v_2*c_138/(u_2 + c_140) + c_139)
        variables[234] = (-states[6]*variables[136]*(states[6] - variables[137] - 1) - states[7])*(states[7]*variables[138]/(states[6] + variables[140]) + variables[139])
        #eta1_3 = Tai_3*(c_33 + (c_131 - c_33)*Heaviside(u_3 - c_132)*Heaviside(-Tai_3 + c_133))
        variables[235] = states[8]*(variables[33] + (variables[131] - variables[33])*Heaviside(states[10] - variables[132])*Heaviside(-states[8] + variables[133]))
        #eta2_3 = c_33 + (c_131 - c_33)*Heaviside(u_3 - c_132)*Heaviside(-Tai_3 + c_133)
        variables[236] = variables[33] + (variables[131] - variables[33])*Heaviside(states[10] - variables[132])*Heaviside(-states[8] + variables[133])
        #kV_3 = exp(-0.5*(u_3 - 1)**2/c_134**2)/(c_134*c_135)
        variables[237] = exp(-0.5*(states[10] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_3 = -u_3*v_3 + u_3*c_136*(1 - u_3)*(u_3 - c_137)
        variables[238] = -states[10]*states[11] + states[10]*variables[136]*(1 - states[10])*(states[10] - variables[137])
        #V_3 = (-u_3*c_136*(u_3 - c_137 - 1) - v_3)*(v_3*c_138/(u_3 + c_140) + c_139)
        variables[239] = (-states[10]*variables[136]*(states[10] - variables[137] - 1) - states[11])*(states[11]*variables[138]/(states[10] + variables[140]) + variables[139])
        #eta1_4 = Tai_4*(c_33 + (c_131 - c_33)*Heaviside(-Tai_4 + c_133)*Heaviside(u_4 - c_132))
        variables[240] = states[12]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[12] + variables[133])*Heaviside(states[14] - variables[132]))
        #eta2_4 = c_33 + (c_131 - c_33)*Heaviside(-Tai_4 + c_133)*Heaviside(u_4 - c_132)
        variables[241] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[12] + variables[133])*Heaviside(states[14] - variables[132])
        #kV_4 = exp(-0.5*(u_4 - 1)**2/c_134**2)/(c_134*c_135)
        variables[242] = exp(-0.5*(states[14] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_4 = -u_4*v_4 + u_4*c_136*(1 - u_4)*(u_4 - c_137)
        variables[243] = -states[14]*states[15] + states[14]*variables[136]*(1 - states[14])*(states[14] - variables[137])
        #V_4 = (-u_4*c_136*(u_4 - c_137 - 1) - v_4)*(v_4*c_138/(u_4 + c_140) + c_139)
        variables[244] = (-states[14]*variables[136]*(states[14] - variables[137] - 1) - states[15])*(states[15]*variables[138]/(states[14] + variables[140]) + variables[139])
        #eta1_5 = Tai_5*(c_33 + (c_131 - c_33)*Heaviside(-Tai_5 + c_133)*Heaviside(u_5 - c_132))
        variables[245] = states[16]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[16] + variables[133])*Heaviside(states[18] - variables[132]))
        #eta2_5 = c_33 + (c_131 - c_33)*Heaviside(-Tai_5 + c_133)*Heaviside(u_5 - c_132)
        variables[246] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[16] + variables[133])*Heaviside(states[18] - variables[132])
        #kV_5 = exp(-0.5*(u_5 - 1)**2/c_134**2)/(c_134*c_135)
        variables[247] = exp(-0.5*(states[18] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_5 = -u_5*v_5 + u_5*c_136*(1 - u_5)*(u_5 - c_137)
        variables[248] = -states[18]*states[19] + states[18]*variables[136]*(1 - states[18])*(states[18] - variables[137])
        #V_5 = (-u_5*c_136*(u_5 - c_137 - 1) - v_5)*(v_5*c_138/(u_5 + c_140) + c_139)
        variables[249] = (-states[18]*variables[136]*(states[18] - variables[137] - 1) - states[19])*(states[19]*variables[138]/(states[18] + variables[140]) + variables[139])
        #eta1_6 = Tai_6*(c_33 + (c_131 - c_33)*Heaviside(-Tai_6 + c_133)*Heaviside(u_6 - c_132))
        variables[250] = states[20]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[20] + variables[133])*Heaviside(states[22] - variables[132]))
        #eta2_6 = c_33 + (c_131 - c_33)*Heaviside(-Tai_6 + c_133)*Heaviside(u_6 - c_132)
        variables[251] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[20] + variables[133])*Heaviside(states[22] - variables[132])
        #kV_6 = exp(-0.5*(u_6 - 1)**2/c_134**2)/(c_134*c_135)
        variables[252] = exp(-0.5*(states[22] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_6 = -u_6*v_6 + u_6*c_136*(1 - u_6)*(u_6 - c_137)
        variables[253] = -states[22]*states[23] + states[22]*variables[136]*(1 - states[22])*(states[22] - variables[137])
        #V_6 = (-u_6*c_136*(u_6 - c_137 - 1) - v_6)*(v_6*c_138/(u_6 + c_140) + c_139)
        variables[254] = (-states[22]*variables[136]*(states[22] - variables[137] - 1) - states[23])*(states[23]*variables[138]/(states[22] + variables[140]) + variables[139])
        #eta1_7 = Tai_7*(c_33 + (c_131 - c_33)*Heaviside(-Tai_7 + c_133)*Heaviside(u_7 - c_132))
        variables[255] = states[24]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[24] + variables[133])*Heaviside(states[26] - variables[132]))
        #eta2_7 = c_33 + (c_131 - c_33)*Heaviside(-Tai_7 + c_133)*Heaviside(u_7 - c_132)
        variables[256] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[24] + variables[133])*Heaviside(states[26] - variables[132])
        #kV_7 = exp(-0.5*(u_7 - 1)**2/c_134**2)/(c_134*c_135)
        variables[257] = exp(-0.5*(states[26] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_7 = -u_7*v_7 + u_7*c_136*(1 - u_7)*(u_7 - c_137)
        variables[258] = -states[26]*states[27] + states[26]*variables[136]*(1 - states[26])*(states[26] - variables[137])
        #V_7 = (-u_7*c_136*(u_7 - c_137 - 1) - v_7)*(v_7*c_138/(u_7 + c_140) + c_139)
        variables[259] = (-states[26]*variables[136]*(states[26] - variables[137] - 1) - states[27])*(states[27]*variables[138]/(states[26] + variables[140]) + variables[139])
        #eta1_8 = Tai_8*(c_33 + (c_131 - c_33)*Heaviside(-Tai_8 + c_133)*Heaviside(u_8 - c_132))
        variables[260] = states[28]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[28] + variables[133])*Heaviside(states[30] - variables[132]))
        #eta2_8 = c_33 + (c_131 - c_33)*Heaviside(-Tai_8 + c_133)*Heaviside(u_8 - c_132)
        variables[261] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[28] + variables[133])*Heaviside(states[30] - variables[132])
        #kV_8 = exp(-0.5*(u_8 - 1)**2/c_134**2)/(c_134*c_135)
        variables[262] = exp(-0.5*(states[30] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_8 = -u_8*v_8 + u_8*c_136*(1 - u_8)*(u_8 - c_137)
        variables[263] = -states[30]*states[31] + states[30]*variables[136]*(1 - states[30])*(states[30] - variables[137])
        #V_8 = (-u_8*c_136*(u_8 - c_137 - 1) - v_8)*(v_8*c_138/(u_8 + c_140) + c_139)
        variables[264] = (-states[30]*variables[136]*(states[30] - variables[137] - 1) - states[31])*(states[31]*variables[138]/(states[30] + variables[140]) + variables[139])
        #eta1_9 = Tai_9*(c_33 + (c_131 - c_33)*Heaviside(-Tai_9 + c_133)*Heaviside(u_9 - c_132))
        variables[265] = states[32]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[32] + variables[133])*Heaviside(states[34] - variables[132]))
        #eta2_9 = c_33 + (c_131 - c_33)*Heaviside(-Tai_9 + c_133)*Heaviside(u_9 - c_132)
        variables[266] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[32] + variables[133])*Heaviside(states[34] - variables[132])
        #kV_9 = exp(-0.5*(u_9 - 1)**2/c_134**2)/(c_134*c_135)
        variables[267] = exp(-0.5*(states[34] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_9 = -u_9*v_9 + u_9*c_136*(1 - u_9)*(u_9 - c_137)
        variables[268] = -states[34]*states[35] + states[34]*variables[136]*(1 - states[34])*(states[34] - variables[137])
        #V_9 = (-u_9*c_136*(u_9 - c_137 - 1) - v_9)*(v_9*c_138/(u_9 + c_140) + c_139)
        variables[269] = (-states[34]*variables[136]*(states[34] - variables[137] - 1) - states[35])*(states[35]*variables[138]/(states[34] + variables[140]) + variables[139])
        #eta1_10 = Tai_10*(c_33 + (c_131 - c_33)*Heaviside(-Tai_10 + c_133)*Heaviside(u_10 - c_132))
        variables[270] = states[36]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[36] + variables[133])*Heaviside(states[38] - variables[132]))
        #eta2_10 = c_33 + (c_131 - c_33)*Heaviside(-Tai_10 + c_133)*Heaviside(u_10 - c_132)
        variables[271] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[36] + variables[133])*Heaviside(states[38] - variables[132])
        #kV_10 = exp(-0.5*(u_10 - 1)**2/c_134**2)/(c_134*c_135)
        variables[272] = exp(-0.5*(states[38] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_10 = -u_10*v_10 + u_10*c_136*(1 - u_10)*(u_10 - c_137)
        variables[273] = -states[38]*states[39] + states[38]*variables[136]*(1 - states[38])*(states[38] - variables[137])
        #V_10 = (-u_10*c_136*(u_10 - c_137 - 1) - v_10)*(v_10*c_138/(u_10 + c_140) + c_139)
        variables[274] = (-states[38]*variables[136]*(states[38] - variables[137] - 1) - states[39])*(states[39]*variables[138]/(states[38] + variables[140]) + variables[139])
        #eta1_11 = Tai_11*(c_33 + (c_131 - c_33)*Heaviside(-Tai_11 + c_133)*Heaviside(u_11 - c_132))
        variables[275] = states[40]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[40] + variables[133])*Heaviside(states[42] - variables[132]))
        #eta2_11 = c_33 + (c_131 - c_33)*Heaviside(-Tai_11 + c_133)*Heaviside(u_11 - c_132)
        variables[276] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[40] + variables[133])*Heaviside(states[42] - variables[132])
        #kV_11 = exp(-0.5*(u_11 - 1)**2/c_134**2)/(c_134*c_135)
        variables[277] = exp(-0.5*(states[42] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_11 = -u_11*v_11 + u_11*c_136*(1 - u_11)*(u_11 - c_137)
        variables[278] = -states[42]*states[43] + states[42]*variables[136]*(1 - states[42])*(states[42] - variables[137])
        #V_11 = (-u_11*c_136*(u_11 - c_137 - 1) - v_11)*(v_11*c_138/(u_11 + c_140) + c_139)
        variables[279] = (-states[42]*variables[136]*(states[42] - variables[137] - 1) - states[43])*(states[43]*variables[138]/(states[42] + variables[140]) + variables[139])
        #eta1_12 = Tai_12*(c_33 + (c_131 - c_33)*Heaviside(-Tai_12 + c_133)*Heaviside(u_12 - c_132))
        variables[280] = states[44]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[44] + variables[133])*Heaviside(states[46] - variables[132]))
        #eta2_12 = c_33 + (c_131 - c_33)*Heaviside(-Tai_12 + c_133)*Heaviside(u_12 - c_132)
        variables[281] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[44] + variables[133])*Heaviside(states[46] - variables[132])
        #kV_12 = exp(-0.5*(u_12 - 1)**2/c_134**2)/(c_134*c_135)
        variables[282] = exp(-0.5*(states[46] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_12 = -u_12*v_12 + u_12*c_136*(1 - u_12)*(u_12 - c_137)
        variables[283] = -states[46]*states[47] + states[46]*variables[136]*(1 - states[46])*(states[46] - variables[137])
        #V_12 = (-u_12*c_136*(u_12 - c_137 - 1) - v_12)*(v_12*c_138/(u_12 + c_140) + c_139)
        variables[284] = (-states[46]*variables[136]*(states[46] - variables[137] - 1) - states[47])*(states[47]*variables[138]/(states[46] + variables[140]) + variables[139])
        #eta1_13 = Tai_13*(c_33 + (c_131 - c_33)*Heaviside(-Tai_13 + c_133)*Heaviside(u_13 - c_132))
        variables[285] = states[48]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[48] + variables[133])*Heaviside(states[50] - variables[132]))
        #eta2_13 = c_33 + (c_131 - c_33)*Heaviside(-Tai_13 + c_133)*Heaviside(u_13 - c_132)
        variables[286] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[48] + variables[133])*Heaviside(states[50] - variables[132])
        #kV_13 = exp(-0.5*(u_13 - 1)**2/c_134**2)/(c_134*c_135)
        variables[287] = exp(-0.5*(states[50] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_13 = -u_13*v_13 + u_13*c_136*(1 - u_13)*(u_13 - c_137)
        variables[288] = -states[50]*states[51] + states[50]*variables[136]*(1 - states[50])*(states[50] - variables[137])
        #V_13 = (-u_13*c_136*(u_13 - c_137 - 1) - v_13)*(v_13*c_138/(u_13 + c_140) + c_139)
        variables[289] = (-states[50]*variables[136]*(states[50] - variables[137] - 1) - states[51])*(states[51]*variables[138]/(states[50] + variables[140]) + variables[139])
        #eta1_14 = Tai_14*(c_33 + (c_131 - c_33)*Heaviside(-Tai_14 + c_133)*Heaviside(u_14 - c_132))
        variables[290] = states[52]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[52] + variables[133])*Heaviside(states[54] - variables[132]))
        #eta2_14 = c_33 + (c_131 - c_33)*Heaviside(-Tai_14 + c_133)*Heaviside(u_14 - c_132)
        variables[291] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[52] + variables[133])*Heaviside(states[54] - variables[132])
        #kV_14 = exp(-0.5*(u_14 - 1)**2/c_134**2)/(c_134*c_135)
        variables[292] = exp(-0.5*(states[54] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_14 = -u_14*v_14 + u_14*c_136*(1 - u_14)*(u_14 - c_137)
        variables[293] = -states[54]*states[55] + states[54]*variables[136]*(1 - states[54])*(states[54] - variables[137])
        #V_14 = (-u_14*c_136*(u_14 - c_137 - 1) - v_14)*(v_14*c_138/(u_14 + c_140) + c_139)
        variables[294] = (-states[54]*variables[136]*(states[54] - variables[137] - 1) - states[55])*(states[55]*variables[138]/(states[54] + variables[140]) + variables[139])
        #eta1_15 = Tai_15*(c_33 + (c_131 - c_33)*Heaviside(-Tai_15 + c_133)*Heaviside(u_15 - c_132))
        variables[295] = states[56]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[56] + variables[133])*Heaviside(states[58] - variables[132]))
        #eta2_15 = c_33 + (c_131 - c_33)*Heaviside(-Tai_15 + c_133)*Heaviside(u_15 - c_132)
        variables[296] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[56] + variables[133])*Heaviside(states[58] - variables[132])
        #kV_15 = exp(-0.5*(u_15 - 1)**2/c_134**2)/(c_134*c_135)
        variables[297] = exp(-0.5*(states[58] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_15 = -u_15*v_15 + u_15*c_136*(1 - u_15)*(u_15 - c_137)
        variables[298] = -states[58]*states[59] + states[58]*variables[136]*(1 - states[58])*(states[58] - variables[137])
        #V_15 = (-u_15*c_136*(u_15 - c_137 - 1) - v_15)*(v_15*c_138/(u_15 + c_140) + c_139)
        variables[299] = (-states[58]*variables[136]*(states[58] - variables[137] - 1) - states[59])*(states[59]*variables[138]/(states[58] + variables[140]) + variables[139])
        #eta1_16 = Tai_16*(c_33 + (c_131 - c_33)*Heaviside(-Tai_16 + c_133)*Heaviside(u_16 - c_132))
        variables[300] = states[60]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[60] + variables[133])*Heaviside(states[62] - variables[132]))
        #eta2_16 = c_33 + (c_131 - c_33)*Heaviside(-Tai_16 + c_133)*Heaviside(u_16 - c_132)
        variables[301] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[60] + variables[133])*Heaviside(states[62] - variables[132])
        #kV_16 = exp(-0.5*(u_16 - 1)**2/c_134**2)/(c_134*c_135)
        variables[302] = exp(-0.5*(states[62] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_16 = -u_16*v_16 + u_16*c_136*(1 - u_16)*(u_16 - c_137)
        variables[303] = -states[62]*states[63] + states[62]*variables[136]*(1 - states[62])*(states[62] - variables[137])
        #V_16 = (-u_16*c_136*(u_16 - c_137 - 1) - v_16)*(v_16*c_138/(u_16 + c_140) + c_139)
        variables[304] = (-states[62]*variables[136]*(states[62] - variables[137] - 1) - states[63])*(states[63]*variables[138]/(states[62] + variables[140]) + variables[139])
        #eta1_17 = Tai_17*(c_33 + (c_131 - c_33)*Heaviside(-Tai_17 + c_133)*Heaviside(u_17 - c_132))
        variables[305] = states[64]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[64] + variables[133])*Heaviside(states[66] - variables[132]))
        #eta2_17 = c_33 + (c_131 - c_33)*Heaviside(-Tai_17 + c_133)*Heaviside(u_17 - c_132)
        variables[306] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[64] + variables[133])*Heaviside(states[66] - variables[132])
        #kV_17 = exp(-0.5*(u_17 - 1)**2/c_134**2)/(c_134*c_135)
        variables[307] = exp(-0.5*(states[66] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_17 = -u_17*v_17 + u_17*c_136*(1 - u_17)*(u_17 - c_137)
        variables[308] = -states[66]*states[67] + states[66]*variables[136]*(1 - states[66])*(states[66] - variables[137])
        #V_17 = (-u_17*c_136*(u_17 - c_137 - 1) - v_17)*(v_17*c_138/(u_17 + c_140) + c_139)
        variables[309] = (-states[66]*variables[136]*(states[66] - variables[137] - 1) - states[67])*(states[67]*variables[138]/(states[66] + variables[140]) + variables[139])
        #eta1_18 = Tai_18*(c_33 + (c_131 - c_33)*Heaviside(-Tai_18 + c_133)*Heaviside(u_18 - c_132))
        variables[310] = states[68]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[68] + variables[133])*Heaviside(states[70] - variables[132]))
        #eta2_18 = c_33 + (c_131 - c_33)*Heaviside(-Tai_18 + c_133)*Heaviside(u_18 - c_132)
        variables[311] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[68] + variables[133])*Heaviside(states[70] - variables[132])
        #kV_18 = exp(-0.5*(u_18 - 1)**2/c_134**2)/(c_134*c_135)
        variables[312] = exp(-0.5*(states[70] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_18 = -u_18*v_18 + u_18*c_136*(1 - u_18)*(u_18 - c_137)
        variables[313] = -states[70]*states[71] + states[70]*variables[136]*(1 - states[70])*(states[70] - variables[137])
        #V_18 = (-u_18*c_136*(u_18 - c_137 - 1) - v_18)*(v_18*c_138/(u_18 + c_140) + c_139)
        variables[314] = (-states[70]*variables[136]*(states[70] - variables[137] - 1) - states[71])*(states[71]*variables[138]/(states[70] + variables[140]) + variables[139])
        #eta1_19 = Tai_19*(c_33 + (c_131 - c_33)*Heaviside(-Tai_19 + c_133)*Heaviside(u_19 - c_132))
        variables[315] = states[72]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[72] + variables[133])*Heaviside(states[74] - variables[132]))
        #eta2_19 = c_33 + (c_131 - c_33)*Heaviside(-Tai_19 + c_133)*Heaviside(u_19 - c_132)
        variables[316] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[72] + variables[133])*Heaviside(states[74] - variables[132])
        #kV_19 = exp(-0.5*(u_19 - 1)**2/c_134**2)/(c_134*c_135)
        variables[317] = exp(-0.5*(states[74] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_19 = -u_19*v_19 + u_19*c_136*(1 - u_19)*(u_19 - c_137)
        variables[318] = -states[74]*states[75] + states[74]*variables[136]*(1 - states[74])*(states[74] - variables[137])
        #V_19 = (-u_19*c_136*(u_19 - c_137 - 1) - v_19)*(v_19*c_138/(u_19 + c_140) + c_139)
        variables[319] = (-states[74]*variables[136]*(states[74] - variables[137] - 1) - states[75])*(states[75]*variables[138]/(states[74] + variables[140]) + variables[139])
        #eta1_20 = Tai_20*(c_33 + (c_131 - c_33)*Heaviside(-Tai_20 + c_133)*Heaviside(u_20 - c_132))
        variables[320] = states[76]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[76] + variables[133])*Heaviside(states[78] - variables[132]))
        #eta2_20 = c_33 + (c_131 - c_33)*Heaviside(-Tai_20 + c_133)*Heaviside(u_20 - c_132)
        variables[321] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[76] + variables[133])*Heaviside(states[78] - variables[132])
        #kV_20 = exp(-0.5*(u_20 - 1)**2/c_134**2)/(c_134*c_135)
        variables[322] = exp(-0.5*(states[78] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_20 = -u_20*v_20 + u_20*c_136*(1 - u_20)*(u_20 - c_137)
        variables[323] = -states[78]*states[79] + states[78]*variables[136]*(1 - states[78])*(states[78] - variables[137])
        #V_20 = (-u_20*c_136*(u_20 - c_137 - 1) - v_20)*(v_20*c_138/(u_20 + c_140) + c_139)
        variables[324] = (-states[78]*variables[136]*(states[78] - variables[137] - 1) - states[79])*(states[79]*variables[138]/(states[78] + variables[140]) + variables[139])
        #eta1_21 = Tai_21*(c_33 + (c_131 - c_33)*Heaviside(-Tai_21 + c_133)*Heaviside(u_21 - c_132))
        variables[325] = states[80]*(variables[33] + (variables[131] - variables[33])*Heaviside(-states[80] + variables[133])*Heaviside(states[82] - variables[132]))
        #eta2_21 = c_33 + (c_131 - c_33)*Heaviside(-Tai_21 + c_133)*Heaviside(u_21 - c_132)
        variables[326] = variables[33] + (variables[131] - variables[33])*Heaviside(-states[80] + variables[133])*Heaviside(states[82] - variables[132])
        #kV_21 = exp(-0.5*(u_21 - 1)**2/c_134**2)/(c_134*c_135)
        variables[327] = exp(-0.5*(states[82] - 1)**2/variables[134]**2)/(variables[134]*variables[135])
        #U_21 = -u_21*v_21 + u_21*c_136*(1 - u_21)*(u_21 - c_137)
        variables[328] = -states[82]*states[83] + states[82]*variables[136]*(1 - states[82])*(states[82] - variables[137])
        #V_21 = (-u_21*c_136*(u_21 - c_137 - 1) - v_21)*(v_21*c_138/(u_21 + c_140) + c_139)
        variables[329] = (-states[82]*variables[136]*(states[82] - variables[137] - 1) - states[83])*(states[83]*variables[138]/(states[82] + variables[140]) + variables[139])
        #\dot{Tai_1} = -0.016602*Tai_1 + kV_1*c_33 # -0.016602*Tai_1 + 0.154143227709786*u_1*exp(-270.806766140199*(u_1 - 1)**2)
        rates[0] = -0.016602*states[0] + variables[227]*variables[33]
        #\dot{Ta_1} = -Ta_1*eta2_1 + eta1_1 # Ta_1*(0.016502*Heaviside(0.2925 - Tai_1)*Heaviside(u_1 - 0.78) - 0.016602) + Tai_(-0.016502*Heaviside(0.2925 - Tai_1)*Heaviside(u_1 - 0.78) + 0.016602)
        rates[1] = -states[1]*variables[226] + variables[225]
        #\dot{u_1} = c_10*\hat{u}_u_20 + c_113*\hat{u}_u_4 - 0.0105347817232959*\hat{u}_u_1 + \hat{u}_u_8*c_24 + U_1*c_28 # -u_(0.0775*v_1 + 0.62*(u_1 - 1)*(u_1 - 0.13)) - 0.0105347817232959*u_u_1 + 0.00134233937586588*u_u_20 + 0.0034875*u_u_4 + 0.00570494234742999*u_u_8
        rates[2] = variables[10]*variables[219] + variables[113]*variables[155] - 0.0105347817232959*variables[143] + variables[171]*variables[24] + variables[228]*variables[28]
        #\dot{v_1} = V_1*c_28 # -0.0775*v_1*(8.0*u_1*(u_1 - 1.13) + v_1)*(0.002*u_1 + 0.2*v_1 + 0.0006)/(u_1 + 0.3)
        rates[3] = variables[229]*variables[28]
        #\dot{Tai_2} = -0.016602*Tai_2 + kV_2*c_33 # -0.016602*Tai_2 + 0.154143227709786*u_2*exp(-270.806766140199*(u_2 - 1)**2)
        rates[4] = -0.016602*states[4] + variables[232]*variables[33]
        #\dot{Ta_2} = -Ta_2*eta2_2 + eta1_2 # Ta_2*(0.016502*Heaviside(0.2925 - Tai_2)*Heaviside(u_2 - 0.78) - 0.016602) + Tai_2**2*(-0.016502*Heaviside(0.2925 - Tai_2)*Heaviside(u_2 - 0.78) + 0.016602)
        rates[5] = -states[5]*variables[231] + variables[230]
        #\dot{u_2} = c_112*\hat{u}_u_17 - 0.0187547835509272*\hat{u}_u_2 + \hat{u}_u_9*c_59 + \hat{u}_u_10*c_31 + \hat{u}_u_14*c_7 + \hat{u}_u_20*c_84 + U_2*c_28 # -u_2**2*(0.0775*v_2 + 0.62*(u_2 - 1)*(u_2 - 0.13)) + 0.00553238341512669*u_u_10 + 0.00335584843966471*u_u_14 + 0.00624309617371499*u_u_17 - 0.0187547835509272*u_u_2 + 0.00183366968793294*u_u_20 + 0.00178978583448784*u_u_9
        rates[6] = variables[112]*variables[207] - 0.0187547835509272*variables[147] + variables[175]*variables[59] + variables[179]*variables[31] + variables[195]*variables[7] + variables[219]*variables[84] + variables[233]*variables[28]
        #\dot{v_2} = V_2*c_28 # -0.0775*v_2*(8.0*u_2*(u_2 - 1.13) + v_2)*(0.002*u_2 + 0.2*v_2 + 0.0006)/(u_2 + 0.3)
        rates[7] = variables[234]*variables[28]
        #\dot{Tai_3} = -0.016602*Tai_3 + kV_3*c_33 # -0.016602*Tai_3 + 0.154143227709786*u_3*exp(-270.806766140199*(u_3 - 1)**2)
        rates[8] = -0.016602*states[8] + variables[237]*variables[33]
        #\dot{Ta_3} = -Ta_3*eta2_3 + eta1_3 # Ta_3*(0.016502*Heaviside(0.2925 - Tai_3)*Heaviside(u_3 - 0.78) - 0.016602) + Tai_3**2*(-0.016502*Heaviside(0.2925 - Tai_3)*Heaviside(u_3 - 0.78) + 0.016602)
        rates[9] = -states[9]*variables[236] + variables[235]
        #\dot{u_3} = c_106*\hat{u}_u_6 + c_107*\hat{u}_u_20 - 0.00994487038649063*\hat{u}_u_3 + \hat{u}_u_15*c_50 + \hat{u}_u_19*c_23 + U_3*c_28 # -u_3**2*(0.0775*v_3 + 0.62*(u_3 - 1)*(u_3 - 0.13)) + 0.000266912464258686*u_u_15 + 0.00233123221661754*u_u_19 + 0.00234909390776529*u_u_20 - 0.00994487038649063*u_u_3 + 0.00499763179784911*u_u_6
        rates[10] = variables[106]*variables[163] + variables[107]*variables[219] - 0.00994487038649063*variables[151] + variables[199]*variables[50] + variables[215]*variables[23] + variables[238]*variables[28]
        #\dot{v_3} = V_3*c_28 # -0.0775*v_3*(8.0*u_3*(u_3 - 1.13) + v_3)*(0.002*u_3 + 0.2*v_3 + 0.0006)/(u_3 + 0.3)
        rates[11] = variables[239]*variables[28]
        #\dot{Tai_4} = -0.016602*Tai_4 + kV_4*c_33 # -0.016602*Tai_4 + 0.154143227709786*u_4*exp(-270.806766140199*(u_4 - 1)**2)
        rates[12] = -0.016602*states[12] + variables[242]*variables[33]
        #\dot{Ta_4} = -Ta_4*eta2_4 + eta1_4 # Ta_4*(0.016502*Heaviside(0.2925 - Tai_4)*Heaviside(u_4 - 0.78) - 0.016602) + Tai_4**2*(-0.016502*Heaviside(0.2925 - Tai_4)*Heaviside(u_4 - 0.78) + 0.016602)
        rates[13] = -states[13]*variables[241] + variables[240]
        #\dot{u_4} = 1.0*i_1_2 + c_113*\hat{u}_u_1 - 0.0100670263595698*\hat{u}_u_4 + \hat{u}_u_15*c_53 + \hat{u}_u_19*c_90 + U_4*c_28 # 1.0*i_1_2 - u_4**2*(0.0775*v_4 + 0.62*(u_4 - 1)*(u_4 - 0.13)) + 0.0034875*u_u_1 + 0.00225306424965365*u_u_15 + 0.00432646210991617*u_u_19 - 0.0100670263595698*u_u_4
        rates[14] = 1.0*variables[0] + variables[113]*variables[143] - 0.0100670263595698*variables[155] + variables[199]*variables[53] + variables[215]*variables[90] + variables[243]*variables[28]
        #\dot{v_4} = V_4*c_28 # -0.0775*v_4*(8.0*u_4*(u_4 - 1.13) + v_4)*(0.002*u_4 + 0.2*v_4 + 0.0006)/(u_4 + 0.3)
        rates[15] = variables[244]*variables[28]
        #\dot{Tai_5} = -0.016602*Tai_5 + kV_5*c_33 # -0.016602*Tai_5 + 0.154143227709786*u_5*exp(-270.806766140199*(u_5 - 1)**2)
        rates[16] = -0.016602*states[16] + variables[247]*variables[33]
        #\dot{Ta_5} = -Ta_5*eta2_5 + eta1_5 # Ta_5*(0.016502*Heaviside(0.2925 - Tai_5)*Heaviside(u_5 - 0.78) - 0.016602) + Tai_5**2*(-0.016502*Heaviside(0.2925 - Tai_5)*Heaviside(u_5 - 0.78) + 0.016602)
        rates[17] = -states[17]*variables[246] + variables[245]
        #\dot{u_5} = 1.0*i_1_2 + c_14*\hat{u}_u_8 - 0.00908557896398634*\hat{u}_u_5 + \hat{u}_u_11*c_58 + \hat{u}_u_12*c_69 + \hat{u}_u_18*c_7 + U_5*c_28 # 1.0*i_1_2 - u_5**2*(0.0775*v_5 + 0.62*(u_5 - 1)*(u_5 - 0.13)) + 0.00181486232217075*u_u_11 + 0.0019375*u_u_12 + 0.0033558484396647*u_u_18 - 0.00908557896398634*u_u_5 + 0.00197736820215088*u_u_8
        rates[18] = 1.0*variables[0] + variables[14]*variables[171] - 0.00908557896398634*variables[159] + variables[183]*variables[58] + variables[187]*variables[69] + variables[211]*variables[7] + variables[248]*variables[28]
        #\dot{v_5} = V_5*c_28 # -0.0775*v_5*(8.0*u_5*(u_5 - 1.13) + v_5)*(0.002*u_5 + 0.2*v_5 + 0.0006)/(u_5 + 0.3)
        rates[19] = variables[249]*variables[28]
        #\dot{Tai_6} = -0.016602*Tai_6 + kV_6*c_33 # -0.016602*Tai_6 + 0.154143227709786*u_6*exp(-270.806766140199*(u_6 - 1)**2)
        rates[20] = -0.016602*states[20] + variables[252]*variables[33]
        #\dot{Ta_6} = -Ta_6*eta2_6 + eta1_6 # Ta_6*(0.016502*Heaviside(0.2925 - Tai_6)*Heaviside(u_6 - 0.78) - 0.016602) + Tai_6**2*(-0.016502*Heaviside(0.2925 - Tai_6)*Heaviside(u_6 - 0.78) + 0.016602)
        rates[21] = -states[21]*variables[251] + variables[250]
        #\dot{u_6} = c_106*\hat{u}_u_3 - 0.0136866968793294*\hat{u}_u_6 + \hat{u}_u_8*c_52 + \hat{u}_u_18*c_5 + U_6*c_28 # -u_6**2*(0.0775*v_6 + 0.62*(u_6 - 1)*(u_6 - 0.13)) + 0.00604052719139646*u_u_18 + 0.00499763179784911*u_u_3 - 0.0136866968793294*u_u_6 + 0.00264853789008382*u_u_8
        rates[22] = variables[106]*variables[151] - 0.0136866968793294*variables[163] + variables[171]*variables[52] + variables[211]*variables[5] + variables[253]*variables[28]
        #\dot{v_6} = V_6*c_28 # -0.0775*v_6*(8.0*u_6*(u_6 - 1.13) + v_6)*(0.002*u_6 + 0.2*v_6 + 0.0006)/(u_6 + 0.3)
        rates[23] = variables[254]*variables[28]
        #\dot{Tai_7} = -0.016602*Tai_7 + kV_7*c_33 # -0.016602*Tai_7 + 0.154143227709786*u_7*exp(-270.806766140199*(u_7 - 1)**2)
        rates[24] = -0.016602*states[24] + variables[257]*variables[33]
        #\dot{Ta_7} = -Ta_7*eta2_7 + eta1_7 # Ta_7*(0.016502*Heaviside(0.2925 - Tai_7)*Heaviside(u_7 - 0.78) - 0.016602) + Tai_7**2*(-0.016502*Heaviside(0.2925 - Tai_7)*Heaviside(u_7 - 0.78) + 0.016602)
        rates[25] = -states[25]*variables[256] + variables[255]
        #\dot{u_7} = -0.00660632319563326*\hat{u}_u_7 + \hat{u}_u_15*c_92 + \hat{u}_u_21*c_7 + U_7*c_28 # -u_7**2*(0.0775*v_7 + 0.62*(u_7 - 1)*(u_7 - 0.13)) + 0.00325047475596855*u_u_15 + 0.0033558484396647*u_u_21 - 0.00660632319563326*u_u_7
        rates[26] = -0.00660632319563326*variables[167] + variables[199]*variables[92] + variables[223]*variables[7] + variables[258]*variables[28]
        #\dot{v_7} = V_7*c_28 # -0.0775*v_7*(8.0*u_7*(u_7 - 1.13) + v_7)*(0.002*u_7 + 0.2*v_7 + 0.0006)/(u_7 + 0.3)
        rates[27] = variables[259]*variables[28]
        #\dot{Tai_8} = -0.016602*Tai_8 + kV_8*c_33 # -0.016602*Tai_8 + 0.154143227709786*u_8*exp(-270.806766140199*(u_8 - 1)**2)
        rates[28] = -0.016602*states[28] + variables[262]*variables[33]
        #\dot{Ta_8} = -Ta_8*eta2_8 + eta1_8 # Ta_8*(0.016502*Heaviside(0.2925 - Tai_8)*Heaviside(u_8 - 0.78) - 0.016602) + Tai_8**2*(-0.016502*Heaviside(0.2925 - Tai_8)*Heaviside(u_8 - 0.78) + 0.016602)
        rates[29] = -states[29]*variables[261] + variables[260]
        #\dot{u_8} = c_130*\hat{u}_u_19 + \hat{u}_u_1*c_24 + c_14*\hat{u}_u_5 + \hat{u}_u_6*c_52 - 0.0172670458146408*\hat{u}_u_8 + \hat{u}_u_13*c_36 + \hat{u}_u_16*c_80 + U_8*c_28 # -u_8**2*(0.0775*v_8 + 0.62*(u_8 - 1)*(u_8 - 0.13)) + 0.00570494234742999*u_u_1 + 0.00233340673210136*u_u_13 + 0.00331131228061603*u_u_16 + 0.00129147836225876*u_u_19 + 0.00197736820215088*u_u_5 + 0.00264853789008382*u_u_6 - 0.0172670458146408*u_u_8
        rates[30] = variables[130]*variables[215] + variables[143]*variables[24] + variables[14]*variables[159] + variables[163]*variables[52] - 0.0172670458146408*variables[171] + variables[191]*variables[36] + variables[203]*variables[80] + variables[263]*variables[28]
        #\dot{v_8} = V_8*c_28 # -0.0775*v_8*(8.0*u_8*(u_8 - 1.13) + v_8)*(0.002*u_8 + 0.2*v_8 + 0.0006)/(u_8 + 0.3)
        rates[31] = variables[264]*variables[28]
        #\dot{Tai_9} = -0.016602*Tai_9 + kV_9*c_33 # -0.016602*Tai_9 + 0.154143227709786*u_9*exp(-270.806766140199*(u_9 - 1)**2)
        rates[32] = -0.016602*states[32] + variables[267]*variables[33]
        #\dot{Ta_9} = -Ta_9*eta2_9 + eta1_9 # Ta_9*(0.016502*Heaviside(0.2925 - Tai_9)*Heaviside(u_9 - 0.78) - 0.016602) + Tai_9**2*(-0.016502*Heaviside(0.2925 - Tai_9)*Heaviside(u_9 - 0.78) + 0.016602)
        rates[33] = -states[33]*variables[266] + variables[265]
        #\dot{u_9} = c_113*\hat{u}_u_17 + c_124*\hat{u}_u_21 + \hat{u}_u_2*c_59 - 0.0284904787482649*\hat{u}_u_9 + \hat{u}_u_11*c_99 + \hat{u}_u_13*c_39 + \hat{u}_u_14*c_63 + \hat{u}_u_15*c_74 + \hat{u}_u_19*c_88 + U_9*c_28 # -u_9**2*(0.0775*v_9 + 0.62*(u_9 - 1)*(u_9 - 0.13)) + 0.0110020181275976*u_u_11 + 0.00177666289008382*u_u_13 + 0.00100675453189941*u_u_14 + 2.3604678562608e-5*u_u_15 + 0.0034875*u_u_17 + 0.00135011643043833*u_u_19 + 0.00178978583448784*u_u_2 + 0.00805403625519528*u_u_21 - 0.0284904787482649*u_u_9
        rates[34] = variables[113]*variables[207] + variables[124]*variables[223] + variables[147]*variables[59] - 0.0284904787482649*variables[175] + variables[183]*variables[99] + variables[191]*variables[39] + variables[195]*variables[63] + variables[199]*variables[74] + variables[215]*variables[88] + variables[268]*variables[28]
        #\dot{v_9} = V_9*c_28 # -0.0775*v_9*(8.0*u_9*(u_9 - 1.13) + v_9)*(0.002*u_9 + 0.2*v_9 + 0.0006)/(u_9 + 0.3)
        rates[35] = variables[269]*variables[28]
        #\dot{Tai_10} = -0.016602*Tai_10 + kV_10*c_33 # -0.016602*Tai_10 + 0.154143227709786*u_10*exp(-270.806766140199*(u_10 - 1)**2)
        rates[36] = -0.016602*states[36] + variables[272]*variables[33]
        #\dot{Ta_10} = -Ta_10*eta2_10 + eta1_10 # Ta_10*(0.016502*Heaviside(0.2925 - Tai_10)*Heaviside(u_10 - 0.78) - 0.016602) + Tai_10**2*(-0.016502*Heaviside(0.2925 - Tai_10)*Heaviside(u_10 - 0.78) + 0.016602)
        rates[37] = -states[37]*variables[271] + variables[270]
        #\dot{u_10} = c_106*\hat{u}_u_19 + \hat{u}_u_2*c_31 - 0.0151800152129758*\hat{u}_u_10 + \hat{u}_u_15*c_89 + U_10*c_28 # -u_10**2*(0.0775*v_10 + 0.62*(u_10 - 1)*(u_10 - 0.13)) - 0.0151800152129758*u_u_10 + 0.00465*u_u_15 + 0.00499763179784911*u_u_19 + 0.00553238341512669*u_u_2
        rates[38] = variables[106]*variables[215] + variables[147]*variables[31] - 0.0151800152129758*variables[179] + variables[199]*variables[89] + variables[273]*variables[28]
        #\dot{v_10} = V_10*c_28 # -0.0775*v_10*(8.0*u_10*(u_10 - 1.13) + v_10)*(0.002*u_10 + 0.2*v_10 + 0.0006)/(u_10 + 0.3)
        rates[39] = variables[274]*variables[28]
        #\dot{Tai_11} = -0.016602*Tai_11 + kV_11*c_33 # -0.016602*Tai_11 + 0.154143227709786*u_11*exp(-270.806766140199*(u_11 - 1)**2)
        rates[40] = -0.016602*states[40] + variables[277]*variables[33]
        #\dot{Ta_11} = -Ta_11*eta2_11 + eta1_11 # Ta_11*(0.016502*Heaviside(0.2925 - Tai_11)*Heaviside(u_11 - 0.78) - 0.016602) + Tai_1(-0.016502*Heaviside(0.2925 - Tai_11)*Heaviside(u_11 - 0.78) + 0.016602)
        rates[41] = -states[41]*variables[276] + variables[275]
        #\dot{u_11} = 1.0*i_1_2 + c_130*\hat{u}_u_17 + \hat{u}_u_5*c_58 + c_15*\hat{u}_u_19 + \hat{u}_u_9*c_99 - 0.0209989121379338*\hat{u}_u_11 + \hat{u}_u_12*c_27 + \hat{u}_u_13*c_50 + \hat{u}_u_15*c_69 + U_11*c_28 # 1.0*i_1_2 - u_1(0.0775*v_11 + 0.62*(u_11 - 1)*(u_11 - 0.13)) - 0.0209989121379338*u_u_11 + 0.00462913406379882*u_u_12 + 0.000266912464258687*u_u_13 + 0.0019375*u_u_15 + 0.00129147836225876*u_u_17 + 5.70067978491158e-5*u_u_19 + 0.00181486232217075*u_u_5 + 0.0110020181275976*u_u_9
        rates[42] = 1.0*variables[0] + variables[130]*variables[207] + variables[159]*variables[58] + variables[15]*variables[215] + variables[175]*variables[99] - 0.0209989121379338*variables[183] + variables[187]*variables[27] + variables[191]*variables[50] + variables[199]*variables[69] + variables[278]*variables[28]
        #\dot{v_11} = V_11*c_28 # -0.0775*v_11*(8.0*u_11*(u_11 - 1.13) + v_11)*(0.002*u_11 + 0.2*v_11 + 0.0006)/(u_11 + 0.3)
        rates[43] = variables[279]*variables[28]
        #\dot{Tai_12} = -0.016602*Tai_12 + kV_12*c_33 # -0.016602*Tai_12 + 0.154143227709786*u_12*exp(-270.806766140199*(u_12 - 1)**2)
        rates[44] = -0.016602*states[44] + variables[282]*variables[33]
        #\dot{Ta_12} = -Ta_12*eta2_12 + eta1_12 # Ta_12*(0.016502*Heaviside(0.2925 - Tai_12)*Heaviside(u_12 - 0.78) - 0.016602) + Tai_12**2*(-0.016502*Heaviside(0.2925 - Tai_12)*Heaviside(u_12 - 0.78) + 0.016602)
        rates[45] = -states[45]*variables[281] + variables[280]
        #\dot{u_12} = \hat{u}_u_5*c_69 + \hat{u}_u_11*c_27 - 0.0118998507056144*\hat{u}_u_12 + \hat{u}_u_13*c_90 + \hat{u}_u_16*c_63 + U_12*c_28 # -u_12**2*(0.0775*v_12 + 0.62*(u_12 - 1)*(u_12 - 0.13)) + 0.00462913406379882*u_u_11 - 0.0118998507056144*u_u_12 + 0.00432646210991617*u_u_13 + 0.00100675453189941*u_u_16 + 0.0019375*u_u_5
        rates[46] = variables[159]*variables[69] + variables[183]*variables[27] - 0.0118998507056144*variables[187] + variables[191]*variables[90] + variables[203]*variables[63] + variables[283]*variables[28]
        #\dot{v_12} = V_12*c_28 # -0.0775*v_12*(8.0*u_12*(u_12 - 1.13) + v_12)*(0.002*u_12 + 0.2*v_12 + 0.0006)/(u_12 + 0.3)
        rates[47] = variables[284]*variables[28]
        #\dot{Tai_13} = -0.016602*Tai_13 + kV_13*c_33 # -0.016602*Tai_13 + 0.154143227709786*u_13*exp(-270.806766140199*(u_13 - 1)**2)
        rates[48] = -0.016602*states[48] + variables[287]*variables[33]
        #\dot{Ta_13} = -Ta_13*eta2_13 + eta1_13 # Ta_13*(0.016502*Heaviside(0.2925 - Tai_13)*Heaviside(u_13 - 0.78) - 0.016602) + Tai_13**2*(-0.016502*Heaviside(0.2925 - Tai_13)*Heaviside(u_13 - 0.78) + 0.016602)
        rates[49] = -states[49]*variables[286] + variables[285]
        #\dot{u_13} = c_117*\hat{u}_u_19 + \hat{u}_u_8*c_36 + \hat{u}_u_9*c_39 + \hat{u}_u_11*c_50 + \hat{u}_u_12*c_90 - 0.022188359773101*\hat{u}_u_13 + \hat{u}_u_16*c_83 + \hat{u}_u_20*c_48 + U_13*c_28 # -u_13**2*(0.0775*v_13 + 0.62*(u_13 - 1)*(u_13 - 0.13)) + 0.000266912464258687*u_u_11 + 0.00432646210991617*u_u_12 - 0.022188359773101*u_u_13 + 0.00603034390776529*u_u_16 + 0.000611223229310978*u_u_19 + 0.0068433484396647*u_u_20 + 0.00233340673210136*u_u_8 + 0.00177666289008382*u_u_9
        rates[50] = variables[117]*variables[215] + variables[171]*variables[36] + variables[175]*variables[39] + variables[183]*variables[50] + variables[187]*variables[90] - 0.022188359773101*variables[191] + variables[203]*variables[83] + variables[219]*variables[48] + variables[288]*variables[28]
        #\dot{v_13} = V_13*c_28 # -0.0775*v_13*(8.0*u_13*(u_13 - 1.13) + v_13)*(0.002*u_13 + 0.2*v_13 + 0.0006)/(u_13 + 0.3)
        rates[51] = variables[289]*variables[28]
        #\dot{Tai_14} = -0.016602*Tai_14 + kV_14*c_33 # -0.016602*Tai_14 + 0.154143227709786*u_14*exp(-270.806766140199*(u_14 - 1)**2)
        rates[52] = -0.016602*states[52] + variables[292]*variables[33]
        #\dot{Ta_14} = -Ta_14*eta2_14 + eta1_14 # Ta_14*(0.016502*Heaviside(0.2925 - Tai_14)*Heaviside(u_14 - 0.78) - 0.016602) + Tai_14**2*(-0.016502*Heaviside(0.2925 - Tai_14)*Heaviside(u_14 - 0.78) + 0.016602)
        rates[53] = -states[53]*variables[291] + variables[290]
        #\dot{u_14} = \hat{u}_u_2*c_7 + \hat{u}_u_9*c_63 - 0.0128788020257737*\hat{u}_u_14 + \hat{u}_u_17*c_44 + \hat{u}_u_20*c_49 + c_28*U_14 # -u_14**2*(0.0775*v_14 + 0.62*(u_14 - 1)*(u_14 - 0.13)) - 0.0128788020257737*u_u_14 + 0.00812293911287484*u_u_17 + 0.00335584843966471*u_u_2 + 0.000393259941334723*u_u_20 + 0.00100675453189941*u_u_9
        rates[54] = variables[147]*variables[7] + variables[175]*variables[63] - 0.0128788020257737*variables[195] + variables[207]*variables[44] + variables[219]*variables[49] + variables[28]*variables[293]
        #\dot{v_14} = c_28*V_14 # -0.0775*v_14*(8.0*u_14*(u_14 - 1.13) + v_14)*(0.002*u_14 + 0.2*v_14 + 0.0006)/(u_14 + 0.3)
        rates[55] = variables[28]*variables[294]
        #\dot{Tai_15} = -0.016602*Tai_15 + kV_15*c_33 # -0.016602*Tai_15 + 0.154143227709786*u_15*exp(-270.806766140199*(u_15 - 1)**2)
        rates[56] = -0.016602*states[56] + variables[297]*variables[33]
        #\dot{Ta_15} = -Ta_15*eta2_15 + eta1_15 # Ta_15*(0.016502*Heaviside(0.2925 - Tai_15)*Heaviside(u_15 - 0.78) - 0.016602) + Tai_15**2*(-0.016502*Heaviside(0.2925 - Tai_15)*Heaviside(u_15 - 0.78) + 0.016602)
        rates[57] = -states[57]*variables[296] + variables[295]
        #\dot{u_15} = c_100*\hat{u}_u_18 + \hat{u}_u_3*c_50 + \hat{u}_u_4*c_53 + \hat{u}_u_7*c_92 + \hat{u}_u_9*c_74 + \hat{u}_u_10*c_89 + \hat{u}_u_11*c_69 - 0.0214542209029461*\hat{u}_u_15 + \hat{u}_u_19*c_5 + c_28*U_15 # -u_15**2*(0.0775*v_15 + 0.62*(u_15 - 1)*(u_15 - 0.13)) + 0.00465*u_u_10 + 0.0019375*u_u_11 - 0.0214542209029461*u_u_15 + 0.00303213756310611*u_u_18 + 0.00604052719139646*u_u_19 + 0.000266912464258686*u_u_3 + 0.00225306424965365*u_u_4 + 0.00325047475596855*u_u_7 + 2.3604678562608e-5*u_u_9
        rates[58] = variables[100]*variables[211] + variables[151]*variables[50] + variables[155]*variables[53] + variables[167]*variables[92] + variables[175]*variables[74] + variables[179]*variables[89] + variables[183]*variables[69] - 0.0214542209029461*variables[199] + variables[215]*variables[5] + variables[28]*variables[298]
        #\dot{v_15} = c_28*V_15 # -0.0775*v_15*(8.0*u_15*(u_15 - 1.13) + v_15)*(0.002*u_15 + 0.2*v_15 + 0.0006)/(u_15 + 0.3)
        rates[59] = variables[28]*variables[299]
        #\dot{Tai_16} = -0.016602*Tai_16 + kV_16*c_33 # -0.016602*Tai_16 + 0.154143227709786*u_16*exp(-270.806766140199*(u_16 - 1)**2)
        rates[60] = -0.016602*states[60] + variables[302]*variables[33]
        #\dot{Ta_16} = -Ta_16*eta2_16 + eta1_16 # Ta_16*(0.016502*Heaviside(0.2925 - Tai_16)*Heaviside(u_16 - 0.78) - 0.016602) + Tai_16**2*(-0.016502*Heaviside(0.2925 - Tai_16)*Heaviside(u_16 - 0.78) + 0.016602)
        rates[61] = -states[61]*variables[301] + variables[300]
        #\dot{u_16} = c_119*\hat{u}_u_20 + \hat{u}_u_8*c_80 + \hat{u}_u_12*c_63 + \hat{u}_u_13*c_83 - 0.0218062245551532*\hat{u}_u_16 + \hat{u}_u_17*c_35 + c_28*U_16 # -u_16**2*(0.0775*v_16 + 0.62*(u_16 - 1)*(u_16 - 0.13)) + 0.00100675453189941*u_u_12 + 0.00603034390776529*u_u_13 - 0.0218062245551532*u_u_16 + 0.00777166709324443*u_u_17 + 0.00368614674162807*u_u_20 + 0.00331131228061603*u_u_8
        rates[62] = variables[119]*variables[219] + variables[171]*variables[80] + variables[187]*variables[63] + variables[191]*variables[83] - 0.0218062245551532*variables[203] + variables[207]*variables[35] + variables[28]*variables[303]
        #\dot{v_16} = c_28*V_16 # -0.0775*v_16*(8.0*u_16*(u_16 - 1.13) + v_16)*(0.002*u_16 + 0.2*v_16 + 0.0006)/(u_16 + 0.3)
        rates[63] = variables[28]*variables[304]
        #\dot{Tai_17} = -0.016602*Tai_17 + kV_17*c_33 # -0.016602*Tai_17 + 0.154143227709786*u_17*exp(-270.806766140199*(u_17 - 1)**2)
        rates[64] = -0.016602*states[64] + variables[307]*variables[33]
        #\dot{Ta_17} = -Ta_17*eta2_17 + eta1_17 # Ta_17*(0.016502*Heaviside(0.2925 - Tai_17)*Heaviside(u_17 - 0.78) - 0.016602) + Tai_17**2*(-0.016502*Heaviside(0.2925 - Tai_17)*Heaviside(u_17 - 0.78) + 0.016602)
        rates[65] = -states[65]*variables[306] + variables[305]
        #\dot{u_17} = 1.0*i_1_2 + c_112*\hat{u}_u_2 + c_113*\hat{u}_u_9 + c_130*\hat{u}_u_11 + \hat{u}_u_14*c_44 + \hat{u}_u_16*c_35 - 0.0303833148058918*\hat{u}_u_17 + \hat{u}_u_20*c_94 + c_28*U_17 # 1.0*i_1_2 - u_17**2*(0.0775*v_17 + 0.62*(u_17 - 1)*(u_17 - 0.13)) + 0.00129147836225876*u_u_11 + 0.00812293911287484*u_u_14 + 0.00777166709324443*u_u_16 - 0.0303833148058918*u_u_17 + 0.00624309617371499*u_u_2 + 0.00346663406379882*u_u_20 + 0.0034875*u_u_9
        rates[66] = 1.0*variables[0] + variables[112]*variables[147] + variables[113]*variables[175] + variables[130]*variables[183] + variables[195]*variables[44] + variables[203]*variables[35] - 0.0303833148058918*variables[207] + variables[219]*variables[94] + variables[28]*variables[308]
        #\dot{v_17} = c_28*V_17 # -0.0775*v_17*(8.0*u_17*(u_17 - 1.13) + v_17)*(0.002*u_17 + 0.2*v_17 + 0.0006)/(u_17 + 0.3)
        rates[67] = variables[28]*variables[309]
        #\dot{Tai_18} = -0.016602*Tai_18 + kV_18*c_33 # -0.016602*Tai_18 + 0.154143227709786*u_18*exp(-270.806766140199*(u_18 - 1)**2)
        rates[68] = -0.016602*states[68] + variables[312]*variables[33]
        #\dot{Ta_18} = -Ta_18*eta2_18 + eta1_18 # Ta_18*(0.016502*Heaviside(0.2925 - Tai_18)*Heaviside(u_18 - 0.78) - 0.016602) + Tai_18**2*(-0.016502*Heaviside(0.2925 - Tai_18)*Heaviside(u_18 - 0.78) + 0.016602)
        rates[69] = -states[69]*variables[311] + variables[310]
        #\dot{u_18} = c_100*\hat{u}_u_15 + c_114*\hat{u}_u_19 + \hat{u}_u_5*c_7 + \hat{u}_u_6*c_5 - 0.0160249351480499*\hat{u}_u_18 + c_28*U_18 # -u_18**2*(0.0775*v_18 + 0.62*(u_18 - 1)*(u_18 - 0.13)) + 0.00303213756310611*u_u_15 - 0.0160249351480499*u_u_18 + 0.00359642195388264*u_u_19 + 0.0033558484396647*u_u_5 + 0.00604052719139646*u_u_6
        rates[70] = variables[100]*variables[199] + variables[114]*variables[215] + variables[159]*variables[7] + variables[163]*variables[5] - 0.0160249351480499*variables[211] + variables[28]*variables[313]
        #\dot{v_18} = c_28*V_18 # -0.0775*v_18*(8.0*u_18*(u_18 - 1.13) + v_18)*(0.002*u_18 + 0.2*v_18 + 0.0006)/(u_18 + 0.3)
        rates[71] = variables[28]*variables[314]
        #\dot{Tai_19} = -0.016602*Tai_19 + kV_19*c_33 # -0.016602*Tai_19 + 0.154143227709786*u_19*exp(-270.806766140199*(u_19 - 1)**2)
        rates[72] = -0.016602*states[72] + variables[317]*variables[33]
        #\dot{Ta_19} = -Ta_19*eta2_19 + eta1_19 # Ta_19*(0.016502*Heaviside(0.2925 - Tai_19)*Heaviside(u_19 - 0.78) - 0.016602) + Tai_19**2*(-0.016502*Heaviside(0.2925 - Tai_19)*Heaviside(u_19 - 0.78) + 0.016602)
        rates[73] = -states[73]*variables[316] + variables[315]
        #\dot{u_19} = c_106*\hat{u}_u_10 + c_108*\hat{u}_u_20 + c_114*\hat{u}_u_18 + c_117*\hat{u}_u_13 + c_130*\hat{u}_u_8 + \hat{u}_u_3*c_23 + \hat{u}_u_4*c_90 + c_15*\hat{u}_u_11 + \hat{u}_u_9*c_88 + \hat{u}_u_15*c_5 - 0.0272967553522912*\hat{u}_u_19 + c_28*U_19 # -u_19**2*(0.0775*v_19 + 0.62*(u_19 - 1)*(u_19 - 0.13)) + 0.00499763179784911*u_u_10 + 5.70067978491158e-5*u_u_11 + 0.000611223229310978*u_u_13 + 0.00604052719139646*u_u_15 + 0.00359642195388264*u_u_18 - 0.0272967553522912*u_u_19 + 0.00269465526277211*u_u_20 + 0.00233123221661754*u_u_3 + 0.00432646210991617*u_u_4 + 0.00129147836225876*u_u_8 + 0.00135011643043833*u_u_9
        rates[74] = variables[106]*variables[179] + variables[108]*variables[219] + variables[114]*variables[211] + variables[117]*variables[191] + variables[130]*variables[171] + variables[151]*variables[23] + variables[155]*variables[90] + variables[15]*variables[183] + variables[175]*variables[88] + variables[199]*variables[5] - 0.0272967553522912*variables[215] + variables[28]*variables[318]
        #\dot{v_19} = c_28*V_19 # -0.0775*v_19*(8.0*u_19*(u_19 - 1.13) + v_19)*(0.002*u_19 + 0.2*v_19 + 0.0006)/(u_19 + 0.3)
        rates[75] = variables[28]*variables[319]
        #\dot{Tai_20} = -0.016602*Tai_20 + kV_20*c_33 # -0.016602*Tai_20 + 0.154143227709786*u_20*exp(-270.806766140199*(u_20 - 1)**2)
        rates[76] = -0.016602*states[76] + variables[322]*variables[33]
        #\dot{Ta_20} = -Ta_20*eta2_20 + eta1_20 # Ta_20*(0.016502*Heaviside(0.2925 - Tai_20)*Heaviside(u_20 - 0.78) - 0.016602) + Tai_20**2*(-0.016502*Heaviside(0.2925 - Tai_20)*Heaviside(u_20 - 0.78) + 0.016602)
        rates[77] = -states[77]*variables[321] + variables[320]
        #\dot{u_20} = c_107*\hat{u}_u_3 + c_108*\hat{u}_u_19 + c_10*\hat{u}_u_1 + c_119*\hat{u}_u_16 + \hat{u}_u_2*c_84 + \hat{u}_u_13*c_48 + \hat{u}_u_14*c_49 + \hat{u}_u_17*c_94 - 0.0226091474207625*\hat{u}_u_20 + c_28*U_20 # -u_20**2*(0.0775*v_20 + 0.62*(u_20 - 1)*(u_20 - 0.13)) + 0.00134233937586588*u_u_1 + 0.0068433484396647*u_u_13 + 0.000393259941334723*u_u_14 + 0.00368614674162807*u_u_16 + 0.00346663406379882*u_u_17 + 0.00269465526277211*u_u_19 + 0.00183366968793294*u_u_2 - 0.0226091474207625*u_u_20 + 0.00234909390776529*u_u_3
        rates[78] = variables[107]*variables[151] + variables[108]*variables[215] + variables[10]*variables[143] + variables[119]*variables[203] + variables[147]*variables[84] + variables[191]*variables[48] + variables[195]*variables[49] + variables[207]*variables[94] - 0.0226091474207625*variables[219] + variables[28]*variables[323]
        #\dot{v_20} = c_28*V_20 # -0.0775*v_20*(8.0*u_20*(u_20 - 1.13) + v_20)*(0.002*u_20 + 0.2*v_20 + 0.0006)/(u_20 + 0.3)
        rates[79] = variables[28]*variables[324]
        #\dot{Tai_21} = -0.016602*Tai_21 + kV_21*c_33 # -0.016602*Tai_21 + 0.154143227709786*u_21*exp(-270.806766140199*(u_21 - 1)**2)
        rates[80] = -0.016602*states[80] + variables[327]*variables[33]
        #\dot{Ta_21} = -Ta_21*eta2_21 + eta1_21 # Ta_21*(0.016502*Heaviside(0.2925 - Tai_21)*Heaviside(u_21 - 0.78) - 0.016602) + Tai_2(-0.016502*Heaviside(0.2925 - Tai_21)*Heaviside(u_21 - 0.78) + 0.016602)
        rates[81] = -states[81]*variables[326] + variables[325]
        #\dot{u_21} = c_124*\hat{u}_u_9 + \hat{u}_u_7*c_7 - 0.01140988469486*\hat{u}_u_21 + c_28*U_21 # -u_2(0.0775*v_21 + 0.62*(u_21 - 1)*(u_21 - 0.13)) - 0.01140988469486*u_u_21 + 0.0033558484396647*u_u_7 + 0.00805403625519528*u_u_9
        rates[82] = variables[124]*variables[175] + variables[167]*variables[7] - 0.01140988469486*variables[223] + variables[28]*variables[328]
        #\dot{v_21} = c_28*V_21 # -0.0775*v_21*(8.0*u_21*(u_21 - 1.13) + v_21)*(0.002*u_21 + 0.2*v_21 + 0.0006)/(u_21 + 0.3)
        rates[83] = variables[28]*variables[329]

    def compute_inputs(self,voi,inputs):
        t,states,variables=voi,self.states,self.variables
        #inputs size 84
        # forstate[0] = 0
        inputs[0] = 0
        # forstate[1] = 0
        inputs[1] = 0
        # forstate[2] = c_10*u_u_20 + c_113*u_u_4 + c_24*u_u_8 - 0.0105347817232959*u_u_1
        inputs[2] = variables[10]*variables[219] + variables[113]*variables[155] - 0.0105347817232959*variables[143] + variables[171]*variables[24]
        # forstate[3] = 0
        inputs[3] = 0
        # forstate[4] = 0
        inputs[4] = 0
        # forstate[5] = 0
        inputs[5] = 0
        # forstate[6] = c_112*u_u_17 + c_31*u_u_10 + c_59*u_u_9 + c_7*u_u_14 + c_84*u_u_20 - 0.0187547835509272*u_u_2
        inputs[6] = variables[112]*variables[207] - 0.0187547835509272*variables[147] + variables[175]*variables[59] + variables[179]*variables[31] + variables[195]*variables[7] + variables[219]*variables[84]
        # forstate[7] = 0
        inputs[7] = 0
        # forstate[8] = 0
        inputs[8] = 0
        # forstate[9] = 0
        inputs[9] = 0
        # forstate[10] = c_106*u_u_6 + c_107*u_u_20 + c_23*u_u_19 + c_50*u_u_15 - 0.00994487038649063*u_u_3
        inputs[10] = variables[106]*variables[163] + variables[107]*variables[219] - 0.00994487038649063*variables[151] + variables[199]*variables[50] + variables[215]*variables[23]
        # forstate[11] = 0
        inputs[11] = 0
        # forstate[12] = 0
        inputs[12] = 0
        # forstate[13] = 0
        inputs[13] = 0
        # forstate[14] = c_113*u_u_1 + c_53*u_u_15 + c_90*u_u_19 + 1.0*i_1_2 - 0.0100670263595698*u_u_4
        inputs[14] = 1.0*variables[0] + variables[113]*variables[143] - 0.0100670263595698*variables[155] + variables[199]*variables[53] + variables[215]*variables[90]
        # forstate[15] = 0
        inputs[15] = 0
        # forstate[16] = 0
        inputs[16] = 0
        # forstate[17] = 0
        inputs[17] = 0
        # forstate[18] = c_14*u_u_8 + c_58*u_u_11 + c_69*u_u_12 + c_7*u_u_18 + 1.0*i_1_2 - 0.00908557896398634*u_u_5
        inputs[18] = 1.0*variables[0] + variables[14]*variables[171] - 0.00908557896398634*variables[159] + variables[183]*variables[58] + variables[187]*variables[69] + variables[211]*variables[7]
        # forstate[19] = 0
        inputs[19] = 0
        # forstate[20] = 0
        inputs[20] = 0
        # forstate[21] = 0
        inputs[21] = 0
        # forstate[22] = c_106*u_u_3 + c_5*u_u_18 + c_52*u_u_8 - 0.0136866968793294*u_u_6
        inputs[22] = variables[106]*variables[151] - 0.0136866968793294*variables[163] + variables[171]*variables[52] + variables[211]*variables[5]
        # forstate[23] = 0
        inputs[23] = 0
        # forstate[24] = 0
        inputs[24] = 0
        # forstate[25] = 0
        inputs[25] = 0
        # forstate[26] = c_7*u_u_21 + c_92*u_u_15 - 0.00660632319563326*u_u_7
        inputs[26] = -0.00660632319563326*variables[167] + variables[199]*variables[92] + variables[223]*variables[7]
        # forstate[27] = 0
        inputs[27] = 0
        # forstate[28] = 0
        inputs[28] = 0
        # forstate[29] = 0
        inputs[29] = 0
        # forstate[30] = c_130*u_u_19 + c_14*u_u_5 + c_24*u_u_1 + c_36*u_u_13 + c_52*u_u_6 + c_80*u_u_16 - 0.0172670458146408*u_u_8
        inputs[30] = variables[130]*variables[215] + variables[143]*variables[24] + variables[14]*variables[159] + variables[163]*variables[52] - 0.0172670458146408*variables[171] + variables[191]*variables[36] + variables[203]*variables[80]
        # forstate[31] = 0
        inputs[31] = 0
        # forstate[32] = 0
        inputs[32] = 0
        # forstate[33] = 0
        inputs[33] = 0
        # forstate[34] = c_113*u_u_17 + c_124*u_u_21 + c_39*u_u_13 + c_59*u_u_2 + c_63*u_u_14 + c_74*u_u_15 + c_88*u_u_19 + c_99*u_u_11 - 0.0284904787482649*u_u_9
        inputs[34] = variables[113]*variables[207] + variables[124]*variables[223] + variables[147]*variables[59] - 0.0284904787482649*variables[175] + variables[183]*variables[99] + variables[191]*variables[39] + variables[195]*variables[63] + variables[199]*variables[74] + variables[215]*variables[88]
        # forstate[35] = 0
        inputs[35] = 0
        # forstate[36] = 0
        inputs[36] = 0
        # forstate[37] = 0
        inputs[37] = 0
        # forstate[38] = c_106*u_u_19 + c_31*u_u_2 + c_89*u_u_15 - 0.0151800152129758*u_u_10
        inputs[38] = variables[106]*variables[215] + variables[147]*variables[31] - 0.0151800152129758*variables[179] + variables[199]*variables[89]
        # forstate[39] = 0
        inputs[39] = 0
        # forstate[40] = 0
        inputs[40] = 0
        # forstate[41] = 0
        inputs[41] = 0
        # forstate[42] = c_130*u_u_17 + c_15*u_u_19 + c_27*u_u_12 + c_50*u_u_13 + c_58*u_u_5 + c_69*u_u_15 + c_99*u_u_9 + 1.0*i_1_2 - 0.0209989121379338*u_u_11
        inputs[42] = 1.0*variables[0] + variables[130]*variables[207] + variables[159]*variables[58] + variables[15]*variables[215] + variables[175]*variables[99] - 0.0209989121379338*variables[183] + variables[187]*variables[27] + variables[191]*variables[50] + variables[199]*variables[69]
        # forstate[43] = 0
        inputs[43] = 0
        # forstate[44] = 0
        inputs[44] = 0
        # forstate[45] = 0
        inputs[45] = 0
        # forstate[46] = c_27*u_u_11 + c_63*u_u_16 + c_69*u_u_5 + c_90*u_u_13 - 0.0118998507056144*u_u_12
        inputs[46] = variables[159]*variables[69] + variables[183]*variables[27] - 0.0118998507056144*variables[187] + variables[191]*variables[90] + variables[203]*variables[63]
        # forstate[47] = 0
        inputs[47] = 0
        # forstate[48] = 0
        inputs[48] = 0
        # forstate[49] = 0
        inputs[49] = 0
        # forstate[50] = c_117*u_u_19 + c_36*u_u_8 + c_39*u_u_9 + c_48*u_u_20 + c_50*u_u_11 + c_83*u_u_16 + c_90*u_u_12 - 0.022188359773101*u_u_13
        inputs[50] = variables[117]*variables[215] + variables[171]*variables[36] + variables[175]*variables[39] + variables[183]*variables[50] + variables[187]*variables[90] - 0.022188359773101*variables[191] + variables[203]*variables[83] + variables[219]*variables[48]
        # forstate[51] = 0
        inputs[51] = 0
        # forstate[52] = 0
        inputs[52] = 0
        # forstate[53] = 0
        inputs[53] = 0
        # forstate[54] = c_44*u_u_17 + c_49*u_u_20 + c_63*u_u_9 + c_7*u_u_2 - 0.0128788020257737*u_u_14
        inputs[54] = variables[147]*variables[7] + variables[175]*variables[63] - 0.0128788020257737*variables[195] + variables[207]*variables[44] + variables[219]*variables[49]
        # forstate[55] = 0
        inputs[55] = 0
        # forstate[56] = 0
        inputs[56] = 0
        # forstate[57] = 0
        inputs[57] = 0
        # forstate[58] = c_100*u_u_18 + c_5*u_u_19 + c_50*u_u_3 + c_53*u_u_4 + c_69*u_u_11 + c_74*u_u_9 + c_89*u_u_10 + c_92*u_u_7 - 0.0214542209029461*u_u_15
        inputs[58] = variables[100]*variables[211] + variables[151]*variables[50] + variables[155]*variables[53] + variables[167]*variables[92] + variables[175]*variables[74] + variables[179]*variables[89] + variables[183]*variables[69] - 0.0214542209029461*variables[199] + variables[215]*variables[5]
        # forstate[59] = 0
        inputs[59] = 0
        # forstate[60] = 0
        inputs[60] = 0
        # forstate[61] = 0
        inputs[61] = 0
        # forstate[62] = c_119*u_u_20 + c_35*u_u_17 + c_63*u_u_12 + c_80*u_u_8 + c_83*u_u_13 - 0.0218062245551532*u_u_16
        inputs[62] = variables[119]*variables[219] + variables[171]*variables[80] + variables[187]*variables[63] + variables[191]*variables[83] - 0.0218062245551532*variables[203] + variables[207]*variables[35]
        # forstate[63] = 0
        inputs[63] = 0
        # forstate[64] = 0
        inputs[64] = 0
        # forstate[65] = 0
        inputs[65] = 0
        # forstate[66] = c_112*u_u_2 + c_113*u_u_9 + c_130*u_u_11 + c_35*u_u_16 + c_44*u_u_14 + c_94*u_u_20 + 1.0*i_1_2 - 0.0303833148058918*u_u_17
        inputs[66] = 1.0*variables[0] + variables[112]*variables[147] + variables[113]*variables[175] + variables[130]*variables[183] + variables[195]*variables[44] + variables[203]*variables[35] - 0.0303833148058918*variables[207] + variables[219]*variables[94]
        # forstate[67] = 0
        inputs[67] = 0
        # forstate[68] = 0
        inputs[68] = 0
        # forstate[69] = 0
        inputs[69] = 0
        # forstate[70] = c_100*u_u_15 + c_114*u_u_19 + c_5*u_u_6 + c_7*u_u_5 - 0.0160249351480499*u_u_18
        inputs[70] = variables[100]*variables[199] + variables[114]*variables[215] + variables[159]*variables[7] + variables[163]*variables[5] - 0.0160249351480499*variables[211]
        # forstate[71] = 0
        inputs[71] = 0
        # forstate[72] = 0
        inputs[72] = 0
        # forstate[73] = 0
        inputs[73] = 0
        # forstate[74] = c_106*u_u_10 + c_108*u_u_20 + c_114*u_u_18 + c_117*u_u_13 + c_130*u_u_8 + c_15*u_u_11 + c_23*u_u_3 + c_5*u_u_15 + c_88*u_u_9 + c_90*u_u_4 - 0.0272967553522912*u_u_19
        inputs[74] = variables[106]*variables[179] + variables[108]*variables[219] + variables[114]*variables[211] + variables[117]*variables[191] + variables[130]*variables[171] + variables[151]*variables[23] + variables[155]*variables[90] + variables[15]*variables[183] + variables[175]*variables[88] + variables[199]*variables[5] - 0.0272967553522912*variables[215]
        # forstate[75] = 0
        inputs[75] = 0
        # forstate[76] = 0
        inputs[76] = 0
        # forstate[77] = 0
        inputs[77] = 0
        # forstate[78] = c_10*u_u_1 + c_107*u_u_3 + c_108*u_u_19 + c_119*u_u_16 + c_48*u_u_13 + c_49*u_u_14 + c_84*u_u_2 + c_94*u_u_17 - 0.0226091474207625*u_u_20
        inputs[78] = variables[107]*variables[151] + variables[108]*variables[215] + variables[10]*variables[143] + variables[119]*variables[203] + variables[147]*variables[84] + variables[191]*variables[48] + variables[195]*variables[49] + variables[207]*variables[94] - 0.0226091474207625*variables[219]
        # forstate[79] = 0
        inputs[79] = 0
        # forstate[80] = 0
        inputs[80] = 0
        # forstate[81] = 0
        inputs[81] = 0
        # forstate[82] = c_124*u_u_9 + c_7*u_u_7 - 0.01140988469486*u_u_21
        inputs[82] = variables[124]*variables[175] + variables[167]*variables[7] - 0.01140988469486*variables[223]
        # forstate[83] = 0
        inputs[83] = 0

    def compute_hamiltonian(self,cellHam):
        t,states,variables=self.time,self.states,self.variables
        #cellHam = np.zeros(21)
        cellHam[0] = states[0]**2*variables[225]/2 - 0.016602*states[1]*variables[227] - 0.00533333333333333*states[2]**3 + 0.00904*states[2]**2 - states[3]*(-0.0105347817232959*variables[143] + 0.0034875*variables[155] + 0.00570494234742999*variables[171] + 0.00134233937586588*variables[219])
        cellHam[1] = states[4]**2*variables[230]/2 - 0.016602*states[5]*variables[232] - 0.00533333333333333*states[6]**3 + 0.00904*states[6]**2 - states[7]*(-0.0187547835509272*variables[147] + 0.00178978583448784*variables[175] + 0.00553238341512669*variables[179] + 0.00335584843966471*variables[195] + 0.00624309617371499*variables[207] + 0.00183366968793294*variables[219])
        cellHam[2] = -0.00533333333333333*states[10]**3 + 0.00904*states[10]**2 - states[11]*(-0.00994487038649063*variables[151] + 0.00499763179784911*variables[163] + 0.000266912464258686*variables[199] + 0.00233123221661754*variables[215] + 0.00234909390776529*variables[219]) + states[8]**2*variables[235]/2 - 0.016602*states[9]*variables[237]
        cellHam[3] = states[12]**2*variables[240]/2 - 0.016602*states[13]*variables[242] - 0.00533333333333333*states[14]**3 + 0.00904*states[14]**2 - states[15]*(1.0*variables[0] + 0.0034875*variables[143] - 0.0100670263595698*variables[155] + 0.00225306424965365*variables[199] + 0.00432646210991617*variables[215])
        cellHam[4] = states[16]**2*variables[245]/2 - 0.016602*states[17]*variables[247] - 0.00533333333333333*states[18]**3 + 0.00904*states[18]**2 - states[19]*(1.0*variables[0] - 0.00908557896398634*variables[159] + 0.00197736820215088*variables[171] + 0.00181486232217075*variables[183] + 0.0019375*variables[187] + 0.0033558484396647*variables[211])
        cellHam[5] = states[20]**2*variables[250]/2 - 0.016602*states[21]*variables[252] - 0.00533333333333333*states[22]**3 + 0.00904*states[22]**2 - states[23]*(0.00499763179784911*variables[151] - 0.0136866968793294*variables[163] + 0.00264853789008382*variables[171] + 0.00604052719139646*variables[211])
        cellHam[6] = states[24]**2*variables[255]/2 - 0.016602*states[25]*variables[257] - 0.00533333333333333*states[26]**3 + 0.00904*states[26]**2 - states[27]*(-0.00660632319563326*variables[167] + 0.00325047475596855*variables[199] + 0.0033558484396647*variables[223])
        cellHam[7] = states[28]**2*variables[260]/2 - 0.016602*states[29]*variables[262] - 0.00533333333333333*states[30]**3 + 0.00904*states[30]**2 - states[31]*(0.00570494234742999*variables[143] + 0.00197736820215088*variables[159] + 0.00264853789008382*variables[163] - 0.0172670458146408*variables[171] + 0.00233340673210136*variables[191] + 0.00331131228061603*variables[203] + 0.00129147836225876*variables[215])
        cellHam[8] = states[32]**2*variables[265]/2 - 0.016602*states[33]*variables[267] - 0.00533333333333333*states[34]**3 + 0.00904*states[34]**2 - states[35]*(0.00178978583448784*variables[147] - 0.0284904787482649*variables[175] + 0.0110020181275976*variables[183] + 0.00177666289008382*variables[191] + 0.00100675453189941*variables[195] + 2.3604678562608e-5*variables[199] + 0.0034875*variables[207] + 0.00135011643043833*variables[215] + 0.00805403625519528*variables[223])
        cellHam[9] = states[36]**2*variables[270]/2 - 0.016602*states[37]*variables[272] - 0.00533333333333333*states[38]**3 + 0.00904*states[38]**2 - states[39]*(0.00553238341512669*variables[147] - 0.0151800152129758*variables[179] + 0.00465*variables[199] + 0.00499763179784911*variables[215])
        cellHam[10] = states[40]**2*variables[275]/2 - 0.016602*states[41]*variables[277] - 0.00533333333333333*states[42]**3 + 0.00904*states[42]**2 - states[43]*(1.0*variables[0] + 0.00181486232217075*variables[159] + 0.0110020181275976*variables[175] - 0.0209989121379338*variables[183] + 0.00462913406379882*variables[187] + 0.000266912464258687*variables[191] + 0.0019375*variables[199] + 0.00129147836225876*variables[207] + 5.70067978491158e-5*variables[215])
        cellHam[11] = states[44]**2*variables[280]/2 - 0.016602*states[45]*variables[282] - 0.00533333333333333*states[46]**3 + 0.00904*states[46]**2 - states[47]*(0.0019375*variables[159] + 0.00462913406379882*variables[183] - 0.0118998507056144*variables[187] + 0.00432646210991617*variables[191] + 0.00100675453189941*variables[203])
        cellHam[12] = states[48]**2*variables[285]/2 - 0.016602*states[49]*variables[287] - 0.00533333333333333*states[50]**3 + 0.00904*states[50]**2 - states[51]*(0.00233340673210136*variables[171] + 0.00177666289008382*variables[175] + 0.000266912464258687*variables[183] + 0.00432646210991617*variables[187] - 0.022188359773101*variables[191] + 0.00603034390776529*variables[203] + 0.000611223229310978*variables[215] + 0.0068433484396647*variables[219])
        cellHam[13] = states[52]**2*variables[290]/2 - 0.016602*states[53]*variables[292] - 0.00533333333333333*states[54]**3 + 0.00904*states[54]**2 - states[55]*(0.00335584843966471*variables[147] + 0.00100675453189941*variables[175] - 0.0128788020257737*variables[195] + 0.00812293911287484*variables[207] + 0.000393259941334723*variables[219])
        cellHam[14] = states[56]**2*variables[295]/2 - 0.016602*states[57]*variables[297] - 0.00533333333333333*states[58]**3 + 0.00904*states[58]**2 - states[59]*(0.000266912464258686*variables[151] + 0.00225306424965365*variables[155] + 0.00325047475596855*variables[167] + 2.3604678562608e-5*variables[175] + 0.00465*variables[179] + 0.0019375*variables[183] - 0.0214542209029461*variables[199] + 0.00303213756310611*variables[211] + 0.00604052719139646*variables[215])
        cellHam[15] = states[60]**2*variables[300]/2 - 0.016602*states[61]*variables[302] - 0.00533333333333333*states[62]**3 + 0.00904*states[62]**2 - states[63]*(0.00331131228061603*variables[171] + 0.00100675453189941*variables[187] + 0.00603034390776529*variables[191] - 0.0218062245551532*variables[203] + 0.00777166709324443*variables[207] + 0.00368614674162807*variables[219])
        cellHam[16] = states[64]**2*variables[305]/2 - 0.016602*states[65]*variables[307] - 0.00533333333333333*states[66]**3 + 0.00904*states[66]**2 - states[67]*(1.0*variables[0] + 0.00624309617371499*variables[147] + 0.0034875*variables[175] + 0.00129147836225876*variables[183] + 0.00812293911287484*variables[195] + 0.00777166709324443*variables[203] - 0.0303833148058918*variables[207] + 0.00346663406379882*variables[219])
        cellHam[17] = states[68]**2*variables[310]/2 - 0.016602*states[69]*variables[312] - 0.00533333333333333*states[70]**3 + 0.00904*states[70]**2 - states[71]*(0.0033558484396647*variables[159] + 0.00604052719139646*variables[163] + 0.00303213756310611*variables[199] - 0.0160249351480499*variables[211] + 0.00359642195388264*variables[215])
        cellHam[18] = states[72]**2*variables[315]/2 - 0.016602*states[73]*variables[317] - 0.00533333333333333*states[74]**3 + 0.00904*states[74]**2 - states[75]*(0.00233123221661754*variables[151] + 0.00432646210991617*variables[155] + 0.00129147836225876*variables[171] + 0.00135011643043833*variables[175] + 0.00499763179784911*variables[179] + 5.70067978491158e-5*variables[183] + 0.000611223229310978*variables[191] + 0.00604052719139646*variables[199] + 0.00359642195388264*variables[211] - 0.0272967553522912*variables[215] + 0.00269465526277211*variables[219])
        cellHam[19] = states[76]**2*variables[320]/2 - 0.016602*states[77]*variables[322] - 0.00533333333333333*states[78]**3 + 0.00904*states[78]**2 - states[79]*(0.00134233937586588*variables[143] + 0.00183366968793294*variables[147] + 0.00234909390776529*variables[151] + 0.0068433484396647*variables[191] + 0.000393259941334723*variables[195] + 0.00368614674162807*variables[203] + 0.00346663406379882*variables[207] + 0.00269465526277211*variables[215] - 0.0226091474207625*variables[219])
        cellHam[20] = states[80]**2*variables[325]/2 - 0.016602*states[81]*variables[327] - 0.00533333333333333*states[82]**3 + 0.00904*states[82]**2 - states[83]*(0.0033558484396647*variables[167] + 0.00805403625519528*variables[175] - 0.01140988469486*variables[223])

        return cellHam

    def compute_external_energy(self,inputEnergy):
        t,states,variables=self.time,self.states,self.variables
        #inputEnergy = np.zeros(21)
        inputEnergy[0] = 0
        inputEnergy[1] = 0
        inputEnergy[2] = 0
        inputEnergy[3] = -states[15]*variables[0]
        inputEnergy[4] = -states[19]*variables[0]
        inputEnergy[5] = 0
        inputEnergy[6] = 0
        inputEnergy[7] = 0
        inputEnergy[8] = 0
        inputEnergy[9] = 0
        inputEnergy[10] = -states[43]*variables[0]
        inputEnergy[11] = 0
        inputEnergy[12] = 0
        inputEnergy[13] = 0
        inputEnergy[14] = 0
        inputEnergy[15] = 0
        inputEnergy[16] = -states[67]*variables[0]
        inputEnergy[17] = 0
        inputEnergy[18] = 0
        inputEnergy[19] = 0
        inputEnergy[20] = 0

        return inputEnergy

    def compute_total_input_energy(self,totalInputEnergy):
        t,states,variables=self.time,self.states,self.variables
        #totalInputEnergy = np.zeros(21)
        totalInputEnergy[0] = states[3]*(0.0105347817232959*variables[143] - 0.0034875*variables[155] - 0.00570494234742999*variables[171] - 0.00134233937586588*variables[219])
        totalInputEnergy[1] = states[7]*(0.0187547835509272*variables[147] - 0.00178978583448784*variables[175] - 0.00553238341512669*variables[179] - 0.00335584843966471*variables[195] - 0.00624309617371499*variables[207] - 0.00183366968793294*variables[219])
        totalInputEnergy[2] = states[11]*(0.00994487038649063*variables[151] - 0.00499763179784911*variables[163] - 0.000266912464258686*variables[199] - 0.00233123221661754*variables[215] - 0.00234909390776529*variables[219])
        totalInputEnergy[3] = states[15]*(-variables[0] - 0.0034875*variables[143] + 0.0100670263595698*variables[155] - 0.00225306424965365*variables[199] - 0.00432646210991617*variables[215])
        totalInputEnergy[4] = states[19]*(-variables[0] + 0.00908557896398634*variables[159] - 0.00197736820215088*variables[171] - 0.00181486232217075*variables[183] - 0.0019375*variables[187] - 0.0033558484396647*variables[211])
        totalInputEnergy[5] = states[23]*(-0.00499763179784911*variables[151] + 0.0136866968793294*variables[163] - 0.00264853789008382*variables[171] - 0.00604052719139646*variables[211])
        totalInputEnergy[6] = states[27]*(0.00660632319563326*variables[167] - 0.00325047475596855*variables[199] - 0.0033558484396647*variables[223])
        totalInputEnergy[7] = states[31]*(-0.00570494234742999*variables[143] - 0.00197736820215088*variables[159] - 0.00264853789008382*variables[163] + 0.0172670458146408*variables[171] - 0.00233340673210136*variables[191] - 0.00331131228061603*variables[203] - 0.00129147836225876*variables[215])
        totalInputEnergy[8] = states[35]*(-0.00178978583448784*variables[147] + 0.0284904787482649*variables[175] - 0.0110020181275976*variables[183] - 0.00177666289008382*variables[191] - 0.00100675453189941*variables[195] - 2.3604678562608e-5*variables[199] - 0.0034875*variables[207] - 0.00135011643043833*variables[215] - 0.00805403625519528*variables[223])
        totalInputEnergy[9] = states[39]*(-0.00553238341512669*variables[147] + 0.0151800152129758*variables[179] - 0.00465*variables[199] - 0.00499763179784911*variables[215])
        totalInputEnergy[10] = states[43]*(-variables[0] - 0.00181486232217075*variables[159] - 0.0110020181275976*variables[175] + 0.0209989121379338*variables[183] - 0.00462913406379882*variables[187] - 0.000266912464258687*variables[191] - 0.0019375*variables[199] - 0.00129147836225876*variables[207] - 5.70067978491158e-5*variables[215])
        totalInputEnergy[11] = states[47]*(-0.0019375*variables[159] - 0.00462913406379882*variables[183] + 0.0118998507056144*variables[187] - 0.00432646210991617*variables[191] - 0.00100675453189941*variables[203])
        totalInputEnergy[12] = states[51]*(-0.00233340673210136*variables[171] - 0.00177666289008382*variables[175] - 0.000266912464258687*variables[183] - 0.00432646210991617*variables[187] + 0.022188359773101*variables[191] - 0.00603034390776529*variables[203] - 0.000611223229310978*variables[215] - 0.0068433484396647*variables[219])
        totalInputEnergy[13] = states[55]*(-0.00335584843966471*variables[147] - 0.00100675453189941*variables[175] + 0.0128788020257737*variables[195] - 0.00812293911287484*variables[207] - 0.000393259941334723*variables[219])
        totalInputEnergy[14] = states[59]*(-0.000266912464258686*variables[151] - 0.00225306424965365*variables[155] - 0.00325047475596855*variables[167] - 2.3604678562608e-5*variables[175] - 0.00465*variables[179] - 0.0019375*variables[183] + 0.0214542209029461*variables[199] - 0.00303213756310611*variables[211] - 0.00604052719139646*variables[215])
        totalInputEnergy[15] = states[63]*(-0.00331131228061603*variables[171] - 0.00100675453189941*variables[187] - 0.00603034390776529*variables[191] + 0.0218062245551532*variables[203] - 0.00777166709324443*variables[207] - 0.00368614674162807*variables[219])
        totalInputEnergy[16] = states[67]*(-variables[0] - 0.00624309617371499*variables[147] - 0.0034875*variables[175] - 0.00129147836225876*variables[183] - 0.00812293911287484*variables[195] - 0.00777166709324443*variables[203] + 0.0303833148058918*variables[207] - 0.00346663406379882*variables[219])
        totalInputEnergy[17] = states[71]*(-0.0033558484396647*variables[159] - 0.00604052719139646*variables[163] - 0.00303213756310611*variables[199] + 0.0160249351480499*variables[211] - 0.00359642195388264*variables[215])
        totalInputEnergy[18] = states[75]*(-0.00233123221661754*variables[151] - 0.00432646210991617*variables[155] - 0.00129147836225876*variables[171] - 0.00135011643043833*variables[175] - 0.00499763179784911*variables[179] - 5.70067978491158e-5*variables[183] - 0.000611223229310978*variables[191] - 0.00604052719139646*variables[199] - 0.00359642195388264*variables[211] + 0.0272967553522912*variables[215] - 0.00269465526277211*variables[219])
        totalInputEnergy[19] = states[79]*(-0.00134233937586588*variables[143] - 0.00183366968793294*variables[147] - 0.00234909390776529*variables[151] - 0.0068433484396647*variables[191] - 0.000393259941334723*variables[195] - 0.00368614674162807*variables[203] - 0.00346663406379882*variables[207] - 0.00269465526277211*variables[215] + 0.0226091474207625*variables[219])
        totalInputEnergy[20] = states[83]*(-0.0033558484396647*variables[167] - 0.00805403625519528*variables[175] + 0.01140988469486*variables[223])

        return totalInputEnergy

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
        #	i_1_2 -> variables[0]

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
        #	i_1_2 -> variables[0]

        #Comment the line below (and uncomment the line after) to solve the model without event processing!    
        #raise("Process events not implemented")

    def getStateValues(self,statename):
        return self.states[self.stateIndexes[statename]]

    def setStateValues(self,statename,values):
        self.states[self.stateIndexes[statename]] = values

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
    grid = plt.GridSpec(7, 3, wspace=0.2, hspace=0.5)

    ix = 0
    for i in range(7):
        for j in range(3):
            ax = plt.subplot(grid[i, j])
            ax.plot(result[ix,:])
            ax.title.set_text(f'{ix//4+1}')
            ix += 4
            if ix+4 > result.shape[0]:
                break
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.3)
    fig.savefig(f"FTUStepper_results.png",dpi=300)
    plt.show()         

