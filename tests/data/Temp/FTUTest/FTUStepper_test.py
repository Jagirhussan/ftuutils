import time
import numpy as np
from numpy import exp
from scipy.integrate import ode

def Heaviside(x):
    if x > 0:
        return 1.0
    return 0.0
__version__ = '0.0.1'

class FTUStepper:
    STATE_COUNT = 84
    VARIABLE_COUNT = 330
    CELL_COUNT = 21
    stateIndexes = {'Tai': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80], 'Ta': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81], 'u': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82], 'v': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83]}

    def __init__(self):
        self.states = np.zeros(self.STATE_COUNT)
        self.rates = np.zeros(self.STATE_COUNT)
        self.variables = np.zeros(self.VARIABLE_COUNT)
        self.time = 0.0
        self.odeintegrator = ode(lambda t, x: self.rhs(t, x))
        self.odeintegrator.set_integrator('vode', method='bdf', atol=1e-06, rtol=1e-06, max_step=1)
        self.odeintegrator.set_initial_value(self.states, self.time)
        (states, variables) = (self.states, self.variables)
        states[0] = 0.0
        states[1] = 0.001
        states[2] = 0.0
        states[3] = 0.03604
        states[4] = 0.0
        states[5] = 0.001
        states[6] = 0.0
        states[7] = 0.03604
        states[8] = 0.0
        states[9] = 0.001
        states[10] = 0.0
        states[11] = 0.03604
        states[12] = 0.0
        states[13] = 0.001
        states[14] = 0.0
        states[15] = 0.03604
        states[16] = 0.0
        states[17] = 0.001
        states[18] = 0.0
        states[19] = 0.03604
        states[20] = 0.0
        states[21] = 0.001
        states[22] = 0.0
        states[23] = 0.03604
        states[24] = 0.0
        states[25] = 0.001
        states[26] = 0.0
        states[27] = 0.03604
        states[28] = 0.0
        states[29] = 0.001
        states[30] = 0.0
        states[31] = 0.03604
        states[32] = 0.0
        states[33] = 0.001
        states[34] = 0.0
        states[35] = 0.03604
        states[36] = 0.0
        states[37] = 0.001
        states[38] = 0.0
        states[39] = 0.03604
        states[40] = 0.0
        states[41] = 0.001
        states[42] = 0.0
        states[43] = 0.03604
        states[44] = 0.0
        states[45] = 0.001
        states[46] = 0.0
        states[47] = 0.03604
        states[48] = 0.0
        states[49] = 0.001
        states[50] = 0.0
        states[51] = 0.03604
        states[52] = 0.0
        states[53] = 0.001
        states[54] = 0.0
        states[55] = 0.03604
        states[56] = 0.0
        states[57] = 0.001
        states[58] = 0.0
        states[59] = 0.03604
        states[60] = 0.0
        states[61] = 0.001
        states[62] = 0.0
        states[63] = 0.03604
        states[64] = 0.0
        states[65] = 0.001
        states[66] = 0.0
        states[67] = 0.03604
        states[68] = 0.0
        states[69] = 0.001
        states[70] = 0.0
        states[71] = 0.03604
        states[72] = 0.0
        states[73] = 0.001
        states[74] = 0.0
        states[75] = 0.03604
        states[76] = 0.0
        states[77] = 0.001
        states[78] = 0.0
        states[79] = 0.03604
        states[80] = 0.0
        states[81] = 0.001
        states[82] = 0.0
        states[83] = 0.03604
        variables[1] = 0.0119
        variables[2] = 0.195871
        variables[3] = 0.022188
        variables[4] = 0.017321
        variables[5] = 0.006041
        variables[6] = 0.029072
        variables[7] = 0.003356
        variables[8] = 0.01299
        variables[9] = 0.021454
        variables[10] = 0.001342
        variables[11] = 0.013687
        variables[12] = 0.003444
        variables[13] = 0.166178
        variables[14] = 0.001977
        variables[15] = 5.7e-05
        variables[16] = 0.352216
        variables[17] = 0.055825
        variables[18] = 0.073612
        variables[19] = 0.270954
        variables[20] = 0.206773
        variables[21] = 0.016664
        variables[22] = 0.041942
        variables[23] = 0.002331
        variables[24] = 0.005705
        variables[25] = 0.064486
        variables[26] = 0.147224
        variables[27] = 0.004629
        variables[28] = 0.0775
        variables[29] = 0.005074
        variables[30] = 0.276829
        variables[31] = 0.005532
        variables[32] = 0.045
        variables[33] = 0.016602
        variables[34] = 0.291731
        variables[35] = 0.007772
        variables[36] = 0.002333
        variables[37] = 0.077811
        variables[38] = 0.071386
        variables[39] = 0.001777
        variables[40] = 0.043301
        variables[41] = 0.141962
        variables[42] = 0.039124
        variables[43] = 0.085243
        variables[44] = 0.008123
        variables[45] = 0.077942
        variables[46] = 0.286301
        variables[47] = 0.007887
        variables[48] = 0.006843
        variables[49] = 0.000393
        variables[50] = 0.000267
        variables[51] = 0.010535
        variables[52] = 0.002649
        variables[53] = 0.002253
        variables[54] = 0.023094
        variables[55] = 0.017267
        variables[56] = 0.030311
        variables[57] = 0.06
        variables[58] = 0.001815
        variables[59] = 0.00179
        variables[60] = 0.02366
        variables[61] = 0.367619
        variables[62] = 0.000305
        variables[63] = 0.001007
        variables[64] = 0.017421
        variables[65] = 0.042727
        variables[66] = 0.153546
        variables[67] = 0.088301
        variables[68] = 0.010067
        variables[69] = 0.001938
        variables[70] = 0.176603
        variables[71] = 0.006606
        variables[72] = 0.018755
        variables[73] = 0.047563
        variables[74] = 2.4e-05
        variables[75] = 0.281371
        variables[76] = 0.030383
        variables[77] = 0.01141
        variables[78] = 0.022925
        variables[79] = 0.128321
        variables[80] = 0.003311
        variables[81] = 0.030108
        variables[82] = 0.046405
        variables[83] = 0.00603
        variables[84] = 0.001834
        variables[85] = 0.020999
        variables[86] = 0.104812
        variables[87] = 0.016025
        variables[88] = 0.00135
        variables[89] = 0.00465
        variables[90] = 0.004326
        variables[91] = 0.10028
        variables[92] = 0.00325
        variables[93] = 0.009086
        variables[94] = 0.003467
        variables[95] = 0.080556
        variables[96] = 0.392043
        variables[97] = 0.117233
        variables[98] = 0.025514
        variables[99] = 0.011002
        variables[100] = 0.003032
        variables[101] = 0.135933
        variables[102] = 0.129897
        variables[103] = 0.000736
        variables[104] = 0.009945
        variables[105] = 0.241997
        variables[106] = 0.004998
        variables[107] = 0.002349
        variables[108] = 0.002695
        variables[109] = 0.02849
        variables[110] = 0.021806
        variables[111] = 0.059731
        variables[112] = 0.006243
        variables[113] = 0.003487
        variables[114] = 0.003596
        variables[115] = 0.023418
        variables[116] = 0.044731
        variables[117] = 0.000611
        variables[118] = 0.03008
        variables[119] = 0.003686
        variables[120] = 0.025
        variables[121] = 0.222801
        variables[122] = 0.034175
        variables[123] = 0.03477
        variables[124] = 0.008054
        variables[125] = 0.027297
        variables[126] = 0.022609
        variables[127] = 0.01518
        variables[128] = 0.012879
        variables[129] = 0.103923
        variables[130] = 0.001291
        variables[131] = 0.0001
        variables[132] = 0.78
        variables[133] = 0.2925
        variables[134] = 0.042969
        variables[135] = 2.506575
        variables[136] = 8.0
        variables[137] = 0.13
        variables[138] = 0.2
        variables[139] = 0.002
        variables[140] = 0.3

    def compute_variables(self, voi):
        t = voi
        (states, rates, variables) = (self.states, self.rates, self.variables)
        variables[141] = states[0]
        variables[142] = states[1]
        variables[143] = states[2]
        variables[144] = states[3]
        variables[145] = states[4]
        variables[146] = states[5]
        variables[147] = states[6]
        variables[148] = states[7]
        variables[149] = states[8]
        variables[150] = states[9]
        variables[151] = states[10]
        variables[152] = states[11]
        variables[153] = states[12]
        variables[154] = states[13]
        variables[155] = states[14]
        variables[156] = states[15]
        variables[157] = states[16]
        variables[158] = states[17]
        variables[159] = states[18]
        variables[160] = states[19]
        variables[161] = states[20]
        variables[162] = states[21]
        variables[163] = states[22]
        variables[164] = states[23]
        variables[165] = states[24]
        variables[166] = states[25]
        variables[167] = states[26]
        variables[168] = states[27]
        variables[169] = states[28]
        variables[170] = states[29]
        variables[171] = states[30]
        variables[172] = states[31]
        variables[173] = states[32]
        variables[174] = states[33]
        variables[175] = states[34]
        variables[176] = states[35]
        variables[177] = states[36]
        variables[178] = states[37]
        variables[179] = states[38]
        variables[180] = states[39]
        variables[181] = states[40]
        variables[182] = states[41]
        variables[183] = states[42]
        variables[184] = states[43]
        variables[185] = states[44]
        variables[186] = states[45]
        variables[187] = states[46]
        variables[188] = states[47]
        variables[189] = states[48]
        variables[190] = states[49]
        variables[191] = states[50]
        variables[192] = states[51]
        variables[193] = states[52]
        variables[194] = states[53]
        variables[195] = states[54]
        variables[196] = states[55]
        variables[197] = states[56]
        variables[198] = states[57]
        variables[199] = states[58]
        variables[200] = states[59]
        variables[201] = states[60]
        variables[202] = states[61]
        variables[203] = states[62]
        variables[204] = states[63]
        variables[205] = states[64]
        variables[206] = states[65]
        variables[207] = states[66]
        variables[208] = states[67]
        variables[209] = states[68]
        variables[210] = states[69]
        variables[211] = states[70]
        variables[212] = states[71]
        variables[213] = states[72]
        variables[214] = states[73]
        variables[215] = states[74]
        variables[216] = states[75]
        variables[217] = states[76]
        variables[218] = states[77]
        variables[219] = states[78]
        variables[220] = states[79]
        variables[221] = states[80]
        variables[222] = states[81]
        variables[223] = states[82]
        variables[224] = states[83]

    def compute_rates(self, voi):
        t = voi
        (states, rates, variables) = (self.states, self.rates, self.variables)
        variables[225] = states[0] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[0] + variables[133]) * Heaviside(states[2] - variables[132]))
        variables[226] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[0] + variables[133]) * Heaviside(states[2] - variables[132])
        variables[227] = exp(-0.5 * (states[2] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[228] = -states[2] * states[3] + states[2] * variables[136] * (1 - states[2]) * (states[2] - variables[137])
        variables[229] = (-states[2] * variables[136] * (states[2] - variables[137] - 1) - states[3]) * (states[3] * variables[138] / (states[2] + variables[140]) + variables[139])
        variables[230] = states[4] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[4] + variables[133]) * Heaviside(states[6] - variables[132]))
        variables[231] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[4] + variables[133]) * Heaviside(states[6] - variables[132])
        variables[232] = exp(-0.5 * (states[6] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[233] = -states[6] * states[7] + states[6] * variables[136] * (1 - states[6]) * (states[6] - variables[137])
        variables[234] = (-states[6] * variables[136] * (states[6] - variables[137] - 1) - states[7]) * (states[7] * variables[138] / (states[6] + variables[140]) + variables[139])
        variables[235] = states[8] * (variables[33] + (variables[131] - variables[33]) * Heaviside(states[10] - variables[132]) * Heaviside(-states[8] + variables[133]))
        variables[236] = variables[33] + (variables[131] - variables[33]) * Heaviside(states[10] - variables[132]) * Heaviside(-states[8] + variables[133])
        variables[237] = exp(-0.5 * (states[10] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[238] = -states[10] * states[11] + states[10] * variables[136] * (1 - states[10]) * (states[10] - variables[137])
        variables[239] = (-states[10] * variables[136] * (states[10] - variables[137] - 1) - states[11]) * (states[11] * variables[138] / (states[10] + variables[140]) + variables[139])
        variables[240] = states[12] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[12] + variables[133]) * Heaviside(states[14] - variables[132]))
        variables[241] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[12] + variables[133]) * Heaviside(states[14] - variables[132])
        variables[242] = exp(-0.5 * (states[14] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[243] = -states[14] * states[15] + states[14] * variables[136] * (1 - states[14]) * (states[14] - variables[137])
        variables[244] = (-states[14] * variables[136] * (states[14] - variables[137] - 1) - states[15]) * (states[15] * variables[138] / (states[14] + variables[140]) + variables[139])
        variables[245] = states[16] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[16] + variables[133]) * Heaviside(states[18] - variables[132]))
        variables[246] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[16] + variables[133]) * Heaviside(states[18] - variables[132])
        variables[247] = exp(-0.5 * (states[18] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[248] = -states[18] * states[19] + states[18] * variables[136] * (1 - states[18]) * (states[18] - variables[137])
        variables[249] = (-states[18] * variables[136] * (states[18] - variables[137] - 1) - states[19]) * (states[19] * variables[138] / (states[18] + variables[140]) + variables[139])
        variables[250] = states[20] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[20] + variables[133]) * Heaviside(states[22] - variables[132]))
        variables[251] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[20] + variables[133]) * Heaviside(states[22] - variables[132])
        variables[252] = exp(-0.5 * (states[22] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[253] = -states[22] * states[23] + states[22] * variables[136] * (1 - states[22]) * (states[22] - variables[137])
        variables[254] = (-states[22] * variables[136] * (states[22] - variables[137] - 1) - states[23]) * (states[23] * variables[138] / (states[22] + variables[140]) + variables[139])
        variables[255] = states[24] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[24] + variables[133]) * Heaviside(states[26] - variables[132]))
        variables[256] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[24] + variables[133]) * Heaviside(states[26] - variables[132])
        variables[257] = exp(-0.5 * (states[26] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[258] = -states[26] * states[27] + states[26] * variables[136] * (1 - states[26]) * (states[26] - variables[137])
        variables[259] = (-states[26] * variables[136] * (states[26] - variables[137] - 1) - states[27]) * (states[27] * variables[138] / (states[26] + variables[140]) + variables[139])
        variables[260] = states[28] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[28] + variables[133]) * Heaviside(states[30] - variables[132]))
        variables[261] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[28] + variables[133]) * Heaviside(states[30] - variables[132])
        variables[262] = exp(-0.5 * (states[30] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[263] = -states[30] * states[31] + states[30] * variables[136] * (1 - states[30]) * (states[30] - variables[137])
        variables[264] = (-states[30] * variables[136] * (states[30] - variables[137] - 1) - states[31]) * (states[31] * variables[138] / (states[30] + variables[140]) + variables[139])
        variables[265] = states[32] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[32] + variables[133]) * Heaviside(states[34] - variables[132]))
        variables[266] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[32] + variables[133]) * Heaviside(states[34] - variables[132])
        variables[267] = exp(-0.5 * (states[34] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[268] = -states[34] * states[35] + states[34] * variables[136] * (1 - states[34]) * (states[34] - variables[137])
        variables[269] = (-states[34] * variables[136] * (states[34] - variables[137] - 1) - states[35]) * (states[35] * variables[138] / (states[34] + variables[140]) + variables[139])
        variables[270] = states[36] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[36] + variables[133]) * Heaviside(states[38] - variables[132]))
        variables[271] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[36] + variables[133]) * Heaviside(states[38] - variables[132])
        variables[272] = exp(-0.5 * (states[38] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[273] = -states[38] * states[39] + states[38] * variables[136] * (1 - states[38]) * (states[38] - variables[137])
        variables[274] = (-states[38] * variables[136] * (states[38] - variables[137] - 1) - states[39]) * (states[39] * variables[138] / (states[38] + variables[140]) + variables[139])
        variables[275] = states[40] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[40] + variables[133]) * Heaviside(states[42] - variables[132]))
        variables[276] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[40] + variables[133]) * Heaviside(states[42] - variables[132])
        variables[277] = exp(-0.5 * (states[42] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[278] = -states[42] * states[43] + states[42] * variables[136] * (1 - states[42]) * (states[42] - variables[137])
        variables[279] = (-states[42] * variables[136] * (states[42] - variables[137] - 1) - states[43]) * (states[43] * variables[138] / (states[42] + variables[140]) + variables[139])
        variables[280] = states[44] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[44] + variables[133]) * Heaviside(states[46] - variables[132]))
        variables[281] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[44] + variables[133]) * Heaviside(states[46] - variables[132])
        variables[282] = exp(-0.5 * (states[46] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[283] = -states[46] * states[47] + states[46] * variables[136] * (1 - states[46]) * (states[46] - variables[137])
        variables[284] = (-states[46] * variables[136] * (states[46] - variables[137] - 1) - states[47]) * (states[47] * variables[138] / (states[46] + variables[140]) + variables[139])
        variables[285] = states[48] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[48] + variables[133]) * Heaviside(states[50] - variables[132]))
        variables[286] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[48] + variables[133]) * Heaviside(states[50] - variables[132])
        variables[287] = exp(-0.5 * (states[50] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[288] = -states[50] * states[51] + states[50] * variables[136] * (1 - states[50]) * (states[50] - variables[137])
        variables[289] = (-states[50] * variables[136] * (states[50] - variables[137] - 1) - states[51]) * (states[51] * variables[138] / (states[50] + variables[140]) + variables[139])
        variables[290] = states[52] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[52] + variables[133]) * Heaviside(states[54] - variables[132]))
        variables[291] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[52] + variables[133]) * Heaviside(states[54] - variables[132])
        variables[292] = exp(-0.5 * (states[54] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[293] = -states[54] * states[55] + states[54] * variables[136] * (1 - states[54]) * (states[54] - variables[137])
        variables[294] = (-states[54] * variables[136] * (states[54] - variables[137] - 1) - states[55]) * (states[55] * variables[138] / (states[54] + variables[140]) + variables[139])
        variables[295] = states[56] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[56] + variables[133]) * Heaviside(states[58] - variables[132]))
        variables[296] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[56] + variables[133]) * Heaviside(states[58] - variables[132])
        variables[297] = exp(-0.5 * (states[58] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[298] = -states[58] * states[59] + states[58] * variables[136] * (1 - states[58]) * (states[58] - variables[137])
        variables[299] = (-states[58] * variables[136] * (states[58] - variables[137] - 1) - states[59]) * (states[59] * variables[138] / (states[58] + variables[140]) + variables[139])
        variables[300] = states[60] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[60] + variables[133]) * Heaviside(states[62] - variables[132]))
        variables[301] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[60] + variables[133]) * Heaviside(states[62] - variables[132])
        variables[302] = exp(-0.5 * (states[62] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[303] = -states[62] * states[63] + states[62] * variables[136] * (1 - states[62]) * (states[62] - variables[137])
        variables[304] = (-states[62] * variables[136] * (states[62] - variables[137] - 1) - states[63]) * (states[63] * variables[138] / (states[62] + variables[140]) + variables[139])
        variables[305] = states[64] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[64] + variables[133]) * Heaviside(states[66] - variables[132]))
        variables[306] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[64] + variables[133]) * Heaviside(states[66] - variables[132])
        variables[307] = exp(-0.5 * (states[66] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[308] = -states[66] * states[67] + states[66] * variables[136] * (1 - states[66]) * (states[66] - variables[137])
        variables[309] = (-states[66] * variables[136] * (states[66] - variables[137] - 1) - states[67]) * (states[67] * variables[138] / (states[66] + variables[140]) + variables[139])
        variables[310] = states[68] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[68] + variables[133]) * Heaviside(states[70] - variables[132]))
        variables[311] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[68] + variables[133]) * Heaviside(states[70] - variables[132])
        variables[312] = exp(-0.5 * (states[70] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[313] = -states[70] * states[71] + states[70] * variables[136] * (1 - states[70]) * (states[70] - variables[137])
        variables[314] = (-states[70] * variables[136] * (states[70] - variables[137] - 1) - states[71]) * (states[71] * variables[138] / (states[70] + variables[140]) + variables[139])
        variables[315] = states[72] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[72] + variables[133]) * Heaviside(states[74] - variables[132]))
        variables[316] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[72] + variables[133]) * Heaviside(states[74] - variables[132])
        variables[317] = exp(-0.5 * (states[74] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[318] = -states[74] * states[75] + states[74] * variables[136] * (1 - states[74]) * (states[74] - variables[137])
        variables[319] = (-states[74] * variables[136] * (states[74] - variables[137] - 1) - states[75]) * (states[75] * variables[138] / (states[74] + variables[140]) + variables[139])
        variables[320] = states[76] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[76] + variables[133]) * Heaviside(states[78] - variables[132]))
        variables[321] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[76] + variables[133]) * Heaviside(states[78] - variables[132])
        variables[322] = exp(-0.5 * (states[78] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[323] = -states[78] * states[79] + states[78] * variables[136] * (1 - states[78]) * (states[78] - variables[137])
        variables[324] = (-states[78] * variables[136] * (states[78] - variables[137] - 1) - states[79]) * (states[79] * variables[138] / (states[78] + variables[140]) + variables[139])
        variables[325] = states[80] * (variables[33] + (variables[131] - variables[33]) * Heaviside(-states[80] + variables[133]) * Heaviside(states[82] - variables[132]))
        variables[326] = variables[33] + (variables[131] - variables[33]) * Heaviside(-states[80] + variables[133]) * Heaviside(states[82] - variables[132])
        variables[327] = exp(-0.5 * (states[82] - 1) ** 2 / variables[134] ** 2) / (variables[134] * variables[135])
        variables[328] = -states[82] * states[83] + states[82] * variables[136] * (1 - states[82]) * (states[82] - variables[137])
        variables[329] = (-states[82] * variables[136] * (states[82] - variables[137] - 1) - states[83]) * (states[83] * variables[138] / (states[82] + variables[140]) + variables[139])
        rates[0] = -0.016602 * states[0] + variables[227] * variables[33]
        rates[1] = -states[1] * variables[226] + variables[225]
        rates[2] = variables[10] * variables[219] + variables[113] * variables[155] - 0.0105347817232959 * variables[143] + variables[171] * variables[24] + variables[228] * variables[28]
        rates[3] = variables[229] * variables[28]
        rates[4] = -0.016602 * states[4] + variables[232] * variables[33]
        rates[5] = -states[5] * variables[231] + variables[230]
        rates[6] = variables[112] * variables[207] - 0.0187547835509272 * variables[147] + variables[175] * variables[59] + variables[179] * variables[31] + variables[195] * variables[7] + variables[219] * variables[84] + variables[233] * variables[28]
        rates[7] = variables[234] * variables[28]
        rates[8] = -0.016602 * states[8] + variables[237] * variables[33]
        rates[9] = -states[9] * variables[236] + variables[235]
        rates[10] = variables[106] * variables[163] + variables[107] * variables[219] - 0.00994487038649063 * variables[151] + variables[199] * variables[50] + variables[215] * variables[23] + variables[238] * variables[28]
        rates[11] = variables[239] * variables[28]
        rates[12] = -0.016602 * states[12] + variables[242] * variables[33]
        rates[13] = -states[13] * variables[241] + variables[240]
        rates[14] = 1.0 * variables[0] + variables[113] * variables[143] - 0.0100670263595698 * variables[155] + variables[199] * variables[53] + variables[215] * variables[90] + variables[243] * variables[28]
        rates[15] = variables[244] * variables[28]
        rates[16] = -0.016602 * states[16] + variables[247] * variables[33]
        rates[17] = -states[17] * variables[246] + variables[245]
        rates[18] = 1.0 * variables[0] + variables[14] * variables[171] - 0.00908557896398634 * variables[159] + variables[183] * variables[58] + variables[187] * variables[69] + variables[211] * variables[7] + variables[248] * variables[28]
        rates[19] = variables[249] * variables[28]
        rates[20] = -0.016602 * states[20] + variables[252] * variables[33]
        rates[21] = -states[21] * variables[251] + variables[250]
        rates[22] = variables[106] * variables[151] - 0.0136866968793294 * variables[163] + variables[171] * variables[52] + variables[211] * variables[5] + variables[253] * variables[28]
        rates[23] = variables[254] * variables[28]
        rates[24] = -0.016602 * states[24] + variables[257] * variables[33]
        rates[25] = -states[25] * variables[256] + variables[255]
        rates[26] = -0.00660632319563326 * variables[167] + variables[199] * variables[92] + variables[223] * variables[7] + variables[258] * variables[28]
        rates[27] = variables[259] * variables[28]
        rates[28] = -0.016602 * states[28] + variables[262] * variables[33]
        rates[29] = -states[29] * variables[261] + variables[260]
        rates[30] = variables[130] * variables[215] + variables[143] * variables[24] + variables[14] * variables[159] + variables[163] * variables[52] - 0.0172670458146408 * variables[171] + variables[191] * variables[36] + variables[203] * variables[80] + variables[263] * variables[28]
        rates[31] = variables[264] * variables[28]
        rates[32] = -0.016602 * states[32] + variables[267] * variables[33]
        rates[33] = -states[33] * variables[266] + variables[265]
        rates[34] = variables[113] * variables[207] + variables[124] * variables[223] + variables[147] * variables[59] - 0.0284904787482649 * variables[175] + variables[183] * variables[99] + variables[191] * variables[39] + variables[195] * variables[63] + variables[199] * variables[74] + variables[215] * variables[88] + variables[268] * variables[28]
        rates[35] = variables[269] * variables[28]
        rates[36] = -0.016602 * states[36] + variables[272] * variables[33]
        rates[37] = -states[37] * variables[271] + variables[270]
        rates[38] = variables[106] * variables[215] + variables[147] * variables[31] - 0.0151800152129758 * variables[179] + variables[199] * variables[89] + variables[273] * variables[28]
        rates[39] = variables[274] * variables[28]
        rates[40] = -0.016602 * states[40] + variables[277] * variables[33]
        rates[41] = -states[41] * variables[276] + variables[275]
        rates[42] = 1.0 * variables[0] + variables[130] * variables[207] + variables[159] * variables[58] + variables[15] * variables[215] + variables[175] * variables[99] - 0.0209989121379338 * variables[183] + variables[187] * variables[27] + variables[191] * variables[50] + variables[199] * variables[69] + variables[278] * variables[28]
        rates[43] = variables[279] * variables[28]
        rates[44] = -0.016602 * states[44] + variables[282] * variables[33]
        rates[45] = -states[45] * variables[281] + variables[280]
        rates[46] = variables[159] * variables[69] + variables[183] * variables[27] - 0.0118998507056144 * variables[187] + variables[191] * variables[90] + variables[203] * variables[63] + variables[283] * variables[28]
        rates[47] = variables[284] * variables[28]
        rates[48] = -0.016602 * states[48] + variables[287] * variables[33]
        rates[49] = -states[49] * variables[286] + variables[285]
        rates[50] = variables[117] * variables[215] + variables[171] * variables[36] + variables[175] * variables[39] + variables[183] * variables[50] + variables[187] * variables[90] - 0.022188359773101 * variables[191] + variables[203] * variables[83] + variables[219] * variables[48] + variables[288] * variables[28]
        rates[51] = variables[289] * variables[28]
        rates[52] = -0.016602 * states[52] + variables[292] * variables[33]
        rates[53] = -states[53] * variables[291] + variables[290]
        rates[54] = variables[147] * variables[7] + variables[175] * variables[63] - 0.0128788020257737 * variables[195] + variables[207] * variables[44] + variables[219] * variables[49] + variables[28] * variables[293]
        rates[55] = variables[28] * variables[294]
        rates[56] = -0.016602 * states[56] + variables[297] * variables[33]
        rates[57] = -states[57] * variables[296] + variables[295]
        rates[58] = variables[100] * variables[211] + variables[151] * variables[50] + variables[155] * variables[53] + variables[167] * variables[92] + variables[175] * variables[74] + variables[179] * variables[89] + variables[183] * variables[69] - 0.0214542209029461 * variables[199] + variables[215] * variables[5] + variables[28] * variables[298]
        rates[59] = variables[28] * variables[299]
        rates[60] = -0.016602 * states[60] + variables[302] * variables[33]
        rates[61] = -states[61] * variables[301] + variables[300]
        rates[62] = variables[119] * variables[219] + variables[171] * variables[80] + variables[187] * variables[63] + variables[191] * variables[83] - 0.0218062245551532 * variables[203] + variables[207] * variables[35] + variables[28] * variables[303]
        rates[63] = variables[28] * variables[304]
        rates[64] = -0.016602 * states[64] + variables[307] * variables[33]
        rates[65] = -states[65] * variables[306] + variables[305]
        rates[66] = 1.0 * variables[0] + variables[112] * variables[147] + variables[113] * variables[175] + variables[130] * variables[183] + variables[195] * variables[44] + variables[203] * variables[35] - 0.0303833148058918 * variables[207] + variables[219] * variables[94] + variables[28] * variables[308]
        rates[67] = variables[28] * variables[309]
        rates[68] = -0.016602 * states[68] + variables[312] * variables[33]
        rates[69] = -states[69] * variables[311] + variables[310]
        rates[70] = variables[100] * variables[199] + variables[114] * variables[215] + variables[159] * variables[7] + variables[163] * variables[5] - 0.0160249351480499 * variables[211] + variables[28] * variables[313]
        rates[71] = variables[28] * variables[314]
        rates[72] = -0.016602 * states[72] + variables[317] * variables[33]
        rates[73] = -states[73] * variables[316] + variables[315]
        rates[74] = variables[106] * variables[179] + variables[108] * variables[219] + variables[114] * variables[211] + variables[117] * variables[191] + variables[130] * variables[171] + variables[151] * variables[23] + variables[155] * variables[90] + variables[15] * variables[183] + variables[175] * variables[88] + variables[199] * variables[5] - 0.0272967553522912 * variables[215] + variables[28] * variables[318]
        rates[75] = variables[28] * variables[319]
        rates[76] = -0.016602 * states[76] + variables[322] * variables[33]
        rates[77] = -states[77] * variables[321] + variables[320]
        rates[78] = variables[107] * variables[151] + variables[108] * variables[215] + variables[10] * variables[143] + variables[119] * variables[203] + variables[147] * variables[84] + variables[191] * variables[48] + variables[195] * variables[49] + variables[207] * variables[94] - 0.0226091474207625 * variables[219] + variables[28] * variables[323]
        rates[79] = variables[28] * variables[324]
        rates[80] = -0.016602 * states[80] + variables[327] * variables[33]
        rates[81] = -states[81] * variables[326] + variables[325]
        rates[82] = variables[124] * variables[175] + variables[167] * variables[7] - 0.01140988469486 * variables[223] + variables[28] * variables[328]
        rates[83] = variables[28] * variables[329]

    def compute_inputs(self, voi, inputs):
        (t, states, variables) = (voi, self.states, self.variables)
        inputs[0] = 0
        inputs[1] = 0
        inputs[2] = variables[10] * variables[219] + variables[113] * variables[155] - 0.0105347817232959 * variables[143] + variables[171] * variables[24]
        inputs[3] = 0
        inputs[4] = 0
        inputs[5] = 0
        inputs[6] = variables[112] * variables[207] - 0.0187547835509272 * variables[147] + variables[175] * variables[59] + variables[179] * variables[31] + variables[195] * variables[7] + variables[219] * variables[84]
        inputs[7] = 0
        inputs[8] = 0
        inputs[9] = 0
        inputs[10] = variables[106] * variables[163] + variables[107] * variables[219] - 0.00994487038649063 * variables[151] + variables[199] * variables[50] + variables[215] * variables[23]
        inputs[11] = 0
        inputs[12] = 0
        inputs[13] = 0
        inputs[14] = 1.0 * variables[0] + variables[113] * variables[143] - 0.0100670263595698 * variables[155] + variables[199] * variables[53] + variables[215] * variables[90]
        inputs[15] = 0
        inputs[16] = 0
        inputs[17] = 0
        inputs[18] = 1.0 * variables[0] + variables[14] * variables[171] - 0.00908557896398634 * variables[159] + variables[183] * variables[58] + variables[187] * variables[69] + variables[211] * variables[7]
        inputs[19] = 0
        inputs[20] = 0
        inputs[21] = 0
        inputs[22] = variables[106] * variables[151] - 0.0136866968793294 * variables[163] + variables[171] * variables[52] + variables[211] * variables[5]
        inputs[23] = 0
        inputs[24] = 0
        inputs[25] = 0
        inputs[26] = -0.00660632319563326 * variables[167] + variables[199] * variables[92] + variables[223] * variables[7]
        inputs[27] = 0
        inputs[28] = 0
        inputs[29] = 0
        inputs[30] = variables[130] * variables[215] + variables[143] * variables[24] + variables[14] * variables[159] + variables[163] * variables[52] - 0.0172670458146408 * variables[171] + variables[191] * variables[36] + variables[203] * variables[80]
        inputs[31] = 0
        inputs[32] = 0
        inputs[33] = 0
        inputs[34] = variables[113] * variables[207] + variables[124] * variables[223] + variables[147] * variables[59] - 0.0284904787482649 * variables[175] + variables[183] * variables[99] + variables[191] * variables[39] + variables[195] * variables[63] + variables[199] * variables[74] + variables[215] * variables[88]
        inputs[35] = 0
        inputs[36] = 0
        inputs[37] = 0
        inputs[38] = variables[106] * variables[215] + variables[147] * variables[31] - 0.0151800152129758 * variables[179] + variables[199] * variables[89]
        inputs[39] = 0
        inputs[40] = 0
        inputs[41] = 0
        inputs[42] = 1.0 * variables[0] + variables[130] * variables[207] + variables[159] * variables[58] + variables[15] * variables[215] + variables[175] * variables[99] - 0.0209989121379338 * variables[183] + variables[187] * variables[27] + variables[191] * variables[50] + variables[199] * variables[69]
        inputs[43] = 0
        inputs[44] = 0
        inputs[45] = 0
        inputs[46] = variables[159] * variables[69] + variables[183] * variables[27] - 0.0118998507056144 * variables[187] + variables[191] * variables[90] + variables[203] * variables[63]
        inputs[47] = 0
        inputs[48] = 0
        inputs[49] = 0
        inputs[50] = variables[117] * variables[215] + variables[171] * variables[36] + variables[175] * variables[39] + variables[183] * variables[50] + variables[187] * variables[90] - 0.022188359773101 * variables[191] + variables[203] * variables[83] + variables[219] * variables[48]
        inputs[51] = 0
        inputs[52] = 0
        inputs[53] = 0
        inputs[54] = variables[147] * variables[7] + variables[175] * variables[63] - 0.0128788020257737 * variables[195] + variables[207] * variables[44] + variables[219] * variables[49]
        inputs[55] = 0
        inputs[56] = 0
        inputs[57] = 0
        inputs[58] = variables[100] * variables[211] + variables[151] * variables[50] + variables[155] * variables[53] + variables[167] * variables[92] + variables[175] * variables[74] + variables[179] * variables[89] + variables[183] * variables[69] - 0.0214542209029461 * variables[199] + variables[215] * variables[5]
        inputs[59] = 0
        inputs[60] = 0
        inputs[61] = 0
        inputs[62] = variables[119] * variables[219] + variables[171] * variables[80] + variables[187] * variables[63] + variables[191] * variables[83] - 0.0218062245551532 * variables[203] + variables[207] * variables[35]
        inputs[63] = 0
        inputs[64] = 0
        inputs[65] = 0
        inputs[66] = 1.0 * variables[0] + variables[112] * variables[147] + variables[113] * variables[175] + variables[130] * variables[183] + variables[195] * variables[44] + variables[203] * variables[35] - 0.0303833148058918 * variables[207] + variables[219] * variables[94]
        inputs[67] = 0
        inputs[68] = 0
        inputs[69] = 0
        inputs[70] = variables[100] * variables[199] + variables[114] * variables[215] + variables[159] * variables[7] + variables[163] * variables[5] - 0.0160249351480499 * variables[211]
        inputs[71] = 0
        inputs[72] = 0
        inputs[73] = 0
        inputs[74] = variables[106] * variables[179] + variables[108] * variables[219] + variables[114] * variables[211] + variables[117] * variables[191] + variables[130] * variables[171] + variables[151] * variables[23] + variables[155] * variables[90] + variables[15] * variables[183] + variables[175] * variables[88] + variables[199] * variables[5] - 0.0272967553522912 * variables[215]
        inputs[75] = 0
        inputs[76] = 0
        inputs[77] = 0
        inputs[78] = variables[107] * variables[151] + variables[108] * variables[215] + variables[10] * variables[143] + variables[119] * variables[203] + variables[147] * variables[84] + variables[191] * variables[48] + variables[195] * variables[49] + variables[207] * variables[94] - 0.0226091474207625 * variables[219]
        inputs[79] = 0
        inputs[80] = 0
        inputs[81] = 0
        inputs[82] = variables[124] * variables[175] + variables[167] * variables[7] - 0.01140988469486 * variables[223]
        inputs[83] = 0

    def compute_hamiltonian(self, cellHam):
        (t, states, variables) = (self.time, self.states, self.variables)
        cellHam[0] = states[0] ** 2 * variables[225] / 2 - 0.016602 * states[1] * variables[227] - 0.00533333333333333 * states[2] ** 3 + 0.00904 * states[2] ** 2 - states[3] * (-0.0105347817232959 * variables[143] + 0.0034875 * variables[155] + 0.00570494234742999 * variables[171] + 0.00134233937586588 * variables[219])
        cellHam[1] = states[4] ** 2 * variables[230] / 2 - 0.016602 * states[5] * variables[232] - 0.00533333333333333 * states[6] ** 3 + 0.00904 * states[6] ** 2 - states[7] * (-0.0187547835509272 * variables[147] + 0.00178978583448784 * variables[175] + 0.00553238341512669 * variables[179] + 0.00335584843966471 * variables[195] + 0.00624309617371499 * variables[207] + 0.00183366968793294 * variables[219])
        cellHam[2] = -0.00533333333333333 * states[10] ** 3 + 0.00904 * states[10] ** 2 - states[11] * (-0.00994487038649063 * variables[151] + 0.00499763179784911 * variables[163] + 0.000266912464258686 * variables[199] + 0.00233123221661754 * variables[215] + 0.00234909390776529 * variables[219]) + states[8] ** 2 * variables[235] / 2 - 0.016602 * states[9] * variables[237]
        cellHam[3] = states[12] ** 2 * variables[240] / 2 - 0.016602 * states[13] * variables[242] - 0.00533333333333333 * states[14] ** 3 + 0.00904 * states[14] ** 2 - states[15] * (1.0 * variables[0] + 0.0034875 * variables[143] - 0.0100670263595698 * variables[155] + 0.00225306424965365 * variables[199] + 0.00432646210991617 * variables[215])
        cellHam[4] = states[16] ** 2 * variables[245] / 2 - 0.016602 * states[17] * variables[247] - 0.00533333333333333 * states[18] ** 3 + 0.00904 * states[18] ** 2 - states[19] * (1.0 * variables[0] - 0.00908557896398634 * variables[159] + 0.00197736820215088 * variables[171] + 0.00181486232217075 * variables[183] + 0.0019375 * variables[187] + 0.0033558484396647 * variables[211])
        cellHam[5] = states[20] ** 2 * variables[250] / 2 - 0.016602 * states[21] * variables[252] - 0.00533333333333333 * states[22] ** 3 + 0.00904 * states[22] ** 2 - states[23] * (0.00499763179784911 * variables[151] - 0.0136866968793294 * variables[163] + 0.00264853789008382 * variables[171] + 0.00604052719139646 * variables[211])
        cellHam[6] = states[24] ** 2 * variables[255] / 2 - 0.016602 * states[25] * variables[257] - 0.00533333333333333 * states[26] ** 3 + 0.00904 * states[26] ** 2 - states[27] * (-0.00660632319563326 * variables[167] + 0.00325047475596855 * variables[199] + 0.0033558484396647 * variables[223])
        cellHam[7] = states[28] ** 2 * variables[260] / 2 - 0.016602 * states[29] * variables[262] - 0.00533333333333333 * states[30] ** 3 + 0.00904 * states[30] ** 2 - states[31] * (0.00570494234742999 * variables[143] + 0.00197736820215088 * variables[159] + 0.00264853789008382 * variables[163] - 0.0172670458146408 * variables[171] + 0.00233340673210136 * variables[191] + 0.00331131228061603 * variables[203] + 0.00129147836225876 * variables[215])
        cellHam[8] = states[32] ** 2 * variables[265] / 2 - 0.016602 * states[33] * variables[267] - 0.00533333333333333 * states[34] ** 3 + 0.00904 * states[34] ** 2 - states[35] * (0.00178978583448784 * variables[147] - 0.0284904787482649 * variables[175] + 0.0110020181275976 * variables[183] + 0.00177666289008382 * variables[191] + 0.00100675453189941 * variables[195] + 2.3604678562608e-05 * variables[199] + 0.0034875 * variables[207] + 0.00135011643043833 * variables[215] + 0.00805403625519528 * variables[223])
        cellHam[9] = states[36] ** 2 * variables[270] / 2 - 0.016602 * states[37] * variables[272] - 0.00533333333333333 * states[38] ** 3 + 0.00904 * states[38] ** 2 - states[39] * (0.00553238341512669 * variables[147] - 0.0151800152129758 * variables[179] + 0.00465 * variables[199] + 0.00499763179784911 * variables[215])
        cellHam[10] = states[40] ** 2 * variables[275] / 2 - 0.016602 * states[41] * variables[277] - 0.00533333333333333 * states[42] ** 3 + 0.00904 * states[42] ** 2 - states[43] * (1.0 * variables[0] + 0.00181486232217075 * variables[159] + 0.0110020181275976 * variables[175] - 0.0209989121379338 * variables[183] + 0.00462913406379882 * variables[187] + 0.000266912464258687 * variables[191] + 0.0019375 * variables[199] + 0.00129147836225876 * variables[207] + 5.70067978491158e-05 * variables[215])
        cellHam[11] = states[44] ** 2 * variables[280] / 2 - 0.016602 * states[45] * variables[282] - 0.00533333333333333 * states[46] ** 3 + 0.00904 * states[46] ** 2 - states[47] * (0.0019375 * variables[159] + 0.00462913406379882 * variables[183] - 0.0118998507056144 * variables[187] + 0.00432646210991617 * variables[191] + 0.00100675453189941 * variables[203])
        cellHam[12] = states[48] ** 2 * variables[285] / 2 - 0.016602 * states[49] * variables[287] - 0.00533333333333333 * states[50] ** 3 + 0.00904 * states[50] ** 2 - states[51] * (0.00233340673210136 * variables[171] + 0.00177666289008382 * variables[175] + 0.000266912464258687 * variables[183] + 0.00432646210991617 * variables[187] - 0.022188359773101 * variables[191] + 0.00603034390776529 * variables[203] + 0.000611223229310978 * variables[215] + 0.0068433484396647 * variables[219])
        cellHam[13] = states[52] ** 2 * variables[290] / 2 - 0.016602 * states[53] * variables[292] - 0.00533333333333333 * states[54] ** 3 + 0.00904 * states[54] ** 2 - states[55] * (0.00335584843966471 * variables[147] + 0.00100675453189941 * variables[175] - 0.0128788020257737 * variables[195] + 0.00812293911287484 * variables[207] + 0.000393259941334723 * variables[219])
        cellHam[14] = states[56] ** 2 * variables[295] / 2 - 0.016602 * states[57] * variables[297] - 0.00533333333333333 * states[58] ** 3 + 0.00904 * states[58] ** 2 - states[59] * (0.000266912464258686 * variables[151] + 0.00225306424965365 * variables[155] + 0.00325047475596855 * variables[167] + 2.3604678562608e-05 * variables[175] + 0.00465 * variables[179] + 0.0019375 * variables[183] - 0.0214542209029461 * variables[199] + 0.00303213756310611 * variables[211] + 0.00604052719139646 * variables[215])
        cellHam[15] = states[60] ** 2 * variables[300] / 2 - 0.016602 * states[61] * variables[302] - 0.00533333333333333 * states[62] ** 3 + 0.00904 * states[62] ** 2 - states[63] * (0.00331131228061603 * variables[171] + 0.00100675453189941 * variables[187] + 0.00603034390776529 * variables[191] - 0.0218062245551532 * variables[203] + 0.00777166709324443 * variables[207] + 0.00368614674162807 * variables[219])
        cellHam[16] = states[64] ** 2 * variables[305] / 2 - 0.016602 * states[65] * variables[307] - 0.00533333333333333 * states[66] ** 3 + 0.00904 * states[66] ** 2 - states[67] * (1.0 * variables[0] + 0.00624309617371499 * variables[147] + 0.0034875 * variables[175] + 0.00129147836225876 * variables[183] + 0.00812293911287484 * variables[195] + 0.00777166709324443 * variables[203] - 0.0303833148058918 * variables[207] + 0.00346663406379882 * variables[219])
        cellHam[17] = states[68] ** 2 * variables[310] / 2 - 0.016602 * states[69] * variables[312] - 0.00533333333333333 * states[70] ** 3 + 0.00904 * states[70] ** 2 - states[71] * (0.0033558484396647 * variables[159] + 0.00604052719139646 * variables[163] + 0.00303213756310611 * variables[199] - 0.0160249351480499 * variables[211] + 0.00359642195388264 * variables[215])
        cellHam[18] = states[72] ** 2 * variables[315] / 2 - 0.016602 * states[73] * variables[317] - 0.00533333333333333 * states[74] ** 3 + 0.00904 * states[74] ** 2 - states[75] * (0.00233123221661754 * variables[151] + 0.00432646210991617 * variables[155] + 0.00129147836225876 * variables[171] + 0.00135011643043833 * variables[175] + 0.00499763179784911 * variables[179] + 5.70067978491158e-05 * variables[183] + 0.000611223229310978 * variables[191] + 0.00604052719139646 * variables[199] + 0.00359642195388264 * variables[211] - 0.0272967553522912 * variables[215] + 0.00269465526277211 * variables[219])
        cellHam[19] = states[76] ** 2 * variables[320] / 2 - 0.016602 * states[77] * variables[322] - 0.00533333333333333 * states[78] ** 3 + 0.00904 * states[78] ** 2 - states[79] * (0.00134233937586588 * variables[143] + 0.00183366968793294 * variables[147] + 0.00234909390776529 * variables[151] + 0.0068433484396647 * variables[191] + 0.000393259941334723 * variables[195] + 0.00368614674162807 * variables[203] + 0.00346663406379882 * variables[207] + 0.00269465526277211 * variables[215] - 0.0226091474207625 * variables[219])
        cellHam[20] = states[80] ** 2 * variables[325] / 2 - 0.016602 * states[81] * variables[327] - 0.00533333333333333 * states[82] ** 3 + 0.00904 * states[82] ** 2 - states[83] * (0.0033558484396647 * variables[167] + 0.00805403625519528 * variables[175] - 0.01140988469486 * variables[223])
        return cellHam

    def compute_external_energy(self, inputEnergy):
        (t, states, variables) = (self.time, self.states, self.variables)
        inputEnergy[0] = 0
        inputEnergy[1] = 0
        inputEnergy[2] = 0
        inputEnergy[3] = -states[15] * variables[0]
        inputEnergy[4] = -states[19] * variables[0]
        inputEnergy[5] = 0
        inputEnergy[6] = 0
        inputEnergy[7] = 0
        inputEnergy[8] = 0
        inputEnergy[9] = 0
        inputEnergy[10] = -states[43] * variables[0]
        inputEnergy[11] = 0
        inputEnergy[12] = 0
        inputEnergy[13] = 0
        inputEnergy[14] = 0
        inputEnergy[15] = 0
        inputEnergy[16] = -states[67] * variables[0]
        inputEnergy[17] = 0
        inputEnergy[18] = 0
        inputEnergy[19] = 0
        inputEnergy[20] = 0
        return inputEnergy

    def compute_total_input_energy(self, totalInputEnergy):
        (t, states, variables) = (self.time, self.states, self.variables)
        totalInputEnergy[0] = states[3] * (0.0105347817232959 * variables[143] - 0.0034875 * variables[155] - 0.00570494234742999 * variables[171] - 0.00134233937586588 * variables[219])
        totalInputEnergy[1] = states[7] * (0.0187547835509272 * variables[147] - 0.00178978583448784 * variables[175] - 0.00553238341512669 * variables[179] - 0.00335584843966471 * variables[195] - 0.00624309617371499 * variables[207] - 0.00183366968793294 * variables[219])
        totalInputEnergy[2] = states[11] * (0.00994487038649063 * variables[151] - 0.00499763179784911 * variables[163] - 0.000266912464258686 * variables[199] - 0.00233123221661754 * variables[215] - 0.00234909390776529 * variables[219])
        totalInputEnergy[3] = states[15] * (-variables[0] - 0.0034875 * variables[143] + 0.0100670263595698 * variables[155] - 0.00225306424965365 * variables[199] - 0.00432646210991617 * variables[215])
        totalInputEnergy[4] = states[19] * (-variables[0] + 0.00908557896398634 * variables[159] - 0.00197736820215088 * variables[171] - 0.00181486232217075 * variables[183] - 0.0019375 * variables[187] - 0.0033558484396647 * variables[211])
        totalInputEnergy[5] = states[23] * (-0.00499763179784911 * variables[151] + 0.0136866968793294 * variables[163] - 0.00264853789008382 * variables[171] - 0.00604052719139646 * variables[211])
        totalInputEnergy[6] = states[27] * (0.00660632319563326 * variables[167] - 0.00325047475596855 * variables[199] - 0.0033558484396647 * variables[223])
        totalInputEnergy[7] = states[31] * (-0.00570494234742999 * variables[143] - 0.00197736820215088 * variables[159] - 0.00264853789008382 * variables[163] + 0.0172670458146408 * variables[171] - 0.00233340673210136 * variables[191] - 0.00331131228061603 * variables[203] - 0.00129147836225876 * variables[215])
        totalInputEnergy[8] = states[35] * (-0.00178978583448784 * variables[147] + 0.0284904787482649 * variables[175] - 0.0110020181275976 * variables[183] - 0.00177666289008382 * variables[191] - 0.00100675453189941 * variables[195] - 2.3604678562608e-05 * variables[199] - 0.0034875 * variables[207] - 0.00135011643043833 * variables[215] - 0.00805403625519528 * variables[223])
        totalInputEnergy[9] = states[39] * (-0.00553238341512669 * variables[147] + 0.0151800152129758 * variables[179] - 0.00465 * variables[199] - 0.00499763179784911 * variables[215])
        totalInputEnergy[10] = states[43] * (-variables[0] - 0.00181486232217075 * variables[159] - 0.0110020181275976 * variables[175] + 0.0209989121379338 * variables[183] - 0.00462913406379882 * variables[187] - 0.000266912464258687 * variables[191] - 0.0019375 * variables[199] - 0.00129147836225876 * variables[207] - 5.70067978491158e-05 * variables[215])
        totalInputEnergy[11] = states[47] * (-0.0019375 * variables[159] - 0.00462913406379882 * variables[183] + 0.0118998507056144 * variables[187] - 0.00432646210991617 * variables[191] - 0.00100675453189941 * variables[203])
        totalInputEnergy[12] = states[51] * (-0.00233340673210136 * variables[171] - 0.00177666289008382 * variables[175] - 0.000266912464258687 * variables[183] - 0.00432646210991617 * variables[187] + 0.022188359773101 * variables[191] - 0.00603034390776529 * variables[203] - 0.000611223229310978 * variables[215] - 0.0068433484396647 * variables[219])
        totalInputEnergy[13] = states[55] * (-0.00335584843966471 * variables[147] - 0.00100675453189941 * variables[175] + 0.0128788020257737 * variables[195] - 0.00812293911287484 * variables[207] - 0.000393259941334723 * variables[219])
        totalInputEnergy[14] = states[59] * (-0.000266912464258686 * variables[151] - 0.00225306424965365 * variables[155] - 0.00325047475596855 * variables[167] - 2.3604678562608e-05 * variables[175] - 0.00465 * variables[179] - 0.0019375 * variables[183] + 0.0214542209029461 * variables[199] - 0.00303213756310611 * variables[211] - 0.00604052719139646 * variables[215])
        totalInputEnergy[15] = states[63] * (-0.00331131228061603 * variables[171] - 0.00100675453189941 * variables[187] - 0.00603034390776529 * variables[191] + 0.0218062245551532 * variables[203] - 0.00777166709324443 * variables[207] - 0.00368614674162807 * variables[219])
        totalInputEnergy[16] = states[67] * (-variables[0] - 0.00624309617371499 * variables[147] - 0.0034875 * variables[175] - 0.00129147836225876 * variables[183] - 0.00812293911287484 * variables[195] - 0.00777166709324443 * variables[203] + 0.0303833148058918 * variables[207] - 0.00346663406379882 * variables[219])
        totalInputEnergy[17] = states[71] * (-0.0033558484396647 * variables[159] - 0.00604052719139646 * variables[163] - 0.00303213756310611 * variables[199] + 0.0160249351480499 * variables[211] - 0.00359642195388264 * variables[215])
        totalInputEnergy[18] = states[75] * (-0.00233123221661754 * variables[151] - 0.00432646210991617 * variables[155] - 0.00129147836225876 * variables[171] - 0.00135011643043833 * variables[175] - 0.00499763179784911 * variables[179] - 5.70067978491158e-05 * variables[183] - 0.000611223229310978 * variables[191] - 0.00604052719139646 * variables[199] - 0.00359642195388264 * variables[211] + 0.0272967553522912 * variables[215] - 0.00269465526277211 * variables[219])
        totalInputEnergy[19] = states[79] * (-0.00134233937586588 * variables[143] - 0.00183366968793294 * variables[147] - 0.00234909390776529 * variables[151] - 0.0068433484396647 * variables[191] - 0.000393259941334723 * variables[195] - 0.00368614674162807 * variables[203] - 0.00346663406379882 * variables[207] - 0.00269465526277211 * variables[215] + 0.0226091474207625 * variables[219])
        totalInputEnergy[20] = states[83] * (-0.0033558484396647 * variables[167] - 0.00805403625519528 * variables[175] + 0.01140988469486 * variables[223])
        return totalInputEnergy

    def process_time_sensitive_events(self, voi):
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
        (states, rates, variables) = (self.states, self.rates, self.variables)
        variables[0] = 0.0
        if voi > 100 and voi < 110:
            variables[0] = 0.5
        self.compute_variables(voi)

    def process_events(self, voi):
        """Method to process events such as (re)setting inputs, updating switches etc
        The method is called after each successful ode step
        Args:
            voi (int) : Current value of the variable of integration (time)
        """
        (states, rates, variables) = (self.states, self.rates, self.variables)

    def getStateValues(self, statename):
        return self.states[self.stateIndexes[statename]]

    def setStateValues(self, statename, values):
        self.states[self.stateIndexes[statename]] = values

    def rhs(self, voi, states):
        self.states = states
        self.process_time_sensitive_events(voi)
        self.compute_rates(voi)
        return self.rates

    def step(self, step=1.0):
        if self.odeintegrator.successful():
            self.odeintegrator.integrate(step)
            self.time = self.odeintegrator.t
            self.states = self.odeintegrator.y
            self.process_events(self.time)
        else:
            raise Exception('ODE integrator in failed state!')

class FTUStepper_test(FTUStepper):
    """
    Machine generated code for running experiment test with 
    time [0, 400, 400]
    and inputcode block
    
i_1 = 0
if t>100 and t<110:
    i_1 = 0.5

    """
    def __init__(self) -> None:
        super().__init__()
        self.cellHam = np.zeros(self.CELL_COUNT)
        self.energyInputs = np.zeros(self.CELL_COUNT)
        self.totalEnergyInputs = np.zeros(self.CELL_COUNT)
        self.inputs = np.zeros(self.STATE_COUNT)
        self.times     = []
        self.allstates = []
        self.allrates  = []
        self.allhamiltonians = []
        self.allenergyinputs = []
        self.alltotalenergyinputs = []                
        self.allinputs = []


    def process_time_sensitive_events(self,voi):
        t = voi
        states, rates, variables = self.states,self.rates,self.variables
        #Refactored code to match variable to array maps
        variables[0] = 0
        if t > 100 and t < 110:
            variables[0] = 0.5

        #End of refactored code
        self.compute_variables(voi)
        
    def process_events(self,voi):
        self.times.append(self.time)
        self.allstates.append(np.copy(self.states))
        self.allrates.append(np.copy(self.rates))
        self.compute_hamiltonian(self.cellHam)
        self.allhamiltonians.append(np.copy(self.cellHam))
        self.compute_external_energy(self.energyInputs)
        self.allenergyinputs.append(np.copy(self.energyInputs))
        self.compute_total_input_energy(self.totalEnergyInputs)
        self.alltotalenergyinputs.append(np.copy(self.totalEnergyInputs))
        self.compute_inputs(self.time,self.inputs)
        self.allinputs.append(np.copy(self.inputs))
                
    def run(self):
        try:
            print(f"Starting experiment FTUStepper test - time steps 0,400,400")
            voi = np.linspace(0,400,400)
            if self.time>0:
                self.time = 0
                self.odeintegrator.set_initial_value(self.states, self.time)
            tic = time.time()
            for t in voi[1:]:
                self.step(t)
            toc = time.time()
            print(f"Completed experiment FTUStepper test in {(toc-tic):4f} seconds")
        except Exception as ex:
            print(f"Failed experiment FTUStepper test with {ex}")
            
    def save(self,filename):
        with open(filename,'wb+') as sv:
            np.save(sv,np.array(self.times))
            # tranpose to get it in states x time
            np.save(sv,np.array(self.allstates).T) 
            np.save(sv,np.array(self.allrates).T)
            np.save(sv,np.array(self.allhamiltonians).T)
            np.save(sv,np.array(self.allenergyinputs).T)
            np.save(sv,np.array(self.alltotalenergyinputs).T)
            np.save(sv,np.array(self.allinputs).T)
            
if __name__ == '__main__':            
    fstep = FTUStepper_test()
    fstep.run()
    import matplotlib.pyplot as plt

    t = np.array(fstep.times)
    states = np.array(fstep.allstates)
    fig = plt.figure(figsize=(50, 50))
    grid = plt.GridSpec((fstep.CELL_COUNT+1)//3, 3, wspace=0.2, hspace=0.5)

    ix = 0
    numstates = len(fstep.stateIndexes)
    for i in range((fstep.CELL_COUNT+1)//3):
        for j in range(3):
            ax = plt.subplot(grid[i, j])
            ax.plot(t,states[:,ix+2])
            ax.title.set_text(f'{ix//numstates+1}')
            ix += numstates
            if ix+numstates > states.shape[1]:
                break
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.3)
    fig.savefig(f"FTUStepper_test_results.png",dpi=300)
    plt.show()             

