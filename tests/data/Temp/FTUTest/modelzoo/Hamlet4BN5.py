import numpy as np
from numpy import exp

def Heaviside(x):
    if x > 0:
        return 1.0
    return 0.0


class HamletForBoundaryNode5():
    
    def __init__(self,
                 states=np.zeros(4),
                 rates=np.zeros(4),
                 fomstates=np.zeros(84),
                 inputs=np.zeros(4)) -> None:
        self.cellIndex = 4
        self.statenames = ["Tai_5","Ta_5","u_5","v_5"]
        self.states = states
        self.rates = rates
        self.fomstates = fomstates
        self.inputs = inputs        
        self.states = 0.00000,0.00100,0.00000,0.03604

    def getHamiltonian(self):
        Tai_5,Ta_5,u_5,v_5= self.states
        i_1_2 = self.inputs[0]
        u_u_18 = self.fomstates[70]
        u_u_5 = self.fomstates[18]
        u_u_12 = self.fomstates[46]
        u_u_8 = self.fomstates[30]
        u_u_11 = self.fomstates[42]

        return (-0.308286455419572*Ta_5 + (Tai_5**3*(-0.016502*Heaviside(0.2925 - Tai_5)*Heaviside(u_5 - 0.78) + 0.016602) - 0.0106666666666667*u_5**3 + 0.01808*u_5**2 - 2*v_5*(1.0*i_1_2 + 0.00181486232217075*u_u_11 + 0.0019375*u_u_12 + 0.0033558484396647*u_u_18 - 0.00908557896398634*u_u_5 + 0.00197736820215088*u_u_8))*exp(270.806766140199*(u_5 - 1)**2))*exp(-270.806766140199*(u_5 - 1)**2)/2
        
    def getEnergyFromExternalInputs(self):
        Tai_5,Ta_5,u_5,v_5= self.states
        i_1_2 = self.inputs[0]
        u_u_18 = self.fomstates[70]
        u_u_5 = self.fomstates[18]
        u_u_12 = self.fomstates[46]
        u_u_8 = self.fomstates[30]
        u_u_11 = self.fomstates[42]

        totE = (-0.308286455419572*Ta_5 + (Tai_5**3*(-0.016502*Heaviside(0.2925 - Tai_5)*Heaviside(u_5 - 0.78) + 0.016602) - 0.0106666666666667*u_5**3 + 0.01808*u_5**2 - 2*v_5*(1.0*i_1_2 + 0.00181486232217075*u_u_11 + 0.0019375*u_u_12 + 0.0033558484396647*u_u_18 - 0.00908557896398634*u_u_5 + 0.00197736820215088*u_u_8))*exp(270.806766140199*(u_5 - 1)**2))*exp(-270.806766140199*(u_5 - 1)**2)/2
        i_1_2 = 0.0
        E = (-0.308286455419572*Ta_5 + (Tai_5**3*(-0.016502*Heaviside(0.2925 - Tai_5)*Heaviside(u_5 - 0.78) + 0.016602) - 0.0106666666666667*u_5**3 + 0.01808*u_5**2 - 2*v_5*(1.0*i_1_2 + 0.00181486232217075*u_u_11 + 0.0019375*u_u_12 + 0.0033558484396647*u_u_18 - 0.00908557896398634*u_u_5 + 0.00197736820215088*u_u_8))*exp(270.806766140199*(u_5 - 1)**2))*exp(-270.806766140199*(u_5 - 1)**2)/2
        return totE - E

        
    def computeRHS(self,t):
        Tai_5,Ta_5,u_5,v_5= self.states
        i_1_2 = self.inputs[0]
        u_u_18 = self.fomstates[70]
        u_u_5 = self.fomstates[18]
        u_u_12 = self.fomstates[46]
        u_u_8 = self.fomstates[30]
        u_u_11 = self.fomstates[42]

        self.rates = [-0.016602*Tai_5 + 0.154143227709786*u_5*exp(-270.806766140199*(u_5 - 1)**2),
            Ta_5*(0.016502*Heaviside(0.2925 - Tai_5)*Heaviside(u_5 - 0.78) - 0.016602) + Tai_5**2*(-0.016502*Heaviside(0.2925 - Tai_5)*Heaviside(u_5 - 0.78) + 0.016602),
            1.0*i_1_2 - u_5**2*(0.0775*v_5 + 0.62*(u_5 - 1)*(u_5 - 0.13)) + 0.00181486232217075*u_u_11 + 0.0019375*u_u_12 + 0.0033558484396647*u_u_18 - 0.00908557896398634*u_u_5 + 0.00197736820215088*u_u_8,
            -0.0775*v_5*(8.0*u_5*(u_5 - 1.13) + v_5)*(0.002*u_5 + 0.2*v_5 + 0.0006)/(u_5 + 0.3)]
        return self.rates

        
if __name__ == '__main__':
    hm = HamletForBoundaryNode5()
    hm.getHamiltonian()
    hm.computeRHS(0.0)
