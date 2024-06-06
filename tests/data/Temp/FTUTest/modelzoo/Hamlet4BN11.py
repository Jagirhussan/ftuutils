import numpy as np
from numpy import exp

def Heaviside(x):
    if x > 0:
        return 1.0
    return 0.0


class HamletForBoundaryNode11():
    
    def __init__(self,
                 states=np.zeros(4),
                 rates=np.zeros(4),
                 fomstates=np.zeros(84),
                 inputs=np.zeros(4)) -> None:
        self.cellIndex = 10
        self.statenames = ["Tai_11","Ta_11","u_11","v_11"]
        self.states = states
        self.rates = rates
        self.fomstates = fomstates
        self.inputs = inputs        
        self.states = 0.00000,0.00100,0.00000,0.03604

    def getHamiltonian(self):
        Tai_11,Ta_11,u_11,v_11= self.states
        i_1_2 = self.inputs[0]
        u_u_13 = self.fomstates[50]
        u_u_17 = self.fomstates[66]
        u_u_9 = self.fomstates[34]
        u_u_15 = self.fomstates[58]
        u_u_5 = self.fomstates[18]
        u_u_12 = self.fomstates[46]
        u_u_19 = self.fomstates[74]
        u_u_11 = self.fomstates[42]

        return (-0.308286455419572*Ta_11 + (Tai_11**3*(-0.016502*Heaviside(0.2925 - Tai_11)*Heaviside(u_11 - 0.78) + 0.016602) - 0.0106666666666667*u_11**3 + 0.01808*u_11**2 - 2*v_11*(1.0*i_1_2 - 0.0209989121379338*u_u_11 + 0.00462913406379882*u_u_12 + 0.000266912464258687*u_u_13 + 0.0019375*u_u_15 + 0.00129147836225876*u_u_17 + 5.70067978491158e-5*u_u_19 + 0.00181486232217075*u_u_5 + 0.0110020181275976*u_u_9))*exp(270.806766140199*(u_11 - 1)**2))*exp(-270.806766140199*(u_11 - 1)**2)/2
        
    def getEnergyFromExternalInputs(self):
        Tai_11,Ta_11,u_11,v_11= self.states
        i_1_2 = self.inputs[0]
        u_u_13 = self.fomstates[50]
        u_u_17 = self.fomstates[66]
        u_u_9 = self.fomstates[34]
        u_u_15 = self.fomstates[58]
        u_u_5 = self.fomstates[18]
        u_u_12 = self.fomstates[46]
        u_u_19 = self.fomstates[74]
        u_u_11 = self.fomstates[42]

        totE = (-0.308286455419572*Ta_11 + (Tai_11**3*(-0.016502*Heaviside(0.2925 - Tai_11)*Heaviside(u_11 - 0.78) + 0.016602) - 0.0106666666666667*u_11**3 + 0.01808*u_11**2 - 2*v_11*(1.0*i_1_2 - 0.0209989121379338*u_u_11 + 0.00462913406379882*u_u_12 + 0.000266912464258687*u_u_13 + 0.0019375*u_u_15 + 0.00129147836225876*u_u_17 + 5.70067978491158e-5*u_u_19 + 0.00181486232217075*u_u_5 + 0.0110020181275976*u_u_9))*exp(270.806766140199*(u_11 - 1)**2))*exp(-270.806766140199*(u_11 - 1)**2)/2
        i_1_2 = 0.0
        E = (-0.308286455419572*Ta_11 + (Tai_11**3*(-0.016502*Heaviside(0.2925 - Tai_11)*Heaviside(u_11 - 0.78) + 0.016602) - 0.0106666666666667*u_11**3 + 0.01808*u_11**2 - 2*v_11*(1.0*i_1_2 - 0.0209989121379338*u_u_11 + 0.00462913406379882*u_u_12 + 0.000266912464258687*u_u_13 + 0.0019375*u_u_15 + 0.00129147836225876*u_u_17 + 5.70067978491158e-5*u_u_19 + 0.00181486232217075*u_u_5 + 0.0110020181275976*u_u_9))*exp(270.806766140199*(u_11 - 1)**2))*exp(-270.806766140199*(u_11 - 1)**2)/2
        return totE - E

        
    def computeRHS(self,t):
        Tai_11,Ta_11,u_11,v_11= self.states
        i_1_2 = self.inputs[0]
        u_u_13 = self.fomstates[50]
        u_u_17 = self.fomstates[66]
        u_u_9 = self.fomstates[34]
        u_u_15 = self.fomstates[58]
        u_u_5 = self.fomstates[18]
        u_u_12 = self.fomstates[46]
        u_u_19 = self.fomstates[74]
        u_u_11 = self.fomstates[42]

        self.rates = [-0.016602*Tai_11 + 0.154143227709786*u_11*exp(-270.806766140199*(u_11 - 1)**2),
            Ta_11*(0.016502*Heaviside(0.2925 - Tai_11)*Heaviside(u_11 - 0.78) - 0.016602) + Tai_1(-0.016502*Heaviside(0.2925 - Tai_11)*Heaviside(u_11 - 0.78) + 0.016602),
            1.0*i_1_2 - u_1(0.0775*v_11 + 0.62*(u_11 - 1)*(u_11 - 0.13)) - 0.0209989121379338*u_u_11 + 0.00462913406379882*u_u_12 + 0.000266912464258687*u_u_13 + 0.0019375*u_u_15 + 0.00129147836225876*u_u_17 + 5.70067978491158e-5*u_u_19 + 0.00181486232217075*u_u_5 + 0.0110020181275976*u_u_9,
            -0.0775*v_11*(8.0*u_11*(u_11 - 1.13) + v_11)*(0.002*u_11 + 0.2*v_11 + 0.0006)/(u_11 + 0.3)]
        return self.rates

        
if __name__ == '__main__':
    hm = HamletForBoundaryNode11()
    hm.getHamiltonian()
    hm.computeRHS(0.0)
