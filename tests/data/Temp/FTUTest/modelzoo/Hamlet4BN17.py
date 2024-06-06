import numpy as np
from numpy import exp

def Heaviside(x):
    if x > 0:
        return 1.0
    return 0.0


class HamletForBoundaryNode17():
    
    def __init__(self,
                 states=np.zeros(4),
                 rates=np.zeros(4),
                 fomstates=np.zeros(84),
                 inputs=np.zeros(4)) -> None:
        self.cellIndex = 16
        self.statenames = ["Tai_17","Ta_17","u_17","v_17"]
        self.states = states
        self.rates = rates
        self.fomstates = fomstates
        self.inputs = inputs        
        self.states = 0.00000,0.00100,0.00000,0.03604

    def getHamiltonian(self):
        Tai_17,Ta_17,u_17,v_17= self.states
        i_1_2 = self.inputs[0]
        u_u_9 = self.fomstates[34]
        u_u_17 = self.fomstates[66]
        u_u_20 = self.fomstates[78]
        u_u_2 = self.fomstates[6]
        u_u_14 = self.fomstates[54]
        u_u_11 = self.fomstates[42]
        u_u_16 = self.fomstates[62]

        return (-0.308286455419572*Ta_17 + (Tai_17**3*(-0.016502*Heaviside(0.2925 - Tai_17)*Heaviside(u_17 - 0.78) + 0.016602) - 0.0106666666666667*u_17**3 + 0.01808*u_17**2 - 2*v_17*(1.0*i_1_2 + 0.00129147836225876*u_u_11 + 0.00812293911287484*u_u_14 + 0.00777166709324443*u_u_16 - 0.0303833148058918*u_u_17 + 0.00624309617371499*u_u_2 + 0.00346663406379882*u_u_20 + 0.0034875*u_u_9))*exp(270.806766140199*(u_17 - 1)**2))*exp(-270.806766140199*(u_17 - 1)**2)/2
        
    def getEnergyFromExternalInputs(self):
        Tai_17,Ta_17,u_17,v_17= self.states
        i_1_2 = self.inputs[0]
        u_u_9 = self.fomstates[34]
        u_u_17 = self.fomstates[66]
        u_u_20 = self.fomstates[78]
        u_u_2 = self.fomstates[6]
        u_u_14 = self.fomstates[54]
        u_u_11 = self.fomstates[42]
        u_u_16 = self.fomstates[62]

        totE = (-0.308286455419572*Ta_17 + (Tai_17**3*(-0.016502*Heaviside(0.2925 - Tai_17)*Heaviside(u_17 - 0.78) + 0.016602) - 0.0106666666666667*u_17**3 + 0.01808*u_17**2 - 2*v_17*(1.0*i_1_2 + 0.00129147836225876*u_u_11 + 0.00812293911287484*u_u_14 + 0.00777166709324443*u_u_16 - 0.0303833148058918*u_u_17 + 0.00624309617371499*u_u_2 + 0.00346663406379882*u_u_20 + 0.0034875*u_u_9))*exp(270.806766140199*(u_17 - 1)**2))*exp(-270.806766140199*(u_17 - 1)**2)/2
        i_1_2 = 0.0
        E = (-0.308286455419572*Ta_17 + (Tai_17**3*(-0.016502*Heaviside(0.2925 - Tai_17)*Heaviside(u_17 - 0.78) + 0.016602) - 0.0106666666666667*u_17**3 + 0.01808*u_17**2 - 2*v_17*(1.0*i_1_2 + 0.00129147836225876*u_u_11 + 0.00812293911287484*u_u_14 + 0.00777166709324443*u_u_16 - 0.0303833148058918*u_u_17 + 0.00624309617371499*u_u_2 + 0.00346663406379882*u_u_20 + 0.0034875*u_u_9))*exp(270.806766140199*(u_17 - 1)**2))*exp(-270.806766140199*(u_17 - 1)**2)/2
        return totE - E

        
    def computeRHS(self,t):
        Tai_17,Ta_17,u_17,v_17= self.states
        i_1_2 = self.inputs[0]
        u_u_9 = self.fomstates[34]
        u_u_17 = self.fomstates[66]
        u_u_20 = self.fomstates[78]
        u_u_2 = self.fomstates[6]
        u_u_14 = self.fomstates[54]
        u_u_11 = self.fomstates[42]
        u_u_16 = self.fomstates[62]

        self.rates = [-0.016602*Tai_17 + 0.154143227709786*u_17*exp(-270.806766140199*(u_17 - 1)**2),
            Ta_17*(0.016502*Heaviside(0.2925 - Tai_17)*Heaviside(u_17 - 0.78) - 0.016602) + Tai_17**2*(-0.016502*Heaviside(0.2925 - Tai_17)*Heaviside(u_17 - 0.78) + 0.016602),
            1.0*i_1_2 - u_17**2*(0.0775*v_17 + 0.62*(u_17 - 1)*(u_17 - 0.13)) + 0.00129147836225876*u_u_11 + 0.00812293911287484*u_u_14 + 0.00777166709324443*u_u_16 - 0.0303833148058918*u_u_17 + 0.00624309617371499*u_u_2 + 0.00346663406379882*u_u_20 + 0.0034875*u_u_9,
            -0.0775*v_17*(8.0*u_17*(u_17 - 1.13) + v_17)*(0.002*u_17 + 0.2*v_17 + 0.0006)/(u_17 + 0.3)]
        return self.rates

        
if __name__ == '__main__':
    hm = HamletForBoundaryNode17()
    hm.getHamiltonian()
    hm.computeRHS(0.0)
