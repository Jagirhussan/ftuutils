import numpy as np
from numpy import exp

def Heaviside(x):
    if x > 0:
        return 1.0
    return 0.0


class HamletForBoundaryNode4():
    
    def __init__(self,
                 states=np.zeros(4),
                 rates=np.zeros(4),
                 fomstates=np.zeros(84),
                 inputs=np.zeros(4)) -> None:
        self.cellIndex = 3
        self.statenames = ["Tai_4","Ta_4","u_4","v_4"]
        self.states = states
        self.rates = rates
        self.fomstates = fomstates
        self.inputs = inputs        
        self.states = 0.00000,0.00100,0.00000,0.03604

    def getHamiltonian(self):
        Tai_4,Ta_4,u_4,v_4= self.states
        i_1_2 = self.inputs[0]
        u_u_1 = self.fomstates[2]
        u_u_19 = self.fomstates[74]
        u_u_4 = self.fomstates[14]
        u_u_15 = self.fomstates[58]

        return -0.154143227709786*Ta_4*exp(-270.806766140199*(u_4 - 1)**2) - 0.008251*Tai_4**3*Heaviside(0.2925 - Tai_4)*Heaviside(u_4 - 0.78) + 0.008301*Tai_4**3 - 1.0*i_1_2*v_4 - 0.00533333333333333*u_4**3 + 0.00904*u_4**2 - 0.0034875*u_u_1*v_4 - 0.00225306424965365*u_u_15*v_4 - 0.00432646210991617*u_u_19*v_4 + 0.0100670263595698*u_u_4*v_4
        
    def getEnergyFromExternalInputs(self):
        Tai_4,Ta_4,u_4,v_4= self.states
        i_1_2 = self.inputs[0]
        u_u_1 = self.fomstates[2]
        u_u_19 = self.fomstates[74]
        u_u_4 = self.fomstates[14]
        u_u_15 = self.fomstates[58]

        totE = -0.154143227709786*Ta_4*exp(-270.806766140199*(u_4 - 1)**2) - 0.008251*Tai_4**3*Heaviside(0.2925 - Tai_4)*Heaviside(u_4 - 0.78) + 0.008301*Tai_4**3 - 1.0*i_1_2*v_4 - 0.00533333333333333*u_4**3 + 0.00904*u_4**2 - 0.0034875*u_u_1*v_4 - 0.00225306424965365*u_u_15*v_4 - 0.00432646210991617*u_u_19*v_4 + 0.0100670263595698*u_u_4*v_4
        i_1_2 = 0.0
        E = -0.154143227709786*Ta_4*exp(-270.806766140199*(u_4 - 1)**2) - 0.008251*Tai_4**3*Heaviside(0.2925 - Tai_4)*Heaviside(u_4 - 0.78) + 0.008301*Tai_4**3 - 1.0*i_1_2*v_4 - 0.00533333333333333*u_4**3 + 0.00904*u_4**2 - 0.0034875*u_u_1*v_4 - 0.00225306424965365*u_u_15*v_4 - 0.00432646210991617*u_u_19*v_4 + 0.0100670263595698*u_u_4*v_4
        return totE - E

        
    def computeRHS(self,t):
        Tai_4,Ta_4,u_4,v_4= self.states
        i_1_2 = self.inputs[0]
        u_u_1 = self.fomstates[2]
        u_u_19 = self.fomstates[74]
        u_u_4 = self.fomstates[14]
        u_u_15 = self.fomstates[58]

        self.rates = [-0.016602*Tai_4 + 0.154143227709786*u_4*exp(-270.806766140199*(u_4 - 1)**2),
            Ta_4*(0.016502*Heaviside(0.2925 - Tai_4)*Heaviside(u_4 - 0.78) - 0.016602) + Tai_4**2*(-0.016502*Heaviside(0.2925 - Tai_4)*Heaviside(u_4 - 0.78) + 0.016602),
            1.0*i_1_2 - u_4**2*(0.0775*v_4 + 0.62*(u_4 - 1)*(u_4 - 0.13)) + 0.0034875*u_u_1 + 0.00225306424965365*u_u_15 + 0.00432646210991617*u_u_19 - 0.0100670263595698*u_u_4,
            -0.0775*v_4*(8.0*u_4*(u_4 - 1.13) + v_4)*(0.002*u_4 + 0.2*v_4 + 0.0006)/(u_4 + 0.3)]
        return self.rates

        
if __name__ == '__main__':
    hm = HamletForBoundaryNode4()
    hm.getHamiltonian()
    hm.computeRHS(0.0)
