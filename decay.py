import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class RadioactiveDecay:
    def __init__(self,initial_concentration:float,decay_const:float):
        if initial_concentration <= 0:
            raise ValueError('Initial concentration must be greater than 0')
        
        if decay_const<= 0:
            raise ValueError('Decay constant must be greater than 0')
        self.K = decay_const
        self.A0 = initial_concentration

    def simulate(self,total_time = 1000,dt=0.001):
        steps = int(total_time/dt)

        t = np.zeros(steps)
        A = np.zeros(steps)

        A[0] = self.A0

        for i in range(steps-1):
            A[i+1] = A[i] - self.K*A[i]*dt
            A[i+1] = np.maximum(A[i+1],0.0)
            t[i+1] = t[i] +dt 

        return A,t
        
    def plot(self):
        A,t = self.simulate()
        exact = self.A0*np.exp(-self.K*t)

        plt.plot(t,A,label='Concentration (Numerical)')
        plt.plot(t,exact,label = 'Concentration (Analytical)')
        plt.xlabel('Time (s)')
        plt.ylabel('Concentration')
        plt.legend()
        plt.show()

    def save_data(self,filename="Radioactive.csv"):
        A,t = self.simulate()

        df = pd.DataFrame({"Time":t,
                           "Numerical":A,
                           "Analytical":self.A0*np.exp(-self.K*t)})
        
        df.to_csv(filename,index=False)

        return df
    
    def half_life(self):
        return np.log(2)/self.K