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
        plt.savefig('Concentration_vs_Time.png')
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
    
    def simulate_monte_carlo(self,total_time=1000,dt=0.1,n_particle = 100000):
        steps = int(total_time/dt)
        
        t = np.zeros(steps)
        survivors = np.zeros(steps)

        alive = np.ones(n_particle,dtype=bool)
        survivors[0] = n_particle

        decay_prob = self.K*dt

        if decay_prob > 0.1:
            raise ValueError("Choose smaller dt: λ·dt must be ≪ 1 for Monte Carlo")

        for i in range(steps-1):
            random_numbers = np.random.rand(n_particle)
            decayed = (random_numbers < decay_prob) & alive
            alive[decayed] = False

            survivors[i+1] = np.sum(alive)
            t[i+1] = t[i] + dt

        return t , survivors
    
    def plot_monte_carlo(self,total_time=1000,dt=0.1,n_particle = 100000):
        t_mc , survivors = self.simulate_monte_carlo(total_time=total_time,dt=dt,n_particle = n_particle)

        analytical = n_particle*np.exp(-self.K*t_mc)

        plt.plot(t_mc,survivors,label ="Monte carlo",alpha = 0.7)
        plt.plot(t_mc,analytical,label='Analytical')
        plt.xlabel("Time")
        plt.ylabel("Number of Nuclei")
        plt.title("Monte Carlo Radioactive Decay")
        plt.legend()
        plt.grid(True)
        plt.savefig("monte_carlo_decay.png")
        plt.show()


if __name__ == "__main__":
    # 1. Setup Parameters
    # Let's use a decay constant (K) of 0.693, which makes the half-life exactly 1.0 seconds.
    initial_conc = 100.0
    decay_constant = 0.693
    
    # 2. Initialize the class
    decay_sim = RadioactiveDecay(initial_concentration=initial_conc, decay_const=decay_constant)
    
    # 3. Test Half-Life calculation
    print(f"--- Radioactive Decay Simulation ---")
    print(f"Initial Concentration: {initial_conc}")
    print(f"Decay Constant (K): {decay_constant}")
    print(f"Calculated Half-Life: {decay_sim.half_life():.3f} seconds")
    print("-" * 40)

    # 4. Test Numerical Simulation and Save Data
    print("Running Numerical Simulation and saving to 'Radioactive.csv'...")
    df = decay_sim.save_data("Radioactive.csv")
    print(f"First 5 rows of data:\n{df.head()}")
    
    # 5. Plot the Numerical Comparison (Forward Euler vs Analytical)
    print("\nDisplaying Numerical Plot...")
    decay_sim.plot()

    # 6. Test Monte Carlo Simulation
    # We use a smaller total_time and n_particle so it runs quickly
    print("Running Monte Carlo Simulation (10,000 particles)...")
    decay_sim.plot_monte_carlo(total_time=10, dt=0.05, n_particle=10000)
    
    print("Test Complete.")