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
        steps = int(total_time/dt)+1

        t = np.zeros(steps)
        A = np.zeros(steps)

        A[0] = self.A0

        if self.K*dt > 0.1:
            raise ValueError("Choose smaller dt: λ·dt must be ≪ 1 for Monte Carlo")

        for i in range(steps-1):
            A[i+1] = A[i] - self.K*A[i]*dt
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
        analytical = self.A0*np.exp(-self.K*t)

        df = pd.DataFrame({"Time":t,
                           "Numerical":A,
                           "Analytical":analytical})
        
        df.to_csv(filename,index=False)

        return df
    
    def half_life(self):
        return np.log(2)/self.K
    
    def simulate_monte_carlo(self,total_time=1000,dt=0.1,n_particle = 100000):
        steps = int(total_time/dt + 1)
        
        t = np.zeros(steps)
        survivors = np.zeros(steps)

        alive = np.ones(n_particle,dtype=bool)
        survivors[0] = n_particle

        decay_prob = self.K*dt

        if decay_prob > 0.1:
            raise ValueError("Choose smaller dt: λ·dt must be ≪ 1 for Monte Carlo")

        for i in range(steps-1):
            alive_idx = np.where(alive)[0]
            n_alive = alive_idx.size

            random_numbers = np.random.rand(n_alive)

            decayed = random_numbers < decay_prob

            alive[alive_idx[decayed]] = False

            survivors[i+1] = np.sum(alive)
            t[i+1] = t[i] + dt

        return t , survivors
    
    def plot_monte_carlo(self,total_time=1000,dt=0.1,n_particle = 100000):
        t_mc , survivors = self.simulate_monte_carlo(total_time=total_time,dt=dt,n_particle = n_particle)

        analytical = n_particle*np.exp(-self.K*t_mc)
        errors = np.sqrt(survivors)
        
        plt.plot(t_mc,survivors,label ="Monte carlo",alpha = 0.7)
        plt.plot(t_mc,analytical,label='Analytical')
        plt.errorbar(t_mc,survivors,yerr=errors,fmt='.',alpha = 0.3,label ='Monte Carlo ±√N')
        plt.xlabel("Time")
        plt.ylabel("Number of Nuclei")
        plt.title("Monte Carlo Radioactive Decay")
        plt.legend()
        plt.grid(True)
        plt.savefig("monte_carlo_decay.png")
        plt.show()


    def estimate_decay_const(self,t,survivors):
        mask = survivors>0
        t_fit = t[mask]
        N_fit = survivors[mask]

        log_N = np.log(N_fit)

        slope, intercept = np.polyfit(t_fit,log_N,1)
        lambda_est = -slope

        return lambda_est
    

class Detector:
    def __init__(self,t,survivors,efiiciency = 0.6,background_rate=2.0):
        self.t = t
        self.sv = survivors
        self.ef = efiiciency
        self.br = background_rate
        self.t_mid = None
        self.measured = None
        self.error = None


    def detector_response(self):
        true_decay = np.maximum( self.sv[:-1] - self.sv[1:],0).astype(int)

        detected = np.random.binomial(true_decay,self.ef)

        background = np.random.poisson(lam=self.br,size=detected.size)

        self.measured = detected + background

        self.error = np.sqrt(self.measured)

        self.t_mid = (self.t[:-1] + self.t[1:])*0.5

    
    def plot_detector_data(self):
        plt.errorbar(self.t_mid,self.measured,yerr=self.error,
                     fmt='o',capsize=3,label = 'Detector Data')
        plt.xlabel("Time (s)")
        plt.ylabel("Counts")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # 1. Setup Parameters
    INITIAL_CONC = 1.0     # For numerical method
    DECAY_CONSTANT = 0.05  # Lambda
    TOTAL_TIME = 100       # Duration of simulation
    N_PARTICLES = 10000    # For Monte Carlo
    
    print(f"--- Starting Radioactive Decay Simulation ---")
    print(f"True Decay Constant (λ): {DECAY_CONSTANT}")
    
    # 2. Initialize the RadioactiveDecay model
    decay_model = RadioactiveDecay(initial_concentration=INITIAL_CONC, decay_const=DECAY_CONSTANT)
    
    # 3. Test Numerical & Analytical Plotting
    # This will save 'Concentration_vs_Time.png'
    print("\n[1/4] Running Numerical/Analytical comparison...")
    decay_model.plot()
    
    # 4. Test Monte Carlo Simulation
    # This will save 'monte_carlo_decay.png'
    print("[2/4] Running Monte Carlo Simulation...")
    decay_model.plot_monte_carlo(total_time=TOTAL_TIME, dt=0.5, n_particle=N_PARTICLES)
    
    # 5. Test Parameter Estimation
    # Let's see if the slope of the log-plot matches our lambda
    t_mc, survivors = decay_model.simulate_monte_carlo(total_time=TOTAL_TIME, dt=0.5, n_particle=N_PARTICLES)
    est_lambda = decay_model.estimate_decay_const(t_mc, survivors)
    print(f"[3/4] Estimated λ from Monte Carlo: {est_lambda:.4f}")
    print(f"      Error: {abs(DECAY_CONSTANT - est_lambda):.4f}")
    
    # 6. Test Detector Class
    print("[4/4] Simulating Detector Response...")
    # Higher background rate and lower efficiency for a realistic challenge
    detector = Detector(t_mc, survivors, efiiciency=0.4, background_rate=5.0)
    detector.detector_response()
    detector.plot_detector_data()
    
    print("\n--- Simulation Complete ---")
