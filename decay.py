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

    def decay_const_detector(self):

        signal = self.measured - self.br
        signal = np.maximum(signal,1)
        corrected = signal/self.ef

        log_y = np.log(corrected)
        log_err = self.error/signal

        weights = 1/log_err**2

        slope , intercept = np.polyfit(
            self.t_mid,
            log_y,
            1,
            w=weights
        )

        lambda_fit = -slope

        cov = np.polyfit(
            self.t_mid,
            log_y,
            1,
            w=weights,
            cov=True
        )[1]

        lambda_error = np.sqrt(cov[0,0])

        plt.errorbar(
        self.t_mid,
        log_y,
        yerr=log_err,
        fmt='o',
        label="Detector data"
        )

        plt.plot(
        self.t_mid,
        intercept + slope * self.t_mid,
        label="Fit"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("ln(Counts)")
        plt.legend()
        plt.grid()
        plt.show()

        return lambda_fit , lambda_error



if __name__ == "__main__":
    # 1. Setup Parameters
    A0 = 50000          # Initial concentration/particles
    decay_constant = 0.05  # lambda (s^-1)
    total_time = 100    # Total duration of experiment
    
    print(f"--- Radioactive Decay Simulation ---")
    print(f"True Lambda: {decay_constant}")
    
    # 2. Instantiate and run Numerical/Analytical Comparison
    decay_sim = RadioactiveDecay(initial_concentration=A0, decay_const=decay_constant)
    print(f"Calculated Half-life: {decay_sim.half_life():.2f} s")
    
    # Run standard numerical plot
    # decay_sim.plot() 
    
    # 3. Monte Carlo Simulation
    print("\nRunning Monte Carlo simulation...")
    t_mc, survivors_mc = decay_sim.simulate_monte_carlo(
        total_time=total_time, 
        dt=0.5, 
        n_particle=A0
    )
    decay_sim.plot_monte_carlo(total_time=total_time, dt=0.5, n_particle=A0)
    
    # Estimate lambda from raw MC data
    est_lambda = decay_sim.estimate_decay_const(t_mc, survivors_mc)
    print(f"Estimated Lambda (Raw MC): {est_lambda:.4f}")

    # 4. Detector Simulation (Adding Noise and Efficiency)
    print("\nSimulating Detector response...")
    detector = Detector(
        t=t_mc, 
        survivors=survivors_mc, 
        efiiciency=0.7, 
        background_rate=5.0
    )
    detector.detector_response()
    
    # Plot detected counts over time
    detector.plot_detector_data()
    
    # 5. Regression / Curve Fitting
    print("\nFitting Detector Data to recover Lambda...")
    fit_lambda, fit_error = detector.decay_const_detector()
    
    print(f"\n--- Results ---")
    print(f"True Lambda:      {decay_constant:.4f}")
    print(f"Recovered Lambda: {fit_lambda:.4f} ± {fit_error:.4f}")
    
    # Check if the result is within 2 sigma
    z_score = abs(fit_lambda - decay_constant) / fit_error
    print(f"Z-score: {z_score:.2f}")
    if z_score < 2:
        print("Success: Recovered lambda is consistent with the true value.")
    else:
        print("Warning: Fit deviates significantly from true value.")