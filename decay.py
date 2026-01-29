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

        fig , ax = plt.subplots(1,2,figsize = (8,8))

        ax[0].plot(t,A,label='Concentration (Numerical)')
        ax[1].plot(t,exact,label = 'Concentration (Analytical)')
        ax[0].set_xlabel('Time (s)')
        ax[1].set_xlabel('Time (s)')

        ax[0].set_ylabel('Concentration')
        ax[1].set_ylabel('Concentration')

        ax[0].legend()
        ax[1].legend()

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

        fig , ax = plt.subplots(2,1,figsize = (8,8))
        
        ax[0].plot(t_mc,survivors,label ="Monte carlo",alpha = 0.7)
        ax[1].plot(t_mc,analytical,label='Analytical')
        ax[0].errorbar(t_mc,survivors,yerr=errors,fmt='.',alpha = 0.3,label ='Monte Carlo ±√N')
        ax[0].set_xlabel("Time")
        ax[1].set_xlabel("Time")

        ax[0].set_ylabel("Number of Nuclei")
        ax[1].set_ylabel("Number of Nuclei")

        plt.title("Monte Carlo Radioactive Decay")
        ax[0].legend()
        ax[1].legend()

        ax[0].grid(True)
        ax[1].grid(True)

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
    # --- 1. Initialization & Basic Simulation ---
    print("--- Testing Basic Numerical Simulation ---")
    N0 = 1000
    lam = 0.05
    rd = RadioactiveDecay(initial_concentration=N0, decay_const=lam)
    
    # Check half-life calculation: ln(2)/lambda
    expected_hl = np.log(2) / lam
    print(f"Calculated Half-life: {rd.half_life():.2f}s (Expected: {expected_hl:.2f}s)")
    
    # Run deterministic simulation
    A, t = rd.simulate(total_time=100, dt=0.01)
    print(f"Numerical end concentration: {A[-1]:.2f}")
    
    # --- 2. Monte Carlo Testing ---
    print("\n--- Testing Monte Carlo Simulation ---")
    # Using a larger dt and fewer particles for a faster test run
    t_mc, survivors = rd.simulate_monte_carlo(total_time=50, dt=0.5, n_particle=50000)
    
    # Estimate lambda from the MC data to see if it recovers our input lam
    lam_est = rd.estimate_decay_const(t_mc, survivors)
    print(f"Estimated λ from MC: {lam_est:.4f} (Original λ: {lam})")
    
    # Test MC Plotting (Checks syntax/matplotlib logic)
    rd.plot_monte_carlo(total_time=50, dt=0.5, n_particle=10000)

    # --- 3. Detector & Data Analysis ---
    print("\n--- Testing Detector Response ---")
    # Using the MC results as input for the detector
    det = Detector(t_mc, survivors, efiiciency=0.7, background_rate=5.0)
    det.detector_response()
    
    # Check if detector data exists
    if det.measured is not None:
        print(f"Total counts detected: {np.sum(det.measured)}")
        print(f"Average background noise: {det.br} counts/bin")
    
    # Attempt to recover the decay constant from "noisy" detector data
    fit_lam, fit_err = det.decay_const_detector()
    print(f"Detector-fitted λ: {fit_lam:.4f} ± {fit_err:.4f}")
    
    # Validation check: Is the true lambda within 3 standard deviations?
    z_score = abs(fit_lam - lam) / fit_err
    if z_score < 3:
        print(f"Success: Fit is consistent with truth (Z-score: {z_score:.2f})")
    else:
        print(f"Warning: Fit deviates from truth (Z-score: {z_score:.2f})")

    # --- 4. Data Persistence ---
    df = rd.save_data("test_decay_data.csv")
    print(f"\nSaved {len(df)} rows to test_decay_data.csv")
    print("Test Suite Complete.")