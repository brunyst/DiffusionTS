from utils.imports_statiques import *


class SimulateData:
    """
    Simulation of synthetic time series with on-disk caching.
    """
    def __init__(self, M, device, data_dir="data_sim", force=False, verbose=True):
        
        self.M = int(M)
        self.data_dir = str(data_dir)
        self.force = bool(force)
        self.device = device
        self.verbose = bool(verbose)

    # ---------------------------------------------------------
    # UTILS
    # ---------------------------------------------------------

    def _format_param(self, value):
        if isinstance(value, float):
            return f"{value:.3f}".rstrip("0").rstrip(".") or "0"
        if isinstance(value, (tuple, list, np.ndarray)):
            return "_".join(self._format_param(v) for v in value)
        return str(value)

    def _build_filepath(self, kind, params_dict, ext="npy"):
        parts = [f"{k}={self._format_param(v)}" for k, v in sorted(params_dict.items())]
        filename = f"{kind}_" + "_".join(parts) + f".{ext}"
        folder = os.path.join(self.data_dir, kind)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)

    def _load_or_generate(self, kind, params_dict, generator_fn):
        filepath = self._build_filepath(kind, params_dict)
        filename = os.path.basename(filepath)

        if os.path.exists(filepath) and not self.force:
            data = np.load(filepath, allow_pickle=False)
            if self.verbose:
                print(f"[SIMULATOR] Loaded cached data from {filepath} (shape={data.shape})")
            return data, filename, filepath

        if self.verbose:
            print(f"[SIMULATOR] Simulating data for {kind} with M={params_dict['M']}")

        data = generator_fn()
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        np.save(filepath, data)
        if self.verbose:
            print(f"[SIMULATOR] Saved simulated data to \n{filepath}")
        
        print()

        return data, filename, filepath


    # ---------------------------------------------------------
    # GENERIC MIXTURE (balanced or weighted)
    # ---------------------------------------------------------


    def simulate_mixture(
        self,
        components,
        kind="Mixture",
        weights=None,
        shuffle=True,
        cache_components=False,
    ):
        if not isinstance(components, (list, tuple)) or len(components) == 0:
            raise ValueError("[ERROR] components must be a non-empty list.")

        K = len(components)

        if weights is None:
            w = np.ones(K, dtype=np.float64) / K
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (K,):
                raise ValueError(f"[ERROR] weights must have length {K}, got {w.shape}")
            if np.any(w < 0):
                raise ValueError("[ERROR] weights must be non-negative")
            s = float(w.sum())
            if s <= 0:
                raise ValueError("[ERROR] weights must sum to a positive value")
            w = w / s

        # split M by weights, keep exact sum = self.M
        counts = np.floor(w * self.M).astype(int)
        # distribute remaining
        rem = self.M - int(counts.sum())
        if rem > 0:
            frac = (w * self.M) - counts
            idx = np.argsort(-frac)
            for i in range(rem):
                counts[idx[i % K]] += 1

        # build caching params
        params = {
            "M": self.M,
            "K": K,
            "weights": tuple(np.round(w, 6).tolist()),
            "shuffle": int(bool(shuffle)),
        }
        # include each component signature in params for unique cache
        for i, comp in enumerate(components):
            label = comp.get("label", None)
            kwargs = comp.get("kwargs", {})
            if label is None:
                raise ValueError(f"[ERROR] components[{i}] missing 'label'")
            params[f"c{i}_label"] = str(label)
            # keep kwargs shallowly serializable (numbers/tuples/strings)
            for kk, vv in sorted(kwargs.items()):
                params[f"c{i}_{kk}"] = vv

        def generator():
            chunks = []
            for i, comp in enumerate(components):
                Mi = int(counts[i])
                if Mi <= 0:
                    continue

                label = comp["label"]
                kwargs = comp.get("kwargs", {})

                sub = SimulateData(
                    M=Mi,
                    device=self.device,
                    data_dir=self.data_dir,
                    force=(not cache_components),  # force=True if we avoid component caching
                    verbose=self.verbose,
                )

                fn = getattr(sub, f"simulate_{label}", None)
                if fn is None:
                    raise ValueError(f"[ERROR] Unknown simulate_{label} for mixture component {i}")

                Xi, _ = fn(**kwargs)
                chunks.append(np.asarray(Xi))

            if len(chunks) == 0:
                raise ValueError("[ERROR] mixture produced no data (all counts were 0).")

            ref_shape = chunks[0].shape[1:]
            for j, arr in enumerate(chunks):
                if arr.shape[1:] != ref_shape:
                    raise ValueError(f"[ERROR] mixture shape mismatch: {arr.shape[1:]} vs {ref_shape} for component {j}")
        
            X = np.concatenate(chunks, axis=0)

            if shuffle:
                perm = np.random.permutation(X.shape[0])
                X = X[perm]

            # enforce exact M rows
            if X.shape[0] != self.M:
                # can happen if some component returns wrong first dimension
                raise ValueError(f"[ERROR] mixture assembled shape[0]={X.shape[0]} != M={self.M}")

            return X

        return self._load_or_generate(kind, params, generator)
        

    def simulate_lines(self, L, a_range=(-1, 1), b_range=(0, 1)):
        """
        Generates synthetic time series of straight lines (y = a*t + b) with support on [0, 1].

        :param L: [int]; number of time steps (the output has L points).
        :param a_range: [tuple(float, float)]; range [min, max] for slope a.
        :param b_range: [tuple(float, float)]; range [min, max] for intercept b.
        
        Return: [np.ndarray]; array of shape (M, L) of lines time series.
        """
        params = dict(
            M=self.M, L=L,
            amin=a_range[0], amax=a_range[1],
            bmin=b_range[0], bmax=b_range[1]
        )

        def generator():
            t = torch.linspace(0.0, 1.0, L, device=self.device)
            slopes = torch.empty(self.M, device=self.device).uniform_(a_range[0], a_range[1])
            intercepts = torch.empty(self.M, device=self.device).uniform_(b_range[0], b_range[1])
            X = slopes.unsqueeze(1) * t.unsqueeze(0) + intercepts.unsqueeze(1)
            X = X.cpu().numpy()
            return X

        return self._load_or_generate("lines", params, generator)
    

    def simulate_sines_1D(self, L, amp_range=(0.5, 1.5), freq_range=(1, 5), phase_range=(0, 2*np.pi)):
        """
        Generates synthetic time series consisting of sinusoidal signals with
        random amplitudes, frequencies, and phases.

        :param L: [int]; number of time steps (the output has L points).
        :param amp_range: [tuple(float, float)]; range [min, max] for amplitude A.
        :param freq_range: [tuple(float, float)]; range [min, max] for frequency f.
        :param phase_range: [tuple(float, float)]; range [min, max] for phase phi in radians.
        
        Return: [np.ndarray]; array of shape (M, L) of sine time series.
        """
        params = dict(
            M=self.M, L=L,
            ampmin=amp_range[0], ampmax=amp_range[1],
            freqmin=freq_range[0], freqmax=freq_range[1],
            phasemin=phase_range[0], phasemax=phase_range[1]
        )

        def generator():
            t = torch.linspace(0.0, 1.0, L, device=self.device)
            amps = torch.empty(self.M, device=self.device).uniform_(amp_range[0], amp_range[1])
            freqs = torch.empty(self.M, device=self.device).uniform_(freq_range[0], freq_range[1])
            phases = torch.empty(self.M, device=self.device).uniform_(phase_range[0], phase_range[1])
            phase_arg = 2 * np.pi * freqs.unsqueeze(1) * t.unsqueeze(0) + phases.unsqueeze(1)
            X = amps.unsqueeze(1) * torch.sin(phase_arg)
            X = X.cpu().numpy()
            return X

        return self._load_or_generate("sines1D", params, generator)


    def simulate_sines_2D(self, L, x_range, y_range, theta=None):
        """
        Generates 2D sine wave data: z = sin(x + y + theta)

        :param L: [int]; number of points per dimension
        :param x_range: [tuple(float, float)]; range for x values
        :param y_range: [tuple(float, float)]; range for y values
        :param theta: [float or None]; phase shift; if None, random for each series

        Return: [np.ndarray]; array of shape (M, L, L) of sine surfaces.
        """
        params = dict(
            M=self.M, L=L,
            xmin=x_range[0], xmax=x_range[1],
            ymin=y_range[0], ymax=y_range[1],
            theta=theta if theta is not None else "random"
        )

        def generator():
            x = torch.linspace(x_range[0], x_range[1], L, device=self.device)
            y = torch.linspace(y_range[0], y_range[1], L, device=self.device)
            Xg, Yg = torch.meshgrid(x, y, indexing="xy")
            base = Xg + Yg

            if theta is None:
                thetas = torch.empty(self.M, device=self.device).uniform_(0.0, 2.0 * np.pi)
                Z = torch.sin(base.unsqueeze(0) + thetas.view(-1, 1, 1))
            else:
                Z_single = torch.sin(base + theta)
                Z = Z_single.unsqueeze(0).expand(self.M, -1, -1)

            return Z.cpu().numpy()

        return self._load_or_generate("sines2D", params, generator)
    

    def simulate_linear_ODE(self, L, a0_range=(-1, 1), a1_range=(-1, 1), 
                            b_range=(-1, 1), x0_range=(-1, 1), dt=1/252):
        """
        Generates synthetic time series solving linear ODEs of the form: dx/dt = (a0+a1*t)*x+b
        where a0, a1, b and x0 are drawn independently from uniform ranges.

        :param L: [int]; number of time steps.
        :param a0_range: [tuple(float,float)]; range for a0.
        :param a1_range: [tuple(float,float)]; range for a1.
        :param b_range:  [tuple(float,float)]; range for b.
        :param x0_range: [tuple(float,float)]; range for x0.
        :param dt: [float]; time step.
        
        Return: [np.ndarray]; array of shape (M, L) of ODEs time series.
        """

        params = dict(
            M=self.M, L=L, dt=dt,
            a0min=a0_range[0], a0max=a0_range[1],
            a1min=a1_range[0], a1max=a1_range[1],
            bmin=b_range[0], bmax=b_range[1],
            x0min=x0_range[0], x0max=x0_range[1]
        )

        def generator():
            M = self.M
            t = torch.arange(L, device=self.device, dtype=torch.float32) * float(dt)
            a0 = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(a0_range[0], a0_range[1])
            a1 = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(a1_range[0], a1_range[1])
            b  = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(b_range[0],  b_range[1])
            x0 = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(x0_range[0], x0_range[1])

            X = torch.empty((M, L), device=self.device, dtype=torch.float32)
            X[:, 0] = x0

            dt_val = float(dt)
            for n in range(L - 1):
                drift = (a0 + a1 * t[n]) * X[:, n] + b
                X[:, n + 1] = X[:, n] + dt_val * drift
            return X.cpu().numpy()

        return self._load_or_generate("ODE", params, generator)
    

    def simulate_BM(self, L, sigma, dt=1/252, x0=0.0):
        """
        Generate Brownian Motion (BM) time series with variance parameter sigma.
        dX_t = sigma dW_t

        :param L: [int]; number of time steps.
        :param sigma: [float]; volatility parameter.
        :param dt: [float]; time step size.
        :param x0: [float]; initial value X_0.
        
        :return: [np.ndarray]; array of shape (M, L) of BM trajectories.
        """
        params = dict(
            M=self.M, L=L, dt=dt, x0=x0,
            sigma=sigma
        )

        def generator():
            M = self.M
            dt_val = float(dt)
            sigma_val = float(sigma)
            x0_val = float(x0)

            dX = sigma_val * dt_val**0.5 * torch.randn(M, L - 1, device=self.device)
            X0 = torch.full((M, 1), x0_val, device=self.device)
            X_increments = torch.cumsum(dX, dim=1)
            X = torch.cat([X0, X0 + X_increments], dim=1)
            return X.cpu().numpy()

        return self._load_or_generate("BM", params, generator)


    def simulate_GBM(self, L, mu, sigma, dt=1/252, x0=1.0):
        """
        Generates Geometric Brownian Motion (GBM) time series with fixed parameters (mu, sigma).

        GBM dynamics (risk-neutral style, in continuous time):
            dX_t = mu * X_t dt + sigma * X_t dW_t

        :param L: [int]; number of time steps.
        :param mu: [float]; drift parameter of the GBM.
        :param sigma: [float]; diffusion (volatility) coefficient.
        :param dt: [float]; time step size.
        :param x0: [float]; initial value at t=0.

        :return: [np.ndarray]; array of shape (M, L) of GBM time series.
        """
        params = dict(
            M=self.M, L=L, dt=dt, x0=x0,
            mu=mu, sigma=sigma
        )

        def generator():
            M = self.M
            dt_val = float(dt)
            mu_val = float(mu)
            sigma_val = float(sigma)
            x0_val = float(x0)

            Z = torch.randn(M, L - 1, device=self.device)
            drift = (mu_val - 0.5 * sigma_val ** 2) * dt_val
            vol = sigma_val * dt_val ** 0.5
            increments = drift + vol * Z
            factors = torch.exp(increments)
            X0 = torch.full((M, 1), x0_val, device=self.device)
            X_rest = X0 * torch.cumprod(factors, dim=1)
            X = torch.cat([X0, X_rest], dim=1)
            return X.cpu().numpy()

        return self._load_or_generate("GBM", params, generator)
    

    def simulate_OU(self, L, theta, mu, sigma, dt=1/252, x0=1):
        """
        Generates Ornstein–Uhlenbeck (OU) time series with fixed parameters (theta, mu, sigma).

        :param L: [int]; number of time steps.
        :param theta: [float]; mean-reversion speed.
        :param mu: [float]; long-run mean.
        :param sigma: [float]; diffusion coefficient.
        :param dt: [float]; time step size.
        :param x0: [float]; initial value at t=0.
        
        Return: [np.ndarray]; array of shape (M, L) of OU time series.
        """
        params = dict(
            M=self.M, L=L, dt=dt, x0=x0,
            theta=theta, mu=mu, sigma=sigma
        )

        def generator():
            M = self.M
            dt_val = float(dt)
            theta_val = float(theta)
            mu_val = float(mu)
            sigma_val = float(sigma)
            x0_val = float(x0)

            a = np.exp(-theta_val * dt_val)
            one_minus_a = 1.0 - a
            var = (sigma_val ** 2 / (2.0 * theta_val)) * (1.0 - np.exp(-2.0 * theta_val * dt_val))
            std = var ** 0.5

            X = torch.empty((M, L), device=self.device, dtype=torch.float32)
            X[:, 0] = x0_val

            Z = torch.randn(M, L - 1, device=self.device, dtype=torch.float32)
            for t in range(1, L):
                X[:, t] = a * X[:, t - 1] + mu_val * one_minus_a + std * Z[:, t - 1]

            return X.cpu().numpy()

        return self._load_or_generate("OU", params, generator)


    def simulate_OUmodes(
        self,
        L,
        theta1, mu1, sigma1,
        theta2, mu2, sigma2,
        dt=1/252,
        x0=1.0,
        shuffle=True,
    ):
        """
        Simule des données Ornstein–Uhlenbeck (OU) avec deux régimes ("modes") :
        - moitié des séries avec (theta1, mu1, sigma1)
        - autre moitié avec (theta2, mu2, sigma2)
    
        :param L: [int] nombre de pas de temps
        :param theta1, mu1, sigma1: paramètres du mode 1
        :param theta2, mu2, sigma2: paramètres du mode 2
        :param dt: [float] pas de temps
        :param x0: [float] valeur initiale
        :param shuffle: [bool] si True, mélange les trajectoires pour ne pas avoir
                        "mode1 puis mode2" dans l'ordre
    
        :return: (data, filename) où data est np.ndarray shape (M, L)
        """
    
        params = dict(
            M=self.M, L=L, dt=dt, x0=x0,
            theta1=theta1, mu1=mu1, sigma1=sigma1,
            theta2=theta2, mu2=mu2, sigma2=sigma2,
            shuffle=int(bool(shuffle)),
        )
    
        def _simulate_ou_fixed(M_local, theta, mu, sigma):
            dt_val = float(dt)
            theta_val = float(theta)
            mu_val = float(mu)
            sigma_val = float(sigma)
            x0_val = float(x0)
    
            a = np.exp(-theta_val * dt_val)
            one_minus_a = 1.0 - a
            var = (sigma_val ** 2 / (2.0 * max(theta_val, 1e-12))) * (1.0 - np.exp(-2.0 * theta_val * dt_val))
            std = var ** 0.5
    
            X = torch.empty((M_local, L), device=self.device, dtype=torch.float32)
            X[:, 0] = x0_val
    
            Z = torch.randn(M_local, L - 1, device=self.device, dtype=torch.float32)
            for t in range(1, L):
                X[:, t] = a * X[:, t - 1] + mu_val * one_minus_a + std * Z[:, t - 1]
            return X
    
        def generator():
            M1 = self.M // 2
            M2 = self.M - M1
    
            X1 = _simulate_ou_fixed(M1, theta1, mu1, sigma1)
            X2 = _simulate_ou_fixed(M2, theta2, mu2, sigma2)
    
            X = torch.cat([X1, X2], dim=0)
    
            if shuffle:
                perm = torch.randperm(self.M, device=self.device)
                X = X[perm]
    
            return X.cpu().numpy()
    
        return self._load_or_generate("OUmodes", params, generator)


    
    def simulate_OU_range(self, theta_range, mu_range, sigma_range, L, dt=1/252, x0=1):
        """
        Generates Ornstein–Uhlenbeck (OU) with range parameters.

        :param theta_range: [list of float]; range [min, max] for mean-reversion speed.
        :param mu_range: [list of float]; range [min, max] for long-run mean.
        :param sigma_range: [list of float]; range [min, max] for diffusion.
        :param L: [int]; number of time steps.
        :param dt: [float]; time step size.
        :param x0: [float]; initial value at t=0.
        
        Return: [np.ndarray]; array of shape (M, L) of OUrange time series.
        """
        params = dict(
            M=self.M, L=L, dt=dt, x0=x0,
            thetamin=theta_range[0], thetamax=theta_range[1],
            mumin=mu_range[0], mumax=mu_range[1],
            sigmamin=sigma_range[0], sigmamax=sigma_range[1]
        )

        def generator():
            M = self.M
            dt_val = float(dt)
            theta_min, theta_max = theta_range[0], theta_range[1]
            mu_min, mu_max = mu_range[0], mu_range[1]
            sigma_min, sigma_max = sigma_range[0], sigma_range[1]

            thetas = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(theta_min, theta_max)
            mus = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(mu_min, mu_max)
            sigmas = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(sigma_min, sigma_max)

            a = torch.exp(-thetas * dt_val)
            one_minus_a = 1.0 - a
            var = (sigmas ** 2 / (2.0 * thetas)) * (1.0 - torch.exp(-2.0 * thetas * dt_val))
            std = torch.sqrt(var)

            X = torch.empty((M, L), device=self.device, dtype=torch.float32)
            X[:, 0] = float(x0)

            
            Z = torch.randn(M, L - 1, device=self.device, dtype=torch.float32)
            for t in range(1, L):
                X[:, t] = a * X[:, t - 1] + mus * one_minus_a + std * Z[:, t - 1]

            return X.cpu().numpy()

        return self._load_or_generate("OUrange", params, generator)


    def simulate_CIR(self, L, theta, mu, sigma, dt=1/252, x0=1):
        """
        Generates Cox–Ingersoll–Ross (CIR) time series with fixed parameters
        (theta, mu, sigma), using exact non-central chi-square transitions.

        :param L: [int]; number of time steps (returns L+1 points).
        :param theta: [float]; mean-reversion speed (>0).
        :param mu:    [float]; long-term mean (>=0).
        :param sigma: [float]; volatility (>0).
        :param dt:    [float]; time step size.
        :param x0:    [float]; initial value (>=0).
        
        Return: [np.ndarray]; array of shape (M, L) of CIR time series.
        """

        params = dict(
            M=self.M, L=L, dt=dt, x0=x0,
            theta=theta, mu=mu, sigma=sigma
        )

        def generator():
            M = self.M
            kappa = max(theta, 1e-12)
            vol = max(sigma, 1e-12)
            mean = max(mu, 0.0)

            exp_kdt = np.exp(-kappa * dt)
            one_minus_exp = 1.0 - exp_kdt

            c = (vol**2) * one_minus_exp / (4.0 * kappa)
            d = 4.0 * kappa * mean / (vol**2)

            X = torch.zeros(M, L, device=self.device, dtype=torch.float32)
            X[:, 0] = max(x0, 0.0)

            for t in range(1, L):
                lam = (4.0 * kappa * exp_kdt * X[:, t - 1]) / (vol**2 * one_minus_exp)
                lam2 = lam / 2.0
                Np = torch.poisson(lam2)
                df = d + 2.0 * Np
                alpha = df / 2.0
                gamma = torch.distributions.Gamma(alpha, 0.5)
                Z = 2.0 * gamma.rsample()
                X[:, t] = c * Z

            return X.cpu().numpy()

        return self._load_or_generate("CIR", params, generator)


    def simulate_CIR_range(self, theta_range, mu_range, sigma_range, L, dt=1/252, x0=1):
        """
        Generates univariate time series from a Cox–Ingersoll–Ross (CIR) process
        using the exact non-central chi-squared sampling scheme, with parameters
        drawn uniformly in given ranges.

        :param theta_range: [tuple(float, float)]; range [min, max] for the mean-reversion rate θ (>0).
        :param mu_range:    [tuple(float, float)]; range [min, max] for the long-term mean μ (≥0).
        :param sigma_range: [tuple(float, float)]; range [min, max] for the volatility σ (>0).
        :param L:           [int]; length of each simulated time series (returns L+1 points).
        :param dt:          [float]; time step size (default: 1/252 for daily data).
        :param x0:          [float]; initial value at t=0 (must be non-negative, default: 1).

        Return: [np.ndarray]; array of shape (M, L) containing simulated CIR time series (non-negative).
        """

        params = dict(
            M=self.M, L=L, dt=dt, x0=x0,
            thetamin=theta_range[0], thetamax=theta_range[1],
            mumin=mu_range[0],       mumax=mu_range[1],
            sigmamin=sigma_range[0], sigmamax=sigma_range[1]
        )

        def generator():
            M = self.M
            dt_val = float(dt)

            theta_min, theta_max = theta_range[0], theta_range[1]
            mu_min, mu_max = mu_range[0], mu_range[1]
            sigma_min, sigma_max = sigma_range[0], sigma_range[1]

            thetas = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(theta_min, theta_max)
            mus = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(mu_min, mu_max)
            sigmas = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(sigma_min, sigma_max)

            kappa = torch.clamp(thetas, min=1e-12)
            vol = torch.clamp(sigmas, min=1e-12)
            mean = torch.clamp(mus, min=0.0)

            exp_kdt = torch.exp(-kappa * dt_val)
            one_minus_exp = 1.0 - exp_kdt

            c = (vol ** 2) * one_minus_exp / (4.0 * kappa)
            d = 4.0 * kappa * mean / (vol ** 2)

            X = torch.zeros(M, L, device=self.device, dtype=torch.float32)
            X[:, 0] = max(x0, 0.0)

            
            for t in range(1, L):
                lam = (4.0 * kappa * exp_kdt * X[:, t - 1]) / (vol ** 2 * one_minus_exp)
                lam2 = lam / 2.0
                Np = torch.poisson(lam2)
                df = d + 2.0 * Np
                alpha = df / 2.0
                gamma = torch.distributions.Gamma(alpha, 0.5)
                Z = 2.0 * gamma.rsample()
                X[:, t] = c * Z

            return X.cpu().numpy()

        return self._load_or_generate("CIRrange", params, generator)
    

    def simulate_Heston(self, kappa, theta, xi, rho, r, L, dt=1/252, S0=1.0, v0=1.0, observe_v=True):
        """
        Generates joint price/variance time series from the Heston model
        using fixed parameter values.

        :param r: [float]; risk-free rate.
        :param kappa: [float]; mean-reversion speed.
        :param theta: [float]; long-run variance.
        :param rho: [float]; correlation between Brownian motions (in [-1, 1]).
        :param xi: [float]; vol-of-vol.
        :param L: [int]; number of steps.
        :param dt: [float]; time step size.
        :param S0: [float]; initial price at t=0.
        :param v0: [float]; initial variance at t=0.

        Return: [np.ndarray]; array of shape (M, L, 2) of Heston 2D time series.
        """
        params = dict(
            M=self.M, L=L, dt=dt,
            kappa=kappa, theta=theta, xi=xi, rho=rho, r=r,
            S0=S0, v0=v0,
            observe_v=int(bool(observe_v))
        )

        def generator():
            M = self.M
            dt_val = float(dt)
            kappa_val = float(kappa)
            theta_val = float(theta)
            xi_val = float(xi)
            rho_val = float(rho)
            r_val = float(r)
            S0_val = float(S0)
            v0_val = float(v0)

            prices = torch.empty(M, L, device=self.device, dtype=torch.float32)
            vol = torch.empty(M, L, device=self.device, dtype=torch.float32)

            prices[:, 0] = S0_val
            v0_clamped = max(v0_val, 0.0)
            vol[:, 0] = v0_clamped

            sqrt_dt = dt_val ** 0.5
            rho_sq = max(0.0, 1.0 - rho_val ** 2) ** 0.5

            Z1 = torch.randn(M, L - 1, device=self.device, dtype=torch.float32)
            Z2 = torch.randn(M, L - 1, device=self.device, dtype=torch.float32)
            dW_v_all = sqrt_dt * Z1
            dW_S_all = sqrt_dt * (rho_val * Z1 + rho_sq * Z2)

            for t in range(1, L):
                v_prev = torch.clamp(vol[:, t - 1], min=0.0)
                sqrt_v = torch.sqrt(v_prev)
                v_t = v_prev + kappa_val * (theta_val - v_prev) * dt_val + xi_val * sqrt_v * dW_v_all[:, t - 1]
                v_t = torch.clamp(v_t, min=0.0)
                S_prev = prices[:, t - 1]
                S_t = S_prev * torch.exp((r_val - 0.5 * v_t) * dt_val + torch.sqrt(v_t) * dW_S_all[:, t - 1])
                vol[:, t] = v_t
                prices[:, t] = S_t

            if observe_v:
                out = torch.stack([prices, vol], dim=2)  # (M, L, 2)
            else:
                out = prices  # (M, L)
                
            return out.cpu().numpy()

        return self._load_or_generate("Heston", params, generator)
        

    def simulate_Heston_range(
        self,
        kappa_range,
        theta_range,
        xi_range,
        rho_range,
        r_range,
        L,
        dt=1/252,
        S0=1.0,
        v0=1.0,
        observe_v=True
    ):
        """
        Generates Heston time series with parameters sampled in given ranges.

        :param kappa_range: [list/tuple]; [kappa_min, kappa_max]
        :param theta_range: [list/tuple]; [theta_min, theta_max]
        :param xi_range:    [list/tuple]; [xi_min,    xi_max]
        :param rho_range:   [list/tuple]; [rho_min,   rho_max] (must be in [-1, 1])
        :param r_range:     [list/tuple]; [r_min,     r_max]
        :param L: [int]; number of steps.
        :param dt: [float]; time step size.
        :param S0: [float]; initial price at t=0 (same for all paths here).
        :param v0: [float]; initial variance at t=0 (same for all paths here).

        :return: np.ndarray of shape (M, L, 2) with [S_t, v_t] in last dimension.
        """

        params = dict(
            M=self.M, L=L, dt=dt,
            kappamin=kappa_range[0], kappamax=kappa_range[1],
            thetamin=theta_range[0], thetamax=theta_range[1],
            ximin=xi_range[0], ximax=xi_range[1],
            rhomin=rho_range[0], rhomax=rho_range[1],
            rmin=r_range[0], rmax=r_range[1],
            S0=S0, v0=v0,
            observe_v=int(bool(observe_v))
        )

        def generator():
            M = self.M
            dt_val = float(dt)
            S0_val = float(S0)
            v0_val = float(v0)

            kappas = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(kappa_range[0], kappa_range[1])
            thetas = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(theta_range[0], theta_range[1])
            xis    = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(xi_range[0],    xi_range[1])
            rhos   = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(rho_range[0],   rho_range[1])
            rs     = torch.empty(M, device=self.device, dtype=torch.float32).uniform_(r_range[0],     r_range[1])

            prices = torch.empty(M, L, device=self.device, dtype=torch.float32)
            vol    = torch.empty(M, L, device=self.device, dtype=torch.float32)

            prices[:, 0] = S0_val
            vol[:, 0] = max(v0_val, 0.0)

            sqrt_dt = dt_val ** 0.5
            rho_clamped = torch.clamp(rhos, -1.0, 1.0)
            rho_sq = torch.sqrt(torch.clamp(1.0 - rho_clamped ** 2, min=0.0))

            Z1 = torch.randn(M, L - 1, device=self.device, dtype=torch.float32)
            Z2 = torch.randn(M, L - 1, device=self.device, dtype=torch.float32)
            dW_v_all = sqrt_dt * Z1
            dW_S_all = sqrt_dt * (rho_clamped.unsqueeze(1) * Z1 + rho_sq.unsqueeze(1) * Z2)

            for t in range(1, L):
                v_prev = torch.clamp(vol[:, t - 1], min=0.0)
                sqrt_v = torch.sqrt(v_prev)
                v_t = v_prev + kappas * (thetas - v_prev) * dt_val + xis * sqrt_v * dW_v_all[:, t - 1]
                v_t = torch.clamp(v_t, min=0.0)
                S_prev = prices[:, t - 1]
                S_t = S_prev * torch.exp((rs - 0.5 * v_t) * dt_val + torch.sqrt(v_t) * dW_S_all[:, t - 1])
                vol[:, t] = v_t
                prices[:, t] = S_t

            
            if observe_v:
                out = torch.stack([prices, vol], dim=2)  # (M, L, 2)
            else:
                out = prices  # (M, L)
    
            return out.cpu().numpy()

        return self._load_or_generate("HestonRange", params, generator)


    def simulate_PDV2factor(
        self,
        beta0,
        beta1,
        beta2,
        lambda1,
        lambda2,
        a1,
        a2,
        L,
        dt=1/252,
        S0=1.0,
        R1_0=0.0,
        R2_0=0.0,
        sigma_floor=1e-6,
    ):
        
        params = dict(
            M=self.M, L=L, dt=dt, S0=S0,
            beta0=beta0, beta1=beta1, beta2=beta2,
            lambda1=lambda1, lambda2=lambda2,
            a1=a1, a2=a2,
            R1_0=R1_0, R2_0=R2_0,
            sigma_floor=sigma_floor,
        )

        def generator():
            M = self.M
            dt_val = float(dt)
            sqrt_dt = dt_val ** 0.5

            beta0_t = float(beta0)
            beta1_t = float(beta1)
            beta2_t = float(beta2)
            lam1_t  = float(lambda1)
            lam2_t  = float(lambda2)
            a1_t    = float(a1)
            a2_t    = float(a2)

            S0_t  = float(S0)
            R1_t0 = float(R1_0)
            R2_t0 = float(R2_0)
            sig_floor = float(sigma_floor)

            prices = torch.empty(M, L, device=self.device, dtype=torch.float32)
            sigmas = torch.empty(M, L, device=self.device, dtype=torch.float32)

            # state
            logS = torch.full((M,), np.log(max(S0_t, 1e-12)), device=self.device, dtype=torch.float32)
            R1   = torch.full((M,), R1_t0, device=self.device, dtype=torch.float32)
            R2   = torch.full((M,), max(R2_t0, 0.0), device=self.device, dtype=torch.float32)

            prices[:, 0] = torch.exp(logS)

            def sigma_fn(R1_, R2_):
                # sigma = beta0 + beta1 R1 + beta2 sqrt(R2)
                # clamp to keep positivity and avoid sqrt issues
                R2c = torch.clamp(R2_, min=0.0)
                sig = beta0_t + beta1_t * R1_ + beta2_t * torch.sqrt(R2c)
                return torch.clamp(sig, min=sig_floor)

            sig = sigma_fn(R1, R2)
            sigmas[:, 0] = sig

            Z = torch.randn(M, L - 1, device=self.device, dtype=torch.float32)

            for t in range(1, L):
                sig = sigma_fn(R1, R2)

                dW = sqrt_dt * Z[:, t - 1]
                dS_over_S = sig * dW  # consistent with dS/S = sigma dW (Ito)

                # log-price update (exact for dS/S = sigma dW when sigma frozen over dt)
                logS = logS - 0.5 * (sig ** 2) * dt_val + dS_over_S

                # R1, R2 updates
                R1 = R1 + (-lam1_t * R1) * dt_val + a1_t * dS_over_S
                R2 = R2 + (-lam2_t * R2) * dt_val + a2_t * (sig ** 2) * dt_val
                R2 = torch.clamp(R2, min=0.0)

                prices[:, t] = torch.exp(logS)
                sigmas[:, t] = sig

            # (M, L, 2): [S, sigma]
            return torch.stack([prices, sigmas], dim=2).cpu().numpy()

        return self._load_or_generate("PDV2factor", params, generator)
    
    
    # ---------- LINES ----------

    @staticmethod
    def get_params_lines(L, a_range, b_range):
        amin, amax = a_range
        bmin, bmax = b_range

        params = {
            "L":    (r"$L$",       L),
            "amin": (r"$a_{min}$", amin),
            "amax": (r"$a_{max}$", amax),
            "bmin": (r"$b_{min}$", bmin),
            "bmax": (r"$b_{max}$", bmax),
        }

        # 1er param estimé : a_range
        params1 = {
            "amin": (r"$a_{min}$", amin),
            "amax": (r"$a_{max}$", amax),
        }

        # 2e param estimé : b_range
        params2 = {
            "bmin": (r"$b_{min}$", bmin),
            "bmax": (r"$b_{max}$", bmax),
        }

        return params, params1, params2

    # ---------- SINES 1D ----------

    @staticmethod
    def get_params_sines_1D(L, amp_range, freq_range, phase_range):
        ampmin, ampmax = amp_range
        freqmin, freqmax = freq_range
        phasemin, phasemax = phase_range

        params = {
            "L":        (r"$L$",          L),
            "ampmin":   (r"$A_{min}$",    ampmin),
            "ampmax":   (r"$A_{max}$",    ampmax),
            "freqmin":  (r"$f_{min}$",    freqmin),
            "freqmax":  (r"$f_{max}$",    freqmax),
            "phasemin": (r"$\phi_{min}$", phasemin),
            "phasemax": (r"$\phi_{max}$", phasemax),
        }

        # 1er param : A_range
        params1 = {
            "ampmin": (r"$A_{min}$", ampmin),
            "ampmax": (r"$A_{max}$", ampmax),
        }

        # 2e param : f_range
        params2 = {
            "freqmin": (r"$f_{min}$", freqmin),
            "freqmax": (r"$f_{max}$", freqmax),
        }

        # 3e param : phi_range
        params3 = {
            "phasemin": (r"$\phi_{min}$", phasemin),
            "phasemax": (r"$\phi_{max}$", phasemax),
        }

        return params, params1, params2, params3

    # ---------- SINES 2D ----------

    @staticmethod
    def get_params_sines_2D(L, x_range, y_range, theta):
        xmin, xmax = x_range
        ymin, ymax = y_range

        params = {
            "L":    (r"$L$",   L),
            "xmin": (r"$x_{min}$", xmin),
            "xmax": (r"$x_{max}$", xmax),
            "ymin": (r"$y_{min}$", ymin),
            "ymax": (r"$y_{max}$", ymax),
            "theta": (r"$\theta$", theta if theta is not None else "random"),
        }

        # Ici pas de range pour theta, on laisse tel quel
        params1 = {
            "theta": (r"$\theta$", theta if theta is not None else "random"),
        }

        return params, params1

    # ---------- LINEAR ODE ----------

    @staticmethod
    def get_params_linear_ODE(L, dt, a0_range, a1_range, b_range, x0_range):
        a0min, a0max = a0_range
        a1min, a1max = a1_range
        bmin, bmax   = b_range
        x0min, x0max = x0_range

        params = {
            "L":     (r"$L$",      L),
            "dt":    (r"$dt$",     dt),
            "a0min": (r"$a_{0,min}$", a0min),
            "a0max": (r"$a_{0,max}$", a0max),
            "a1min": (r"$a_{1,min}$", a1min),
            "a1max": (r"$a_{1,max}$", a1max),
            "bmin":  (r"$b_{min}$",   bmin),
            "bmax":  (r"$b_{max}$",   bmax),
            "x0min": (r"$x_{0,min}$", x0min),
            "x0max": (r"$x_{0,max}$", x0max),
        }

        # 1er param : a0_range
        params1 = {
            "a0min": (r"$a_{0,min}$", a0min),
            "a0max": (r"$a_{0,max}$", a0max),
        }

        # 2e param : a1_range
        params2 = {
            "a1min": (r"$a_{1,min}$", a1min),
            "a1max": (r"$a_{1,max}$", a1max),
        }

        # 3e param : b_range
        params3 = {
            "bmin": (r"$b_{min}$", bmin),
            "bmax": (r"$b_{max}$", bmax),
        }

        params4 = {
            "x0min": (r"$x0_{min}$", x0min),
            "x0max": (r"$x0_{max}$", x0max),
        }

        return params, params1, params2, params3, params4

    # ---------- BM (pas de range) ----------

    @staticmethod
    def get_params_BM(L, dt, x0, sigma):
        params = {
            "L":     (r"$L$",      L),
            "dt":    (r"$dt$",     dt),
            "x0":    (r"$x_0$",    x0),
            "sigma": (r"$\sigma$", sigma),
        }

        params1 = {
            "sigma": (r"$\sigma$", sigma),
        }

        return params, params1

    # ---------- GBM (pas de range) ----------

    @staticmethod
    def get_params_GBM(L, dt, x0, mu, sigma):
        params = {
            "L":     (r"$L$",      L),
            "dt":    (r"$dt$",     dt),
            "x0":    (r"$x_0$",    x0),
            "mu":    (r"$\mu$",    mu),
            "sigma": (r"$\sigma$", sigma),
        }

        params1 = {
            "mu": (r"$\mu$", mu),
        }

        params2 = {
            "sigma": (r"$\sigma$", sigma),
        }

        return params, params1, params2

    # ---------- OU ----------

    @staticmethod
    def get_params_OU(L, dt, x0, theta, mu, sigma):
        params = {
            "L":     (r"$L$",      L),
            "dt":    (r"$d_{\tau}$",     dt),
            "x0":    (r"$x_0$",    x0),
            "theta": (r"$\theta$", theta),
            "mu":    (r"$\mu$",    mu),
            "sigma": (r"$\sigma$", sigma),
        }

        params1 = {
            "theta": (r"$\theta$", theta),
        }

        params2 = {
            "mu":    (r"$\mu$",    mu),
        }

        params3 = {
            "sigma": (r"$\sigma$", sigma),
        }

        return params, params1, params2, params3


    # ---------- OU modes ----------

    @staticmethod
    def get_params_OUmodes(L, dt, x0, theta1, theta2, mu1, mu2, sigma1, sigma2):

        params = {
            "L":     (r"$L$",      L),
            "dt":    (r"$d_{\tau}$",     dt),
            "x0":    (r"$x_0$",    x0),
            "theta1": (r"$\theta_1$", theta1),
            "mu1":    (r"$\mu_1$",    mu1),
            "sigma1": (r"$\sigma_1$", sigma1),
            "theta2": (r"$\theta_2$", theta2),
            "mu2":    (r"$\mu_2$",    mu2),
            
            
        }

        params1 = {
            "theta1": (r"$\theta_1$", theta1),
            "theta2": (r"$\theta_2$", theta2),
        }

        params2 = {
            "mu1":    (r"$\mu_1$",    mu1),
            "mu2":    (r"$\mu_2$",    mu2),
        }

        params3 = {
            "sigma1": (r"$\sigma_1$", sigma1),
            "sigma2": (r"$\sigma_2$", sigma2),
        }

        return params, params1, params2, params3

    # ---------- OU RANGE ----------

    @staticmethod
    def get_params_OU_range(L, dt, x0, theta_range, mu_range, sigma_range):
        thetamin, thetamax = theta_range
        mumin, mumax       = mu_range
        sigmamin, sigmamax = sigma_range

        params = {
            "L":        (r"$L$",       L),
            "dt":       (r"$dt$",      dt),
            "x0":       (r"$x_0$",     x0),
            "thetamin": (r"$\theta_{min}$", thetamin),
            "thetamax": (r"$\theta_{max}$", thetamax),
            "mumin":    (r"$\mu_{min}$",    mumin),
            "mumax":    (r"$\mu_{max}$",    mumax),
            "sigmamin": (r"$\sigma_{min}$", sigmamin),
            "sigmamax": (r"$\sigma_{max}$", sigmamax),
        }

        # 1er param : theta_range
        params1 = {
            "thetamin": (r"$\theta_{min}$", thetamin),
            "thetamax": (r"$\theta_{max}$", thetamax),
        }

        # 2e param : mu_range
        params2 = {
            "mumin": (r"$\mu_{min}$", mumin),
            "mumax": (r"$\mu_{max}$", mumax),
        }

        # 3e param : sigma_range
        params3 = {
            "sigmamin": (r"$\sigma_{min}$", sigmamin),
            "sigmamax": (r"$\sigma_{max}$", sigmamax),
        }

        return params, params1, params2, params3

    # ---------- CIR ----------

    @staticmethod
    def get_params_CIR(L, dt, x0, theta, mu, sigma):
        params = {
            "L":     (r"$L$",      L),
            "dt":    (r"$dt$",     dt),
            "x0":    (r"$x_0$",    x0),
            "theta": (r"$\kappa$", theta),
            "mu":    (r"$\mu$",    mu),
            "sigma": (r"$\sigma$", sigma),
        }

        params1 = {
            "theta": (r"$\kappa$", theta),
        }

        params2 = {
            "mu": (r"$\mu$", mu),
        }

        params3 = {
            "sigma": (r"$\sigma$", sigma),
        }

        return params, params1, params2, params3

    # ---------- CIR RANGE ----------

    @staticmethod
    def get_params_CIR_range(L, dt, x0, theta_range, mu_range, sigma_range):
        thetamin, thetamax = theta_range
        mumin, mumax       = mu_range
        sigmamin, sigmamax = sigma_range

        params = {
            "L":        (r"$L$",       L),
            "dt":       (r"$dt$",      dt),
            "x0":       (r"$x_0$",     x0),
            "thetamin": (r"$\kappa_{min}$", thetamin),
            "thetamax": (r"$\kappa_{max}$", thetamax),
            "mumin":    (r"$\mu_{min}$",    mumin),
            "mumax":    (r"$\mu_{max}$",    mumax),
            "sigmamin": (r"$\sigma_{min}$", sigmamin),
            "sigmamax": (r"$\sigma_{max}$", sigmamax),
        }

        params1 = {
            "thetamin": (r"$\kappa_{min}$", thetamin),
            "thetamax": (r"$\kappa_{max}$", thetamax),
        }

        params2 = {
            "mumin": (r"$\mu_{min}$", mumin),
            "mumax": (r"$\mu_{max}$", mumax),
        }

        params3 = {
            "sigmamin": (r"$\sigma_{min}$", sigmamin),
            "sigmamax": (r"$\sigma_{max}$", sigmamax),
        }

        return params, params1, params2, params3

    # ---------- HESTON ----------

    @staticmethod
    def get_params_Heston(L, dt, kappa, theta, xi, rho, r, S0, v0, observe_v):
        params = {
            "L":     (r"$L$",        L),
            "dt":    (r"$dt$",       dt),
            "kappa": (r"$\kappa$",   kappa),
            "theta": (r"$\theta$",   theta),
            "xi":    (r"$\xi$",      xi),
            "rho":   (r"$\rho$",     rho),
            "r":     (r"$r$",        r),
            "S0":    (r"$S_0$",      S0),
            "v0":    (r"$v_0$",      v0),
            "observe_v": (r"Observe $v_t$", int(bool(observe_v))),
        }

        # Souvent, on estime surtout (kappa, theta, xi)
        params1 = {
            "kappa": (r"$\kappa$", kappa),
        }

        params2 = {
            "theta": (r"$\theta$", theta),
        }

        params3 = {
            "xi": (r"$\xi$", xi),
        }

        params4 = {
            "rho": (r"$\rho$", rho),
        }

        params5 = {
            "r": (r"$r$", r),
        }

        return params, params1, params2, params3, params4, params5


    # ---------- HESTON RANGE ----------

    @staticmethod
    def get_params_Heston_range(L, dt, kappa_range, theta_range, xi_range, rho_range, r_range, S0, v0, observe_v):
        kappamin, kappamax = kappa_range
        thetamin, thetamax = theta_range
        ximin, ximax       = xi_range
        rhomin, rhomax     = rho_range
        rmin, rmax         = r_range

        params = {
            "L":        (r"$L$",        L),
            "dt":       (r"$dt$",       dt),
            "kappamin": (r"$\kappa_{min}$", kappamin),
            "kappamax": (r"$\kappa_{max}$", kappamax),
            "thetamin": (r"$\theta_{min}$", thetamin),
            "thetamax": (r"$\theta_{max}$", thetamax),
            "ximin":    (r"$\xi_{min}$",    ximin),
            "ximax":    (r"$\xi_{max}$",    ximax),
            "rhomin":   (r"$\rho_{min}$",   rhomin),
            "rhomax":   (r"$\rho_{max}$",   rhomax),
            "rmin":     (r"$r_{min}$",      rmin),
            "rmax":     (r"$r_{max}$",      rmax),
            "S0":       (r"$S_0$",         S0),
            "v0":       (r"$v_0$",         v0),
            "observe_v": (r"Observe $v_t$", int(bool(observe_v))),
        }

        params1 = {
            "kappamin": (r"$\kappa_{min}$", kappamin),
            "kappamax": (r"$\kappa_{max}$", kappamax),
        }

        params2 = {
            "thetamin": (r"$\theta_{min}$", thetamin),
            "thetamax": (r"$\theta_{max}$", thetamax),
        }

        params3 = {
            "ximin": (r"$\xi_{min}$", ximin),
            "ximax": (r"$\xi_{max}$", ximax),
        }

        params4 = {
            "rhomin": (r"$\rho_{min}$", rhomin),
            "rhomax": (r"$\rho_{max}$", rhomax),
        }

        params5 = {
            "rmin": (r"$r_{min}$", rmin),
            "rmax": (r"$r_{max}$", rmax),
        }

        return params, params1, params2, params3, params4, params5


    @staticmethod
    def get_params_PDV2factor(
        L,
        dt,
        S0,
        beta0,
        beta1,
        beta2,
        lambda1,
        lambda2,
        a1,
        a2,
        R1_0,
        R2_0,
        sigma_floor,
    ):
        params = {
            "L":           (r"$L$", L),
            "dt":          (r"$dt$", dt),
            "S0":          (r"$S_0$", S0),
            "beta0":       (r"$\beta_0$", beta0),
            "beta1":       (r"$\beta_1$", beta1),
            "beta2":       (r"$\beta_2$", beta2),
            "lambda1":     (r"$\lambda_1$", lambda1),
            "lambda2":     (r"$\lambda_2$", lambda2),
            "a1":          (r"$a_1$", a1),
            "a2":          (r"$a_2$", a2),
            "R1_0":        (r"$R_{1,0}$", R1_0),
            "R2_0":        (r"$R_{2,0}$", R2_0),
            "sigma_floor": (r"$\sigma_{\min}$", sigma_floor),
        }
    
        params1 = {
            "beta0": (r"$\beta_0$", beta0),
            "beta1": (r"$\beta_1$", beta1),
            "beta2": (r"$\beta_2$", beta2),
        }
    
        params2 = {
            "lambda1": (r"$\lambda_1$", lambda1),
            "lambda2": (r"$\lambda_2$", lambda2),
        }
    
        params3 = {
            "a1": (r"$a_1$", a1),
            "a2": (r"$a_2$", a2),
        }
    
        params4 = {
            "R1_0": (r"$R_{1,0}$", R1_0),
            "R2_0": (r"$R_{2,0}$", R2_0),
        }
    
        return params, params1, params2, params3, params4


    @staticmethod
    def get_params_mixture(components, weights=None, shuffle=True, kind="Mixture"):
        if not isinstance(components, (list, tuple)) or len(components) == 0:
            raise ValueError("[ERROR] components must be a non-empty list.")
    
        K = len(components)
    
        # Normalize weights (same logic as simulate_mixture)
        if weights is None:
            w = np.ones(K, dtype=np.float64) / K
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (K,):
                raise ValueError(f"[ERROR] weights must have length {K}, got {w.shape}")
            if np.any(w < 0):
                raise ValueError("[ERROR] weights must be non-negative")
            s = float(w.sum())
            if s <= 0:
                raise ValueError("[ERROR] weights must sum to a positive value")
            w = w / s
    
        # --- main params dict (for full display)
        params = {
            "kind":    (r"$\mathrm{kind}$", str(kind)),
            "K":       (r"$K$", K),
            "weights": (r"$w$", tuple(np.round(w, 6).tolist())),
            "shuffle": (r"$\mathrm{shuffle}$", int(bool(shuffle))),
        }
    
        # --- component-level params (one dict per component)
        comp_params = []
        for i, comp in enumerate(components):
            label = comp.get("label", None)
            kwargs = comp.get("kwargs", {}) or {}
            if label is None:
                raise ValueError(f"[ERROR] components[{i}] missing 'label'")
    
            d = {
                f"c{i}_label": (rf"$c_{{{i}}}$", str(label)),
                f"c{i}_w":     (rf"$w_{{{i}}}$", float(w[i])),
            }
            for kk, vv in sorted(kwargs.items()):
                d[f"c{i}_{kk}"] = (rf"$c_{{{i}}}:{kk}$", vv)
            comp_params.append(d)
    
        # return like other get_params_*: (global params, comp1, comp2, ...)
        return (params, *comp_params)
