from utils.imports_statiques import *


class Schedule:
    def alpha(self, t):
        raise NotImplementedError

    def sigma(self, t):
        raise NotImplementedError

    def alpha_dot(self, t):
        raise NotImplementedError

    def sigma_dot(self, t):
        raise NotImplementedError


class VPOUSchedule(Schedule):
    """
    Pure Ornstein–Uhlenbeck VP schedule.
    """
    def __init__(self, eps=1e-12):
        self.eps = float(eps)

    def alpha(self, t):
        return torch.exp(-t)

    def sigma(self, t):
        return torch.sqrt((1.0 - torch.exp(-2.0 * t)).clamp(min=self.eps))

    def alpha_dot(self, t):
        return -torch.exp(-t)

    def sigma_dot(self, t):
        sigma = self.sigma(t).clamp(min=self.eps)
        return torch.exp(-2.0 * t) / sigma

    def drift(self, x, t):
        return -x

    def diffusion(self, t):
        return math.sqrt(2.0)

'''
class VPSchedule(Schedule):
    """
    VP schedule (DDPM / VP-SDE) — sans normalisation par T.
    """
    def __init__(self, betamin=0.1, betamax=20.0, T=3.0, eps=1e-12):
        self.betamin = float(betamin)
        self.betamax = float(betamax)
        self.T = float(T)
        self.eps = float(eps)

    def beta(self, t):
        # Plus de (t / T)
        return self.betamin + (self.betamax - self.betamin) * t

    def integral_beta(self, t):  # ∫_0^t beta(s) ds
        # Cohérent avec beta(t) ci-dessus
        return self.betamin * t + 0.5 * (self.betamax - self.betamin) * t**2

    def bar_alpha(self, t):
        return torch.exp(-self.integral_beta(t))

    def alpha(self, t):
        return torch.sqrt(self.bar_alpha(t).clamp(min=self.eps))

    def sigma(self, t):
        return torch.sqrt((1.0 - self.bar_alpha(t)).clamp(min=self.eps))

    def alpha_dot(self, t):
        return -0.5 * self.beta(t) * self.alpha(t)

    def sigma_dot(self, t):
        sigma = self.sigma(t).clamp(min=self.eps)
        return 0.5 * self.beta(t) * self.bar_alpha(t) / sigma

    def drift(self, x, t):
        beta_t = self.beta(t)[:, None, None]
        return -0.5 * beta_t * x

    def diffusion(self, t):
        return torch.sqrt(self.beta(t))
'''

class VPSchedule(Schedule):
    def __init__(self, betatype="linear", betamin=1e-4, betamax=2e-2):
        self.betatype = betatype
        self.betamin = betamin
        self.betamax = betamax

    def beta(self, t):
        if self.betatype == "linear":
            return self.betamin + t * (self.betamax - self.betamin)
        if self.betatype == "cosine":
            return 0.5 * (1 - torch.cos(np.pi * t))
        raise ValueError("Unknown betatype")

    #def integral_beta(self, t, steps=100):
    #    ts = torch.linspace(0, t, steps, device=t.device)
    #    bt = self.beta(ts)
    #    return torch.trapz(bt, ts)

    def integral_beta(self, t, steps=100):
        if self.betatype == "linear":
            return self.betamin * t + 0.5 * (self.betamax - self.betamin) * t**2
        elif self.betatype == "cosine":
            # Si tu veux vraiment ce cas, on peut approx avec une intégrale numérique
            # mais il faut la vectoriser proprement (et ce sera plus coûteux).
            raise NotImplementedError("integral_beta for betatype='cosine' not implemented")
        else:
            raise ValueError("Unknown betatype")

    def alpha(self, t):
        return torch.exp(-0.5 * self.integral_beta(t))

    def sigma(self, t):
        a = self.alpha(t)
        return torch.sqrt(1 - a**2)

    def diffusion(self, t):
        return torch.sqrt(self.beta(t))

    def drift(self, x, t):
        beta_t = self.beta(t)[:, None, None]
        return -0.5 * beta_t * x

class SubVPSchedule(Schedule):
    """
    SubVP schedule (Score-SDE).
    """
    def __init__(self, betamin=0.1, betamax=20.0, T=3.0, eps=1e-12):
        self.betamin = float(betamin)
        self.betamax = float(betamax)
        self.T = float(T)
        self.eps = float(eps)

    def beta(self, t):
        return self.betamin + (self.betamax - self.betamin) * (t / self.T)

    def integral_beta(self, t):
        b0 = self.betamin
        b1 = self.betamax
        return b0 * t + 0.5 * (b1 - b0) * (t * t) / self.T

    def bar_alpha(self, t):
        return torch.exp(-self.integral_beta(t))

    def alpha(self, t):
        return torch.sqrt(self.bar_alpha(t).clamp(min=self.eps))

    def sigma(self, t):
        return (1.0 - self.bar_alpha(t)).clamp(min=self.eps)

    def alpha_dot(self, t):
        return -0.5 * self.beta(t) * self.alpha(t)

    def sigma_dot(self, t):
        return self.beta(t) * self.bar_alpha(t)

    def drift(self, x, t):
        beta_t = self.beta(t)[:, None, None]
        return -0.5 * beta_t * x

    def diffusion(self, t):
        g2 = self.beta(t) * (1.0 - self.bar_alpha(t))
        return torch.sqrt(g2.clamp(min=self.eps))


class CosineVPSchedule(Schedule):  # X_t = alpha(t) X_0 + sigma(t) eps
    """
    Cosine VP schedule (Improved DDPM)
    """
    def __init__(self, s=0.008, T=3.0, eps=1e-12):
        self.s = float(s)
        self.T = float(T)
        self.eps = float(eps)

    def _f(self, t):
        u = (t / self.T + self.s) / (1.0 + self.s)
        return u.clamp(0.0, 1.0)

    def bar_alpha(self, t):
        u = self._f(t)
        return torch.cos(0.5 * math.pi * u) ** 2

    def alpha(self, t):
        return torch.sqrt(self.bar_alpha(t).clamp(min=self.eps))

    def sigma(self, t):
        return torch.sqrt((1.0 - self.bar_alpha(t)).clamp(min=self.eps))

    def bar_alpha_dot(self, t):
        u = self._f(t)
        theta = 0.5 * math.pi * u
        dtheta_dt = (0.5 * math.pi) * (1.0 / (self.T * (1.0 + self.s)))
        return -(torch.sin(2.0 * theta)) * dtheta_dt

    def alpha_dot(self, t):
        ba = self.bar_alpha(t).clamp(min=self.eps)
        a = torch.sqrt(ba)
        ba_dot = self.bar_alpha_dot(t)
        return 0.5 * ba_dot / a

    def sigma_dot(self, t):
        ba = self.bar_alpha(t)
        ba_dot = self.bar_alpha_dot(t)
        sig = torch.sqrt((1.0 - ba).clamp(min=self.eps))
        return -0.5 * ba_dot / sig

    def beta(self, t):
        ba = self.bar_alpha(t).clamp(min=self.eps)
        ba_dot = self.bar_alpha_dot(t)
        return (-ba_dot / ba).clamp(min=0.0)
    
    def drift(self, x, t):
        beta_t = self.beta(t)[:, None, None]
        return -0.5 * beta_t * x
    
    def diffusion(self, t):
        return torch.sqrt(self.beta(t))


class VESchedule(Schedule):
    """
    VE schedule (Score-SDE / NCSN).
    """
    def __init__(self, sigmamin=0.02, sigmamax=50.0, T=3.0, eps=1e-12):
        self.sigmamin = float(sigmamin)
        self.sigmamax = float(sigmamax)
        self.T = float(T)
        self.eps = float(eps)
        self._log_ratio = math.log(self.sigmamax / self.sigmamin)

    def alpha(self, t):
        return torch.ones_like(t)

    def sigma(self, t):
        return self.sigmamin * torch.exp((t / self.T) * self._log_ratio)

    def alpha_dot(self, t):
        return torch.zeros_like(t)

    def sigma_dot(self, t):
        return self.sigma(t) * (self._log_ratio / self.T)

    def drift(self, x, t):
        return torch.zeros_like(x)
    
    def diffusion(self, t):
        g2 = 2.0 * self.sigma(t) * self.sigma_dot(t)
        return torch.sqrt(g2.clamp(min=self.eps))


class EDMSchedule(Schedule):  # X_t = alpha(t) X_0 + sigma(t) eps
    """
    EDM (Karras) rho-schedule for the noise level sigma(t).
    """
    def __init__(self, sigmamin=0.002, sigmamax=80.0, rho=7.0, T=3.0, eps=1e-12):
        self.sigmamin = float(sigmamin)
        self.sigmamax = float(sigmamax)
        self.rho = float(rho)
        self.T = float(T)
        self.eps = float(eps)

        self._sigmamin_r = self.sigmamin ** (1.0 / self.rho)
        self._sigmamax_r = self.sigmamax ** (1.0 / self.rho)
        self._delta = self._sigmamax_r - self._sigmamin_r

    def alpha(self, t):
        return torch.ones_like(t)

    def sigma(self, t):
        u = self._sigmamin_r + (t / self.T) * self._delta
        return u.clamp(min=self.eps) ** self.rho

    def alpha_dot(self, t):
        return torch.zeros_like(t)

    def sigma_dot(self, t):
        u = self._sigmamin_r + (t / self.T) * self._delta
        u = u.clamp(min=self.eps)
        u_dot = self._delta / self.T
        return self.rho * (u ** (self.rho - 1.0)) * u_dot

    def drift(self, x, t):
        return torch.zeros_like(x)
    
    def diffusion(self, t):
        g2 = 2.0 * self.sigma(t) * self.sigma_dot(t)
        return torch.sqrt(g2.clamp(min=self.eps))


class GaussianFlowSchedule(Schedule):
    """
    Gaussian Flow Matching schedule (trigonometric).
    """
    def __init__(self, T=1.0, eps=1e-12):
        self.T = float(T)
        self.eps = float(eps)
        self._c = 0.5 * math.pi / self.T

    def alpha(self, t):
        return torch.cos(self._c * t)

    def sigma(self, t):
        return torch.sin(self._c * t).clamp(min=self.eps)

    def alpha_dot(self, t):
        return -self._c * torch.sin(self._c * t)

    def sigma_dot(self, t):
        return self._c * torch.cos(self._c * t)


class LogSNRSchedule(Schedule):
    """
    Log-SNR linear schedule.
    Very stable, unifies VP / cosine-like schedules.
    """
    def __init__(self, lambdamin=-20.0, lambdamax=20.0, T=3.0, eps=1e-12):
        self.lambdamin = float(lambdamin)
        self.lambdamax = float(lambdamax)
        self.T = float(T)
        self.eps = float(eps)
        self._lambda_dot = (self.lambdamin - self.lambdamax) / self.T

    def log_snr(self, t):
        return self.lambdamax + (self.lambdamin - self.lambdamax) * (t / self.T)

    def alpha(self, t):
        a2 = torch.sigmoid(self.log_snr(t))
        return torch.sqrt(a2.clamp(min=self.eps))

    def sigma(self, t):
        s2 = torch.sigmoid(-self.log_snr(t))
        return torch.sqrt(s2.clamp(min=self.eps))

    def alpha_dot(self, t):
        a = self.alpha(t)
        s2 = torch.sigmoid(-self.log_snr(t))
        return 0.5 * a * s2 * self._lambda_dot
    
    def sigma_dot(self, t):
        s = self.sigma(t)
        a2 = torch.sigmoid(self.log_snr(t))
        return -0.5 * s * a2 * self._lambda_dot
