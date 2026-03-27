from utils.imports_statiques import *


class GaussianDiffusion:

    def __init__(self, schedule, loss_type="epsilon", T=3.0, eps=1e-3, sigma_data=None):
        self.schedule = schedule
        self.T = float(T)
        self.eps = float(eps)
        self.loss_type = loss_type
        self.sigma_data = float(sigma_data)
    
    
    def loss(self, model, x_0):
        B = x_0.shape[0]
        device = x_0.device
        
        t = torch.rand(B, device=device) * (self.T - self.eps) + self.eps  # t ~ Uniform(eps, T) ~ [B]
        alpha_t = self.schedule.alpha(t)[:, None, None]  # alpha(t) ~ [B, 1, 1]
        sigma_t = self.schedule.sigma(t)[:, None, None]  # sigma(t) ~ [B, 1, 1]
        epsilon = torch.randn_like(x_0)  # epsilon ~ N(0, I_{dxL}) ~ [B, d, L]

        x_t = alpha_t * x_0 + sigma_t * epsilon  # ~ [B, d, L]   
        pred = model(x_t, t)  # ~ neural net [B, d, L]
        
        if self.loss_type == "epsilon":
            target = epsilon
            per_elem = (pred - target) ** 2
            w = torch.ones_like(t)  # ~ [B]
            
        elif self.loss_type == "data":
            target = x_0
            per_elem = (pred - target) ** 2
            w = 1.0 / (sigma_t[:, 0, 0]**2 + self.sigma_data**2)  # ~ [B]
            
        elif self.loss_type == "velocity":
            a_dot = self.schedule.alpha_dot(t)[:, None, None] # ~ [B, 1, 1]
            s_dot = self.schedule.sigma_dot(t)[:, None, None] # ~ [B, 1, 1]
            target = a_dot * x_0 + s_dot * epsilon
            per_elem = (pred - target) ** 2
            w = torch.ones_like(t)  # ~ [B]

        elif self.loss_type == "score":
            per_elem = (sigma_t * pred + epsilon) ** 2
            w = torch.ones_like(t)  # ~ [B]

        
        per_sample = per_elem.sum(dim=(1, 2))  # ~ [B]
        loss = (w * per_sample).mean()  # scalar
        return loss


    def to_score(self, x_t, t, pred):
        alpha_t = self.schedule.alpha(t)[:, None, None]  # alpha(t) ~ [B, 1, 1]
        sigma_t = self.schedule.sigma(t)[:, None, None]  # sigma(t) ~ [B, 1, 1]
        sigma2 = (sigma_t * sigma_t).clamp(min=1e-12)

        if self.loss_type == "score":
            return pred # already a score network

        if self.loss_type == "epsilon":
            return -pred / sigma2

        if self.loss_type == "data":
            return (alpha_t * pred - x_t) / sigma2

        if self.loss_type == "velocity":
            a_dot = self.schedule.alpha_dot(t)[:, None, None]  # [B, 1, 1]
            s_dot = self.schedule.sigma_dot(t)[:, None, None]  # [B, 1, 1]
            alpha_safe = alpha_t.clamp(min=1e-12)
            sigma_safe = sigma_t.clamp(min=1e-12)
            denom = (sigma2 * ((a_dot/alpha_safe) - (s_dot/sigma_safe))).clamp(min=1e-12)
            return (pred - (a_dot/alpha_safe) * x_t) / denom

        return pred


