from utils.imports_statiques import *


class Sampler:
    """
    Generic sampler for diffusion / flow models.
    """
    def __init__(
        self,
        model,
        diffusion,
        schedule,
        device="cpu",
        T=3.0,
        sigma_data=1.0,
        input_size=None,  # ~[d, L]
        save_dir=None
    ):
        self.model = model
        self.diffusion = diffusion
        self.schedule = schedule
        self.device = device
        self.input_size = input_size
        self.T = T
        self.sigma_data = sigma_data
        self.save_dir = save_dir


    def score(self, x, t):
        pred = self.model(x, t)
        return self.diffusion.to_score(x, t, pred)


    def velocity(self, x, t):
        return self.model(x, t)
        

    def sample_sde_euler(
        self,
        batch_size,
        num_steps=1000,
        eps=1e-3,
        show_progress=True,
    ):
        """
        Euler–Maruyama reverse SDE sampler.
        """
        x = torch.randn(batch_size, *self.input_size, device=self.device)
        t_grid = torch.linspace(self.T, eps, num_steps, device=self.device)
        dt = t_grid[0] - t_grid[1]

        iterator = tqdm(t_grid) if show_progress else t_grid
        self.model.eval()
        with torch.no_grad():
            for t in iterator:
                bt = torch.full((batch_size,), t, device=self.device)

                f = self.schedule.drift(x, bt)
                g = self.schedule.diffusion(bt)
                score = self.score(x, bt)

                drift_rev = (-f + (g ** 2)[:, None, None] * score) * dt
                noise = torch.sqrt(dt) * g[:, None, None] * torch.randn_like(x)

                x = x + drift_rev + noise

        return x

    
    def sample_sde_heun(
        self,
        batch_size,
        num_steps=1000,
        eps=1e-3,
        show_progress=True,
    ):
        """
        Heun (predictor–corrector) reverse SDE sampler.
        """
        x = torch.randn(batch_size, *self.input_size, device=self.device)
        t_grid = torch.linspace(self.T, eps, num_steps, device=self.device)
        iterator = range(num_steps - 1)
        if show_progress:
            iterator = tqdm(iterator, desc="SDE Heun")

        self.model.eval()
        with torch.no_grad():
            for i in iterator:
                t = t_grid[i]
                t_next = t_grid[i + 1]
                dt = t - t_next

                bt = torch.full((batch_size,), t, device=self.device)
                bt_next = torch.full((batch_size,), t_next, device=self.device)

                f = self.schedule.drift(x, bt)
                g = self.schedule.diffusion(bt)
                score = self.score(x, bt)
                f_rev = -f + (g ** 2)[:, None, None] * score

                dW = torch.sqrt(dt) * torch.randn_like(x)
                x_pred = x + f_rev * dt + g[:, None, None] * dW

                f_next = self.schedule.drift(x_pred, bt_next)
                g_next = self.schedule.diffusion(bt_next)
                score_next = self.score(x_pred, bt_next)
                f_rev_next = -f_next + (g_next ** 2)[:, None, None] * score_next

                x = (x + 0.5 * (f_rev + f_rev_next) * dt
                       + 0.5 * (g[:, None, None] + g_next[:, None, None]) * dW)

        return x


    def sample_sde_pc_langevin(
        self,
        batch_size,
        num_steps=1000,
        eps=1e-3,
        snr=0.16,
        n_corrector=1,
        show_progress=True,
    ):
        """
        Langevin predictor–corrector reverse SDE sampler.
        """
        x = self.sigma_data * torch.randn(batch_size, *self.input_size, device=self.device)
        t_grid = torch.linspace(self.T, eps, num_steps, device=self.device)
        dt = t_grid[0] - t_grid[1]  # > 0

        iterator = tqdm(t_grid, desc="SDE PC") if show_progress else t_grid

        self.model.eval()
        with torch.no_grad():
            for t in iterator:
                bt = torch.full((batch_size,), t, device=self.device)

                # ----- Corrector: Langevin dynamics -----
                for _ in range(int(n_corrector)):
                    score = self.score(x, bt)  # (B,d,L)
                    grad = score.reshape(batch_size, -1)
                    grad_norm = torch.norm(grad, dim=-1).mean()

                    noise = torch.randn_like(x)
                    noise_norm = torch.norm(noise.reshape(batch_size, -1), dim=-1).mean()

                    # Song et al. step size: alpha = 2 * (snr * ||noise|| / ||grad||)^2
                    langevin_step = 2.0 * (snr * noise_norm / (grad_norm + 1e-12)) ** 2

                    x = x + langevin_step * score + torch.sqrt(2.0 * langevin_step) * noise

                # ----- Predictor: reverse SDE Euler–Maruyama -----
                f = self.schedule.drift(x, bt)
                g = self.schedule.diffusion(bt)
                score = self.score(x, bt)

                drift_rev = (-f + (g ** 2)[:, None, None] * score) * dt
                noise = torch.sqrt(dt) * g[:, None, None] * torch.randn_like(x)

                x = x + drift_rev + noise

        return x


    def sample_ode_pf(
        self,
        batch_size,
        num_steps=1000,
        eps=1e-3,
        show_progress=True,
    ):
        """
        Probability flow ODE sampler (Euler).
        """
        x = self.sigma_data * torch.randn(batch_size, *self.input_size, device=self.device)
        t_grid = torch.linspace(self.T, eps, num_steps, device=self.device)
        dt = t_grid[0] - t_grid[1]

        iterator = tqdm(t_grid) if show_progress else t_grid

        self.model.eval()
        with torch.no_grad():
            for t in iterator:
                bt = torch.full((batch_size,), t, device=self.device)

                f = self.schedule.drift(x, bt)
                g = self.schedule.diffusion(bt)
                score = self.score(x, bt)

                drift_ode = f - 0.5 * (g ** 2)[:, None, None] * score
                x = x - drift_ode * dt

        return x


    def sample_edm_heun(
        self,
        batch_size,
        num_steps=30,
    ):
        """
        EDM Heun sampler (sigma-based).
        Model must be a denoiser: model(x, sigma).
        """
        t = torch.linspace(1.0, 0.0, num_steps, device=self.device)
        sigmas = self.schedule.sigma(t)
        x = sigmas[0] * torch.randn(batch_size, *self.input_size, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for i in range(num_steps - 1):
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1]

                s = torch.full((batch_size,), sigma, device=self.device)
                s_next = torch.full((batch_size,), sigma_next, device=self.device)

                denoised = self.model(x, s)
                d = (x - denoised) / sigma

                x_euler = x + (sigma_next - sigma) * d

                denoised_next = self.model(x_euler, s_next)
                d_next = (x_euler - denoised_next) / sigma_next

                x = x + 0.5 * (sigma_next - sigma) * (d + d_next)

        return x

    
    def sample_ode_velocity_euler(
        self,
        batch_size,
        num_steps=1000,
        eps=1e-3,
        show_progress=True,
    ):
        x = self.sigma_data * torch.randn(batch_size, *self.input_size, device=self.device)
    
        t_grid = torch.linspace(self.T, eps, num_steps, device=self.device)
        dt = t_grid[0] - t_grid[1]  # > 0
        iterator = tqdm(t_grid) if show_progress else t_grid
    
        self.model.eval()
        with torch.no_grad():
            for t in iterator:
                bt = torch.full((batch_size,), t, device=self.device)
                v = self.velocity(x, bt)
                x = x - v * dt
    
        return x
    
    
    def sample_ode_velocity_heun(
        self,
        batch_size,
        num_steps=1000,
        eps=1e-3,
        show_progress=True,
    ):
        x = self.sigma_data * torch.randn(batch_size, *self.input_size, device=self.device)
    
        t_grid = torch.linspace(self.T, eps, num_steps, device=self.device)
        iterator = range(num_steps - 1)
        if show_progress:
            iterator = tqdm(iterator, desc="ODE Velocity Heun")
    
        self.model.eval()
        with torch.no_grad():
            for i in iterator:
                t = t_grid[i]
                t_next = t_grid[i + 1]
                dt = t - t_next  # > 0
    
                bt = torch.full((batch_size,), t, device=self.device)
                bt_next = torch.full((batch_size,), t_next, device=self.device)
    
                v = self.velocity(x, bt)
                x_euler = x - v * dt
    
                v_next = self.velocity(x_euler, bt_next)
                x = x - 0.5 * (v + v_next) * dt
    
        return x


    def sample_raw(
        self,
        method="sde_euler",
        batch_size=16,
        num_steps=1000,
        eps=1e-3,
        show_progress=True,
    ):
        if method == "sde_euler":
            return self.sample_sde_euler(
                batch_size=batch_size,
                num_steps=num_steps,
                eps=eps,
                show_progress=show_progress,
            )
    
        if method == "sde_heun":
            return self.sample_sde_heun(
                batch_size=batch_size,
                num_steps=num_steps,
                eps=eps,
                show_progress=show_progress,
            )

        if method == "sde_pc":
            return self.sample_sde_pc_langevin(
                batch_size=batch_size,
                num_steps=num_steps,
                eps=eps,
                show_progress=show_progress,
            )
    
        if method == "ode_pf":
            return self.sample_ode_pf(
                batch_size=batch_size,
                num_steps=num_steps,
                eps=eps,
                show_progress=show_progress,
            )
    
        if method == "edm_heun":
            return self.sample_edm_heun(
                batch_size=batch_size,
                num_steps=num_steps,
            )
    
        if method == "ode_euler":
            return self.sample_ode_velocity_euler(
                batch_size=batch_size,
                num_steps=num_steps,
                eps=eps,
                show_progress=show_progress,
            )
    
        if method == "ode_heun":
            return self.sample_ode_velocity_heun(
                batch_size=batch_size,
                num_steps=num_steps,
                eps=eps,
                show_progress=show_progress,
            )

        methods = [
            "sde_euler",
            "sde_heun",
            "ode_pf",
            "edm_heun",
            "ode_euler",
            "ode_heun",
        ]
        raise ValueError(
            f"Unknown sampling method: {method}. "
            f"Available methods are: {methods}"
        )


    def sample(
        self,
        series_label,
        method="sde_euler",
        batch_size=2000,
        num_steps=1000,
        eps=1e-3,
        show_progress=True,
        datagen_label="",
        force=False,
    ):
        filename = (
            f"{datagen_label}_{method}"
            f"_Mgen={batch_size}"
            f"_steps={num_steps}"
            f"_eps={eps}"
            f"_{series_label}.npy"
        )

        filename = filename[:251] + ".npy"
    
        save_path = os.path.join(self.save_dir, filename) if self.save_dir is not None else None
    
        if save_path is not None and os.path.exists(save_path) and not force:
            data = np.load(save_path)
            rel = os.path.basename(save_path).split("_", 1)[1]
            return torch.tensor(data, device=self.device), rel
    
        samples = self.sample_raw(
            method=method,
            batch_size=batch_size,
            num_steps=num_steps,
            eps=eps,
            show_progress=show_progress,
        )
    
        if save_path is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            np.save(save_path, samples.detach().cpu().numpy())
            rel = os.path.basename(save_path).split("_", 1)[1]
            return samples, rel
    
        return samples, None


