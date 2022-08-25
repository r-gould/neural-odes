from functools import partial
import torch

from .operable_list import OperableList

class ODEAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z0, dynamics, solver, params):
        
        z1 = solver.solve(z0, dynamics, t0=0, t1=1)
        ctx.dynamics = dynamics
        ctx.solver = solver
        ctx.params = params
        ctx.save_for_backward(z1)
        return z1

    @staticmethod
    def backward(ctx, dL_dz1):
        
        dynamics = ctx.dynamics
        solver = ctx.solver
        params = ctx.params
        z1, = ctx.saved_tensors

        dL_dp1 = torch.zeros_like(params)
        s0 = OperableList([z1, dL_dz1, dL_dp1])

        @torch.set_grad_enabled(True)
        def aug_dynamics(state, t, dynamics):

            params = [w for w in dynamics.parameters()]
            z, a, _ = state
            z = z.detach()
            z.requires_grad = True
            out = dynamics(z, t)

            a_df_dz, *a_df_dp = torch.autograd.grad(out, [z] + params, grad_outputs=a,
                                                allow_unused=True, retain_graph=True)
            param_arr = [torch.flatten(p) for p in a_df_dp]
            a_df_dp = torch.cat(param_arr)

            return OperableList([out, -a_df_dz, -a_df_dp])


        aug_dynamics_f = partial(aug_dynamics, dynamics=dynamics)
        [z0, dL_dz0, dL_dp] = solver.solve(s0, aug_dynamics_f, 1, 0)
        return dL_dz0, None, None, dL_dp