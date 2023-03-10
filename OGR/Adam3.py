import numpy as np
import torch
from torch.optim import Optimizer
import math

class AdamOGR(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self, parameters, lr=1e-1, inner_beta=0.4, gamma=0.5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= inner_beta:
            raise ValueError("Invalid inner beta value: {}".format(inner_beta))
        if not 0.0 <= gamma:
            raise ValueError("Invalid inner gamma value: {}".format(gamma))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, inner_beta=inner_beta, gamma=gamma, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamOGR, self).__init__(parameters, defaults)

    def __setstate__(self, state):
        super(AdamOGR, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                data = p.data

                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(data) + 1
                    # Exponential moving average of data values
                    state['data_exp_avg'] = torch.zeros_like(data)
                    # Exponential moving average of squared data values
                    state['data_exp_avg_sq'] = torch.zeros_like(data) + 1
                    state['m'] = torch.zeros_like(data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                data_exp_avg, data_exp_avg_sq = state['data_exp_avg'], state['data_exp_avg_sq']
                m = state['m']
                inner_beta = group['inner_beta']
                gamma = group['gamma']
                beta1, beta2 = group['betas']
                state['step'] += 1
                

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']



                # Decay the first and second moment running average coefficient
                #exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg.add_(torch.mul(inner_beta, torch.add(grad, exp_avg, alpha=-1))) #mg += β(ng-mg)
                grad_sq=torch.mul(torch.add(grad, exp_avg, alpha=-1), torch.add(grad, exp_avg, alpha=-1)) #dgg += β((ng-mg)²-dgg)
                exp_avg_sq.add_(torch.mul(inner_beta, torch.add(grad_sq, exp_avg_sq, alpha=-1))) #dgg += β((ng-mg)²-dgg)

                m.add_(torch.mul(gamma, torch.add(grad, m, alpha=-1))) #mg += γ(ng-m)

                #exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                data_exp_avg.add_(torch.mul(inner_beta, torch.add(data, data_exp_avg, alpha=-1))) #mΘ += β(Θ-mΘ)
                data_sq=torch.mul(torch.add(data, data_exp_avg, alpha=-1), torch.add(data, data_exp_avg, alpha=-1)) #dΘΘ += β((Θ-mΘ)²-dΘΘ)
                data_exp_avg_sq.add_(torch.mul(inner_beta, torch.add(data_sq, data_exp_avg_sq, alpha=-1))) #dΘΘ += β((Θ-mΘ)²-dΘΘ)

                v = torch.tensor([1.3])
                big_gamma = 0.1
                V = v+big_gamma*grad
                V = torch.nn.functional.normalize(V, dim=-1)
                pTheta = torch.tensordot(V, torch.add(data, data_exp_avg, alpha=-1), dims=1)
                print(pTheta)


                denom = torch.div(exp_avg_sq, data_exp_avg_sq).sqrt()
                div = 1.1
                cut = 0.1
                denom.mul_(div)
                denom[denom < cut] = cut
                perturb = torch.div(m, denom)
                # Projection
                wd_ratio = 1e-4
                """
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(
                        p,
                        grad,
                        perturb,
                        group['delta'],
                        group['wd_ratio'],
                        group['eps'],
                    )
                """
                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(
                        1 - group['lr'] * group['weight_decay'] * wd_ratio
                    )
                step_size = group['lr']
                p.data.add_(perturb, alpha=-step_size)
                #print(p.data)
                #wait = input("Press Enter to continue.")

                # denom = (exp_avg_sq / math.sqrt(bias_correction2)).add_(
                #     group['eps']
                # )
                
                # """
                # denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                #     group['eps']
                # )
                # """
                # step_size = group['lr'] / bias_correction1

                # perturb = ((data_exp_avg_sq / denom).sqrt()).mul_(grad)

                # # Projection
                # wd_ratio = 1e-2
                # """
                # if len(p.shape) > 1:
                #     perturb, wd_ratio = self._projection(
                #         p,
                #         grad,
                #         perturb,
                #         group['delta'],
                #         group['wd_ratio'],
                #         group['eps'],
                #     )
                # """
                # # Weight decay
                # if group['weight_decay'] > 0:
                #     p.data.mul_(
                #         1 - group['lr'] * group['weight_decay'] * wd_ratio
                #     )

                # p.data.add_(perturb, alpha=-step_size)

        
        return loss
