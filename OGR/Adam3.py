import numpy as np
import torch
from torch.optim import Optimizer
import math
import random

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
    def __init__(self, parameters, lr=1e-1, beta=0.5, gamma=0.9, div=1.5, cut=10, eps=1e-05):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta:
            raise ValueError("Invalid inner beta value: {}".format(beta))
        if not 0.0 <= gamma:
            raise ValueError("Invalid inner gamma value: {}".format(gamma))
        if not 0.0 <= div:
            raise ValueError("Invalid inner div value: {}".format(div))
        if not 0.0 <= cut:
            raise ValueError("Invalid inner cut value: {}".format(cut))

        defaults = dict(lr=lr, beta=beta, gamma=gamma, div=div, cut=cut, eps=eps)
        super(AdamOGR, self).__init__(parameters, defaults)

    def __setstate__(self, state):
        super(AdamOGR, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def eh(self,lm, div, cut):
        aux = abs(lm) * div
        aux [aux <= cut] = cut
        return 1/aux

    def ma(self,A, div, cut):
        #print(A)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        # npA = A.cpu().numpy()
        # npeigen = np.linalg.eig(npA)
        # e = torch.from_numpy(npeigen[0].real).to(device)
        # ev = torch.from_numpy(npeigen[1].real).to(device)
        # print(e,ev)
        eigen = torch.linalg.eig(A)
        e = eigen[0].real
        ev = eigen[1].real
        #print(e,ev)
        aux = self.eh(e, div, cut)
        aux = torch.diag(aux)
        aux = ev @ aux @ ev.T
        return aux

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                data = p.data.flatten()
                eps = group['eps']
                data += eps
                inner_grad = grad.flatten()

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(data)
                    state['mTheta'] = torch.zeros_like(data)
                    state['mThetaTheta'] = torch.eye(data.shape[0], device=device)
                    state['mg'] = torch.zeros_like(data)
                    state['mgTheta'] = torch.zeros_like(torch.eye(data.shape[0]),device=device)
                #     # Exponential moving average of gradient values
                #     state['exp_avg'] = torch.zeros_like(data)
                #     # Exponential moving average of squared gradient values
                #     state['exp_avg_sq'] = torch.zeros_like(data) + 1
                #     # Exponential moving average of data values
                #     state['data_exp_avg'] = torch.zeros_like(data)
                #     # Exponential moving average of squared data values
                #     state['data_exp_avg_sq'] = torch.zeros_like(data) + 1
                #     state['m'] = torch.zeros_like(data)
                m = state['m']
                step = state['step']
                gamma = group['gamma']
                m += gamma * (inner_grad-m)
                beta = group['beta']
                gamma = group['gamma']
                mTheta = state['mTheta']
                mThetaTheta = state['mThetaTheta']
                mTheta += beta * (data - mTheta)
                #print(data.shape)

                # print(mgTheta.get_device())
                mThetaTheta += beta * (torch.kron(data - mTheta, data - mTheta).reshape(data.shape[0], data.shape[0]) - mThetaTheta)
                mg =  state['mg']
                mg += beta * (inner_grad - mg)
                mgTheta = state['mgTheta']
                mgTheta += beta * (torch.kron(inner_grad - mg, data - mTheta).reshape(data.shape[0], data.shape[0]) - mgTheta)
                mgThetas = mgTheta + mgTheta.T
                # npmThetaTheta = mThetaTheta.cpu().numpy()
                # npeigen = np.linalg.eig(npmThetaTheta)
                # e = torch.from_numpy(npeigen[0].real).to(device)
                # ev = torch.from_numpy(npeigen[1].real).to(device)
                eigen = torch.linalg.eig(mThetaTheta)
                e = eigen[0].real
                ev = eigen[1].real
                #if not (step % 2):
                # e = torch.flip(e, [-1])
                # ev = torch.flip(ev, [-1])
                # e_aux = npeigen[0].real
                # ev_aux = npeigen[1].real
                # eigen = torch.linalg.eig(mThetaTheta)
                # e = torch.flip(eigen[0].real, [-1])
                # ev = torch.flip(eigen[1].real, [-1])
                #print(e, ev)
                # aux = torch.matmul(torch.matmul(ev,mgThetas),ev.T)
                # aux = (ev @ mgThetas)
                table = e.repeat(data.shape[0],1)
                # aux /= (table + table.T)
                # pH = torch.matmul(torch.matmul(ev.T,aux),ev)
                #nev = ev.numpy()
                #nmgThetas = mgThetas.numpy()
                pH = ev @ ((ev.T @ (mgThetas @ ev)) / (table + table.T)) @ ev.T
                #print(pH)
                div = group['div']
                cut = group['cut']
                aux = m @ self.ma(pH, div, cut)
                #print(aux)
                aux = torch.reshape(aux, p.data.shape)
                p.data -= aux
                state['step'] += 1
                #print(p.data[0])
                
                
                # exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                # data_exp_avg, data_exp_avg_sq = state['data_exp_avg'], state['data_exp_avg_sq']
                # m = state['m']
                # inner_beta = group['inner_beta']
                # gamma = group['gamma']
                # beta1, beta2 = group['betas']
                # state['step'] += 1
                

                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                

                
                # # Decay the first and second moment running average coefficient
                # #exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # exp_avg.add_(torch.mul(inner_beta, torch.add(grad, exp_avg, alpha=-1))) #mg += β(ng-mg)
                # grad_sq=torch.mul(torch.add(grad, exp_avg, alpha=-1), torch.add(grad, exp_avg, alpha=-1)) #dgg += β((ng-mg)²-dgg)
                # exp_avg_sq.add_(torch.mul(inner_beta, torch.add(grad_sq, exp_avg_sq, alpha=-1))) #dgg += β((ng-mg)²-dgg)

                # m.add_(torch.mul(gamma, torch.add(grad, m, alpha=-1))) #mg += γ(ng-m)

                # data_exp_avg.add_(torch.mul(inner_beta, torch.add(data, data_exp_avg, alpha=-1))) #mΘ += β(Θ-mΘ)
                # data_sq=torch.mul(torch.add(data, data_exp_avg, alpha=-1), torch.add(data, data_exp_avg, alpha=-1)) #dΘΘ += β((Θ-mΘ)²-dΘΘ)
                # data_exp_avg_sq.add_(torch.mul(inner_beta, torch.add(data_sq, data_exp_avg_sq, alpha=-1))) #dΘΘ += β((Θ-mΘ)²-dΘΘ)

                # v = torch.tensor([1.3])
                # big_gamma = 0.1
                # V = v+big_gamma*grad
                # V = torch.nn.functional.normalize(V, dim=-1)
                # pTheta = torch.tensordot(V, torch.add(data, data_exp_avg, alpha=-1), dims=1)
                # print(pTheta)


                # denom = torch.div(exp_avg_sq, data_exp_avg_sq).sqrt()
                # div = 1.1
                # cut = 0.1
                # denom.mul_(div)
                # denom[denom < cut] = cut
                # perturb = torch.div(m, denom)
                # # Projection
                # wd_ratio = 1e-4
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
                # step_size = group['lr']
                # p.data.add_(perturb, alpha=-step_size)

        
        return loss
