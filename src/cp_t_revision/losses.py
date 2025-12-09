import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ReweightLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, T, target):
        out_softmax = F.softmax(out, dim=1)
        noisy_prob = torch.matmul(T.T, out_softmax.unsqueeze(-1)).squeeze()
        eprob_clean = torch.gather(out_softmax, dim=-1, index=target.unsqueeze(1)).squeeze()
        eprob_noisy = torch.gather(noisy_prob, dim=-1, index=target.unsqueeze(1)).squeeze()
        weight = (eprob_clean / eprob_noisy).requires_grad_()
        cross_loss = F.cross_entropy(out, target, reduction="none")
        _loss = weight * cross_loss
        return torch.mean(_loss)


class ReweightCorrectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, T, correction, target):
        T = T.to(device="cuda")
        out_softmax = F.softmax(out, dim=1)
        T += correction
        noisy_prob = torch.matmul(T.T, out_softmax.unsqueeze(-1)).squeeze()
        eprob_clean = torch.gather(out_softmax, dim=-1, index=target.unsqueeze(1)).squeeze()
        eprob_noisy = torch.gather(noisy_prob, dim=-1, index=target.unsqueeze(1)).squeeze()
        weight = (eprob_clean / eprob_noisy).requires_grad_()
        cross_loss = F.cross_entropy(out, target, reduction="none")
        _loss = weight * cross_loss
        return torch.mean(_loss)


class Reweighting_Revision_Loss(nn.Module):
    def __init__(self, T):
        super(Reweighting_Revision_Loss, self).__init__()
        if isinstance(T, np.ndarray):
            T = torch.from_numpy(T).float()
        self.T = T

    def forward(self, out, correction, target):
        device = out.device
        self.T = self.T.to(device)
        loss = 0.0
        correction = correction.to(device)
        out_softmax = F.softmax(out, dim=1)
        for i in range(len(target)):
            temp_softmax = out_softmax[i]
            temp = out[i]
            temp = torch.unsqueeze(temp, 0).to(device)
            temp_softmax = torch.unsqueeze(temp_softmax, 0).to(device)
            temp_target = target[i]
            temp_target = torch.unsqueeze(temp_target, 0)
            pro1 = temp_softmax[:, target[i]]

            # Adjust transformation matrix T with the correction term and apply softmax for normalization
            T_result = self.T + correction
            T_result = F.softmax(T_result, dim=1)
            out_T = torch.matmul(T_result.t(), temp_softmax.t())
            out_T = out_T.t()
            pro2 = out_T[:, target[i]]
            beta = pro1 / (pro2 + 1e-15)

            cross_loss = F.cross_entropy(temp, temp_target)
            _loss = beta.detach() * cross_loss

            loss += _loss
        return loss / len(target)


class Reweighting_Revision_Loss_v2(nn.Module):
    def __init__(self, T):
        super(Reweighting_Revision_Loss_v2, self).__init__()
        if isinstance(T, np.ndarray):
            T = torch.from_numpy(T).float()
        self.T = T
        self.alpha = 0.01  # Scaling factor alpha for the correction term

    def forward(self, out, correction, target):
        device = out.device
        self.T = self.T.to(device)
        loss = 0.0
        correction = correction.to(device)
        out_softmax = F.softmax(out, dim=1)

        alpha = self.alpha  # Set alpha to the initialized scaling factor

        for i in range(len(target)):
            temp_softmax = out_softmax[i]
            temp = out[i]
            temp = torch.unsqueeze(temp, 0).to(device)
            temp_softmax = torch.unsqueeze(temp_softmax, 0).to(device)
            temp_target = target[i]
            temp_target = torch.unsqueeze(temp_target, 0)
            pro1 = temp_softmax[:, target[i]]

            # Adjust transformation matrix T with the scaled correction term
            T_result = self.T + (alpha * correction)
            out_T = torch.matmul(T_result.t(), temp_softmax.t())
            out_T = out_T.t()
            pro2 = out_T[:, target[i]]
            beta = pro1 / (pro2 + 1e-15)

            cross_loss = F.cross_entropy(temp, temp_target)
            _loss = beta.detach() * cross_loss
            loss += _loss
        return loss / len(target)
