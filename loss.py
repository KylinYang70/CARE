import torch
import torch.nn.functional as F
import torch.nn as nn


def cal_output(alpha, c):
    S_a = torch.sum(alpha, dim=1, keepdim=True)
    E_a = alpha - 1
    b = E_a / S_a
    u = (c / S_a).squeeze()
    return b, u


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, beta=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1
    B = KL(alp, c)
    return A + beta * B


def DS_Combin(alpha, classes):
    """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """

    def DS_Combin_two(alpha1, alpha2):
        """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    for v in range(len(alpha) - 1):
        if v == 0:
            alpha_a = DS_Combin_two(alpha[0], alpha[1])
        else:
            alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
    return alpha_a


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        eps = 1e-7
        preds = preds.clamp(eps, 1.0 - eps)  # Clamp predictions to avoid log(0)
        loss_y1 = -1 * self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds) * labels
        loss_y0 = -1 * (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds) * (1 - labels)
        loss = loss_y0 + loss_y1
        return torch.mean(loss)

class MultiFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(MultiFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        if labels.dim() == 1:
            labels = F.one_hot(labels, num_classes=preds.size(1)).float()
        total_loss = 0
        binary_focal_loss = BinaryFocalLoss(alpha=self.alpha, gamma=self.gamma)
        logits = F.softmax(preds, dim=1)
        nums = labels.shape[1]
        for i in range(nums):
            loss = binary_focal_loss(logits[:, i], labels[:, i])
            total_loss += loss
        return total_loss / nums