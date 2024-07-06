# coding: utf-8

"""
@File     :Criterion.py
@Author    :XieJing
@Date      :2021/9/1
@Copyright :AI
@Desc      :
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):

    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()


class Focal_loss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(Focal_loss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size()[0], logit.size()[1], -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# class FocalLoss(nn.Module):
#     def __init__(self, class_num, alpha=0.20, gamma=1.5, use_alpha=False, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.class_num = class_num
#         self.alpha = alpha
#         self.gamma = gamma
#         if use_alpha:
#             self.alpha = torch.tensor(alpha).cuda()
#             # self.alpha = torch.tensor(alpha)
#
#         self.softmax = nn.Softmax(dim=1)
#         self.use_alpha = use_alpha
#         self.size_average = size_average
#
#     def forward(self, pred, target):
#
#         prob = self.softmax(pred.view(-1, self.class_num))
#         prob = prob.clamp(min=0.0001, max=1.0)
#
#         target_ = torch.zeros(target.size(0), self.class_num)
#         target_ = target_.to(device)
#         # target_ = torch.zeros(target.size(0),self.class_num)
#         # target_.scatter_(1, target.view(-1, 1).long(), 1.)
#         target_.scatter_(1, target.long(), 1.)
#
#         if self.use_alpha:
#             batch_loss = - self.alpha.double() * torch.pow(1 - prob,
#                                                            self.gamma).double() * prob.log().double() * target_.double()
#         else:
#             batch_loss = - torch.pow(1 - prob, self.gamma).double() * prob.log().double() * target_.double()
#
#         batch_loss = batch_loss.sum(dim=1)
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#
#         return loss


class PriorMultiLabelSoftMarginLoss(nn.Module):
    def __init__(self, prior=None, num_labels=None, reduction="mean", eps=1e-9, tau=1.0):
        """PriorCrossEntropy
        categorical-crossentropy-with-prior
        urls: [通过互信息思想来缓解类别不平衡问题](https://spaces.ac.cn/archives/7615)
        args:
            prior: List<float>, prior of label, 先验知识.  eg. [0.6, 0.2, 0.1, 0.1]
            num_labels: int, num of labels, 类别数.  eg. 10
            reduction: str, Specifies the reduction to apply to the output, 输出形式.
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            eps: float, Minimum of maths, 极小值.  eg. 1e-9
            tau: float, weight of prior in loss, 先验知识的权重, eg. ``1.0``
        returns:
            Tensor of loss.
        examples:
        # >>> loss = PriorCrossEntropy(prior)(logits, label)
        """
        super(PriorMultiLabelSoftMarginLoss, self).__init__()
        self.loss_mlsm = torch.nn.MultiLabelSoftMarginLoss(reduction=reduction)
        if not prior: prior = np.array([1 / num_labels for _ in range(num_labels)])  # 如果不存在就设置为num
        if type(prior) == list: prior = np.array(prior)
        self.log_prior = torch.tensor(np.log(prior + eps)).unsqueeze(0)
        self.eps = eps
        self.tau = tau

    def forward(self, logits, labels):
        # 使用与输入label相同的device
        logits = logits + self.tau * self.log_prior.to(labels.device)
        loss = self.loss_mlsm(logits, labels)
        return loss


class MultiLabelCircleLoss(nn.Module):
    def __init__(self, reduction="mean", inf=1e12):
        """CircleLoss of MultiLabel, 多个目标类的多标签分类场景，希望“每个目标类得分都不小于每个非目标类的得分”
        多标签分类的交叉熵(softmax+crossentropy推广, N选K问题), LSE函数的梯度恰好是softmax函数
        让同类相似度与非同类相似度之间拉开一定的margin。
          - 使同类相似度比最大的非同类相似度更大。
          - 使最小的同类相似度比最大的非同类相似度更大。
          - 所有同类相似度都比所有非同类相似度更大。
        urls: [将“softmax+交叉熵”推广到多标签分类问题](https://spaces.ac.cn/archives/7359)
        args:
            reduction: str, Specifies the reduction to apply to the output, 输出形式.
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            inf: float, Minimum of maths, 无穷大.  eg. 1e12
        returns:
            Tensor of loss.
        examples:
            >>> label, logits = [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 1, 1, 0], [1, 0, 0, 1],]
            >>> label, logits = torch.tensor(label).float(), torch.tensor(logits).float()
            >>> loss = MultiLabelCircleLoss()(logits, label)
        """
        super(MultiLabelCircleLoss, self).__init__()
        self.reduction = reduction
        self.inf = inf  # 无穷大

    def forward(self, logits, labels):
        logits = (1 - 2 * labels) * logits  # <3, 4>
        logits_neg = logits - labels * self.inf  # <3, 4>
        logits_pos = logits - (1 - labels) * self.inf  # <3, 4>
        zeros = torch.zeros_like(logits[..., :1])  # <3, 1>
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)  # <3, 5>
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)  # <3, 5>
        neg_loss = torch.logsumexp(logits_neg, dim=-1)  # <3, >
        pos_loss = torch.logsumexp(logits_pos, dim=-1)  # <3, >
        loss = neg_loss + pos_loss
        if "mean" == self.reduction:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def Facal_loss_criterion(y_pred, y_true, weight=None, alpha=0.25, gamma=2):
    sigmoid_p = nn.Sigmoid()(y_pred)
    zeros = torch.zeros_like(sigmoid_p)
    pos_p_sub = torch.where(y_true > zeros, y_true - sigmoid_p, zeros)
    neg_p_sub = torch.where(y_true > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = -alpha * (pos_p_sub ** gamma) * torch.log(torch.clamp(sigmoid_p, 1e-8, 1.0)) - (1 - alpha) * (
            neg_p_sub ** gamma) * torch.log(torch.clamp(1.0 - sigmoid_p, 1e-8, 1.0))
    return per_entry_cross_ent.sum()


if __name__ == '__main__':
    pred = np.array([[-0.4089, -1.2471, 0.5907, 0.1257],
                     [-0.4897, -0.8267, -0.7349, 0.6547],
                     [0.5241, -0.1246, -0.4751, -0.1284]])
    label = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [1, 0, 1, 0]])
    pred = torch.from_numpy(pred).float()
    label = torch.from_numpy(label).float()
    weights = [0.1995515618217051, 0.23923624222389323, 0.30392200269000713, 0.25729019326439445]
    fl = FocalLoss(4)

    print(fl(pred, label))

    fl2 = PriorMultiLabelSoftMarginLoss(num_labels=4)
    print(fl2(pred, label))

    fl3 = MultiLabelCircleLoss()
    print(fl3(pred, label))
