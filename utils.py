import logging
import random
import torch
from torchvision import transforms
import os
import tensorflow as tf
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics._classification import multilabel_confusion_matrix, _check_set_wise_labels, _prf_divide

model_config = {
    "model_name": "vgg16",
    "image_channels": 3,
    "image_size": 224,
    "pretrained": False,
    "num_classes": 4,
    # Training settings
    "batch_size": 16,
    "epochs": 300,
    "seed": 18888,
    "logging_steps": 20,
    "earlystop_patience": 200,
    # LR scheduler
    "lr": 1e-4,
    "lr_scheduler_name": 'cosine',
    "decay_epochs": 30,
    "decay_rate": 0.1,
    "optimizer_eps": 1e-8,
    "optimizer_betas": (0.9, 0.999),
    "warmup_epochs": 20,
    "weight_decay": 0.05,
    "warmup_lr": 5e-7,
    "min_lr": 5e-6,
    "lr_scheduler_decay_rate": 0.01,
    # AUG
    "mix_up": 0.0,
    "label_smoothing": 0.0
}

map_dict = {'0': [0, 0, 0, 1], '1': [1, 0, 0, 0], '2': [1, 1, 0, 0], '3': [0, 0, 1, 0],
            '4': [1, 0, 1, 0], '5': [0, 1, 1, 0], '6': [0, 1, 0, 0]}

data_transform = {
    "train": transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.RandomChoice(
                                     [transforms.RandomRotation(15),
                                      transforms.ColorJitter(contrast=(0.8, 1.2)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ColorJitter(brightness=(0.8, 1.2))]),
                                 # transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((model_config['image_size'], model_config['image_size'])),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)
    return logger


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 图像任意区域提取
import numpy as np
import cv2


def region_segmentation(img_path):
    # 读取图像
    img = cv2.imread(img_path)
    w, h = img.shape[0], img.shape[1]
    # 坐标点points
    pts = np.array([[int(h / 4), 0], [0, int(w / 4)], [0, 3 * int(w / 4)], [int(h / 4), w], [3 * int(h / 4), w],
                    [h, 3 * int(w / 4)], [h, int(w / 4)], [3 * int(h / 4), 0]])
    pts = np.array([pts])
    # 和原始图像一样大小的0矩阵，作为mask
    mask = np.zeros(img.shape[:2], np.uint8)
    # 在mask上将多边形区域填充为白色
    cv2.polylines(mask, pts, 1, 255)  # 描绘边缘
    cv2.fillPoly(mask, pts, 255)  # 填充
    # 逐位与，得到裁剪后图像，此时是黑色背景
    dst = cv2.bitwise_and(img, img, mask=mask)
    # 添加白色背景
    bg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)  # bg的多边形区域为0，背景区域为255
    dst_white = bg + dst

    # cv2.imwrite("mask.jpg", mask)
    cv2.imwrite("dst.jpg", dst)
    # cv2.imwrite("dst_white.jpg", dst_white)


def img_split(data):
    # 图片尺寸
    B, C, W, H = data.shape[0], data.shape[1], data.shape[2], data.shape[3]
    h = H / 2
    w = W / 2
    # 分割后的图像
    splited_chunks = []
    # 将图片分成四部分，这部分可以自己更改
    for i in range(2):
        for j in range(2):
            region = data[:, :, int(w * i):int(w * (i + 1)), int(h * j):int(h * (j + 1))]
            splited_chunks.append(region.resize_(B, C, 224, 224))

    return splited_chunks


# normally loss functions

# sigmoid_cross_entropy
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_cross_entropy(y_true, y_pred):
    """
    :param y_true: labels
    :param y_pred: outputs of network without final activation.
    :return: mean losses
    """
    t_loss = y_true * np.log(sigmoid(y_pred)) + \
             (1 - y_true) * np.log(1 - sigmoid(y_pred))  # [batch_size,num_class]
    loss = t_loss.mean(axis=-1)  # 得到每个样本的损失值,按行平均
    return -loss.mean()  # 返回整体样本的损失均值,按列平均


def tf_sigmoid_cross_entropy(y_true, y_pred):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss = tf.reduce_mean(loss, axis=-1)
    return tf.reduce_mean(loss)


def torch_sigmoid_cross_entropy(y_pred, y_true):
    loss_func = nn.MultiLabelSoftMarginLoss(reduction='mean')
    loss = loss_func(y_pred, y_true)
    return loss


# softmax_cross_entropy
def softmax(x):
    s = np.exp(x)
    return s / np.sum(s, axis=-1, keepdims=True)


def softmax_cross_entropy(y_pred, y_true):
    """
    :param y_true: labels
    :param y_pred: outputs of network without final activation.
    :return: mean losses
    """
    logits = softmax(y_pred)
    c = -(y_true * np.log(logits)).sum(axis=-1)  # 计算每一个样本的在各个标签上的损失和
    return np.mean(c)  # 计算所有样本损失的平均值


def tf_softmax_cross_entropy(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
    return tf.reduce_mean(loss)


def torch_softmax_cross_entropy(y_pred, y_true):
    s = torch.exp(y_pred)
    logits = s / torch.sum(s, dim=1, keepdim=True)
    c = -(y_true * torch.log(logits)).sum(dim=-1)
    return torch.mean(c)


# Hamming Loss
def hamming_loss(y_true, y_pred):
    """
    return the entire incorrect predictions over entire labels.
    :param y_true:  np.array
    :param y_pred:  np.array
    Examples:
        >>> from utils import hamming_loss
        >>> import numpy as np
        >>> y_true =  np.array([[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]])
        >>> y_pred = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1]])
        >>> print(hamming_loss(y_true, y_pred))
        (2 + 0 + 3) / 12 = 0.4166666666666667
    """
    count = 0
    for i in range(y_true.shape[0]):
        p = np.size(y_true[i] == y_pred[i])
        q = np.count_nonzero(y_true[i] == y_pred[i])
        count += p - q
    return count / (y_true.shape[0] * y_true.shape[1])


# evaluation metrics

# Exact Match Ratio
def entire_match_rate(y_true, y_pred):
    """
    return the entire match ratio between predictions and labels.
    :param y_true:  np.array
    :param y_pred:  np.array
    Examples:
        >>> from utils import entire_match_rate
        >>> import numpy as np
        >>> y_true =  np.array([[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]])
        >>> y_pred = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1]])
        >>> print(entire_match_rate(y_true, y_pred))
        1 / 3 = 0.3333333333333333
    """
    return accuracy_score(y_true, y_pred)


# incompletely math rate. Consider patio prediction.

# Accuracy
def accuracy(y_true, y_pred):
    """
    Compute intersection over uion of each instance, return the mean result.
    :param y_true: np.array
    :param y_pred: np.array
    Examples:
        >>> from utils import accuracy
        >>> import numpy as np
        >>> y_true =  np.array([[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]])
        >>> y_pred = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1]])
        >>> print(accuracy(y_true, y_pred))
        ((1 / 3) + (2 / 2) + (1 / 4)) / 3 = 0.5277777777777778
    """
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]


# Accuracy
def accuracy_v2(y_true, y_pred):
    """
    Compute correct prediction over number of labels of each instance, return the mean result.
    :param y_true: np.array
    :param y_pred: np.array
    Examples:
        >>> from utils import accuracy_v2
        >>> import numpy as np
        >>> y_true =  np.array([[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]])
        >>> y_pred = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1]])
        >>> print(accuracy_v2(y_true, y_pred))
        ((2 / 4) + (4 / 4) + (1 / 4)) / 3 = 0.5833333333333334
    """
    count = 0
    num_labels = len(y_true[0])
    for i in range(y_true.shape[0]):
        mask = (y_true[i] == y_pred[i])
        acc = mask.sum() / num_labels
        count += acc
    return count / y_true.shape[0]


# Precise
def precision(y_true, y_pred):
    """
    Compute intersection over prediction of every instance, return the mean result.
    :param y_true: np.array
    :param y_pred: np.array
    Examples:
        >>> from utils import precision
        >>> import numpy as np
        >>> y_true =  np.array([[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]])
        >>> y_pred = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1]])
        >>> print(precision(y_true, y_pred))
        ((1 / 2) + (2 / 2) + (1 / 2)) / 3 = 0.66666666667
    """
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return count / y_true.shape[0]


# Recall(Sensitivity)灵敏度
def recall(y_true, y_pred):
    """
    Compute intersection over labels of every instance, return the mean result.
    :param y_true: np.array
    :param y_pred: np.array
    Examples:
        >>> from utils import recall
        >>> import numpy as np
        >>> y_true =  np.array([[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]])
        >>> y_pred = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1]])
        >>> print(recall(y_true, y_pred))
        ((1 / 2) + (2 / 2) + (1 / 3)) / 3 = 0.611111111111111
    """
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    return count / y_true.shape[0]


# Specificity(特异性、多标签)
def specificity(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        t_verse = [1 - i for i in y_true[i]]
        p_verse = [1 - i for i in y_pred[i]]
        if sum(t_verse) == 0:
            continue
        count += sum(np.logical_and(t_verse, p_verse)) / sum(t_verse)
    return count / y_true.shape[0]


def specificity_multiclass(y_true, y_pred):
    labels = _check_set_wise_labels(y_true, y_pred, average='macro', labels=None,
                                    pos_label=1)

    # Calculate tp_sum, pred_sum, true_sum ###
    MCM = multilabel_confusion_matrix(y_true, y_pred,
                                      sample_weight=None,
                                      labels=labels, samplewise=False)
    tn_sum = MCM[:, 0, 0]
    pred_sum = tn_sum + MCM[:, 0, 1]

    tn_sum = np.array([tn_sum.sum()])
    pred_sum = np.array([pred_sum.sum()])

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    specificity = _prf_divide(tn_sum, pred_sum, 'specificity', 'predicted',
                              average='micro', warn_for=('precision', 'recall', 'f-score'), zero_division="warn")

    return specificity


# Specificity(单类别特异性)
def specificity_single(y_true, y_pred):
    count = 0
    t_verse = [1 - i for i in y_true]
    p_verse = [1 - i for i in y_pred]
    count += sum(np.logical_and(t_verse, p_verse)) / sum(t_verse)
    return count


# F1
def f1_value(y_true, y_pred):
    """
    Compute the weighted average between precise and recall.
    :param y_true: np.array
    :param y_pred: np.array
    Examples:
        >>> from utils import f1_value
        >>> import numpy as np
        >>> y_true =  np.array([[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]])
        >>> y_pred = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1]])
        >>> print(f1_value(y_true, y_pred))
        (2 / 3) * ((1 / 4) + (2 / 4) + (1 / 5)) = 0.6333333333333333
    """
    count = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(y_true[i]) + sum(y_pred[i])
        count += (2 * p) / q
    return count / y_true.shape[0]


# metrics for each class
def accuracy_v2_report(y_true, y_pred):
    """
    Compute accuracy rate of each class.
    :param y_true: np.array
    :param y_pred: np.array
    Examples:
        >>> from utils import accuracy_v2_report
        >>> import numpy as np
        >>> y_true =  np.array([[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]])
        >>> y_pred = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1]])
        >>> print(accuracy_v2_report(y_true, y_pred))
        np.array([0.66667, 0.66667, 0.33333, 0.66667])
    """
    accuracy_rates = []
    classes = len(y_true[0])
    for i in range(classes):
        y = y_true[:, i]
        pred = np.array(y_pred)[:, i]
        mask = (y == pred)
        accuracy = sum(mask) / y_true.shape[0]
        accuracy_rates.append(accuracy)
    return np.array(accuracy_rates)


# Precise
def precision_report(y_true, y_pred):
    """
    Compute precise score for each class.
    :param y_true: np.array
    :param y_pred: np.array
    Examples:
        >>> from utils import precision_report
        >>> import numpy as np
        >>> y_true =  np.array([[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]])
        >>> y_pred = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1]])
        >>> print(precision_report(y_true, y_pred))
        np.array([nan, 0.66667, 0.5, 1])
    """
    precises = []
    classes = len(y_true[0])
    for i in range(classes):
        y = y_true[:, i]
        pred = np.array(y_pred)[:, i]
        tp = sum(np.logical_and(y, pred))
        tp_plus_fp = sum(pred)
        if tp_plus_fp == 0:
            precises.append(0)
            continue
        precise = tp / tp_plus_fp
        precises.append(precise)
    return np.array(precises)


# Recall(Sensitivity)灵敏度
def recall_report(y_true, y_pred):
    """
    Compute recall for each class.
    :param y_true: np.array
    :param y_pred: np.array
    Examples:
        >>> from utils import recall_report
        >>> import numpy as np
        >>> y_true =  np.array([[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]])
        >>> y_pred = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1]])
        >>> print(recall_report(y_true, y_pred))
        np.array([0, 1, 0.5, 0.5])
    """
    recalls = []
    classes = len(y_true[0])
    for i in range(classes):
        y = y_true[:, i]
        pred = np.array(y_pred)[:, i]
        tp = sum(np.logical_and(y, pred))
        fn_plus_fn = sum(y)
        if fn_plus_fn == 0:
            recalls.append(0)
            continue
        recall = tp / fn_plus_fn
        recalls.append(recall)
    return np.array(recalls)


# Specificity(特异性)
def specificity_report(y_true, y_pred):
    specificities = []
    y_true = y_true.cpu().detach().numpy()
    classes = len(y_true[0])
    for i in range(classes):
        y = y_true[:, i]
        pred = np.array(y_pred)[:, i]
        y_verse = [1 - i for i in y]
        pred_verse = [1 - i for i in pred]
        if sum(y_verse) == 0:
            specificities.append(0)
            continue
        specificity = sum(np.logical_and(y_verse, pred_verse)) / sum(y_verse)
        specificities.append(specificity)
    return np.array(specificities)


def f1_report(y_true, y_pred):
    """
    Compute the f1 value for each class.
    :param y_true: np.array
    :param y_pred: np.array
    Examples:
        >>> from utils import f1_report
        >>> import numpy as np
        >>> y_true =  np.array([[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]])
        >>> y_pred = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1]])
        >>> print(f1_report(y_true, y_pred))
        np.array([])
    """
    f1_scores = []
    precises = precision_report(y_true, y_pred)
    recalls = recall_report(y_true, y_pred)
    for i in range(len(y_true[0])):
        f1 = 2 * precises[i] * recalls[i] / (precises[i] + recalls[i])
        f1_scores.append(f1)
    return np.array(f1_scores)


def absolute_accuracy_score(y_true, y_pred):
    count = 0
    num_labels = len(y_true[0])
    for i in range(y_true.shape[0]):
        mask = (y_true[i] == y_pred[i])
        if mask.sum() == num_labels:
            count += 1
    return count / y_true.shape[0]
