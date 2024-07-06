# coding: utf-8

"""
@File     :train.py
@Author    :chengyu
@Date      :2021/8/16
@Copyright :AI
@Desc      :
"""
import ctypes

libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import json

import torch.nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms, datasets
import logging
from sklearn.metrics import auc, precision_recall_fscore_support, roc_auc_score, classification_report, accuracy_score, \
    f1_score, recall_score, confusion_matrix
from lr_scheduler import build_scheduler
from m import *
from torch.utils.tensorboard import SummaryWriter
from Criterion import *
from utils import *

import warnings

warnings.filterwarnings('ignore')

writer = SummaryWriter("logs")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)


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


logger = init_logger()


# seed_everything(model_config['seed'])


class Experiment(object):
    def __init__(self, *, model_name='xception', data_root='/', pretrained=False, config=None):
        self.model_name = model_name
        self.data_root = data_root
        self.pretrained = pretrained
        self.config = config

    def data_prepare(self, is_train=True):
        """
        进行数据类型校验和训练集、验证集准备
        :return:
        """
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
            "val": transforms.Compose([transforms.Resize((self.config['image_size'], self.config['image_size'])),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
        if is_train:

            train_dataset = datasets.ImageFolder(
                root=os.path.join(self.data_root, "/EYE/cy/leiqibing/dataset/reorganize_data2/augument/train_aug"),
                transform=data_transform['train'])

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'],
                                                       num_workers=32,
                                                       shuffle=True)
            label_list = train_dataset.class_to_idx
            print(label_list)

            return train_loader, label_list
        else:
            val_dataset = datasets.ImageFolder(
                root=os.path.join(self.data_root, "/EYE/cy/leiqibing/dataset/reorganize_data2/augument/val_aug"),
                transform=data_transform['val'])
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config['batch_size'], num_workers=32,
                                                     shuffle=True)

            return val_loader

    def train(self):
        global_step = 0
        best_step = None
        best_score = .8
        cnt_patience = 0
        train_loader, label_list = self.data_prepare()
        val_loader = self.data_prepare(False)
        # model, parameters = build_model(self.config)
        model = resnet101(num_classes=2)

        #        checkpoint = torch.load(r"/EYE/cy/UWF_6classes/model_results/new_uwf/new_9cls/KeNet-0.7092520694454618_9cls/pytorch_model.h5")
        #        model = checkpoint['net']
        model.to(device)
        #        optimizer = checkpoint['opt']
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        loss_func = nn.CrossEntropyLoss()

        lr_scheduler = build_scheduler(self.config, optimizer, len(val_loader) * 4)
        for epoch in range(300):
            model.train()
            num_steps = len(val_loader) * 4
            losses = []
            steps = 0
            mean_acc, mean_sensitivity, mean_specificity, mean_f1 = [], [], [], []
            y_preds, y_labels = [], []
            pbar = tqdm(train_loader, desc="Training")

            for idx, (data, label) in enumerate(pbar):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                predict_y = torch.max(output, dim=1)[1].long().cpu().detach().numpy()
                y_preds.append(predict_y)
                y_labels.append(label.cpu().detach().numpy())
                loss = loss_func(torch.squeeze(output, dim=1).to(device), label)

                acc = accuracy_score(label.cpu().detach().numpy(), predict_y)
                sensi = recall_score(label.cpu().detach().numpy(), predict_y, average='macro')
                speci = specificity_multiclass(label.cpu().detach().numpy(), predict_y)
                f1_ = f1_score(label.cpu().detach().numpy(), predict_y, average='macro')
                mean_acc.append(acc)
                mean_sensitivity.append(sensi)
                mean_specificity.append(speci)
                mean_f1.append(f1_)
                #                print("train_accuracy_recall_specificity_f1:", (acc, sensi, speci, f1_))

                global_step += 1
                steps += 1
                losses.append(loss.cpu().detach().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + idx)
                pbar.set_description(f'epoch:{epoch} loss:{np.mean(losses)} acc:{acc} st:{sensi} st:{speci} f1:{f1_}')

            mean_acc = sum(mean_acc) / steps
            mean_sensitivity = sum(mean_sensitivity) / steps
            mean_specificity = sum(mean_specificity) / steps
            mean_f1 = sum(mean_f1) / steps
            print("train_accuracy_sensitivity_specificity_f1:", (mean_acc, mean_sensitivity, mean_specificity, mean_f1))

            model.eval()
            acc_v2, recal, speci, f1_, losses = self.evaluate(model, val_loader, epoch)
            logger.info(" acc: %s ,  lossL %s ", acc_v2, losses)
            if acc_v2 > best_score:
                best_score = acc_v2
                best_step = global_step
                cnt_patience = 0
                self._save_checkpoint(model, model._get_name(), optimizer, lr_scheduler, best_score)
                print("best_score", best_score)
            else:
                cnt_patience += 1
                logger.info("Earlystopper counter: %s out of %s", cnt_patience,
                            self.config['earlystop_patience'])
                if cnt_patience >= self.config['earlystop_patience']:
                    break
            if cnt_patience >= self.config['earlystop_patience']:
                break
        logger.info("Training Stop! The best step %s: %s", best_step, best_score)
        if device == 'cuda':
            torch.cuda.empty_cache()

    def evaluate(self, model, data_loader, epoch):
        losses = []
        y_pred = []
        y_labels = []
        criterion = nn.CrossEntropyLoss()
        for data, label in data_loader:
            with torch.no_grad():
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                predict_y = torch.max(output, dim=1)[1].long().cpu().detach().numpy()
                loss = criterion(torch.squeeze(output, dim=1).to(device), label)
                losses.append(loss.cpu().detach().numpy())
                y_pred.append(predict_y)
                y_labels.append(label.cpu().detach().numpy())
        y_pred = np.concatenate(y_pred)
        y_labels = np.concatenate(y_labels)
        acc_v2 = accuracy_score(y_labels, y_pred)
        sensi = recall_score(y_labels, y_pred, average='macro')
        speci = specificity_multiclass(y_labels, y_pred)
        f1_ = f1_score(y_labels, y_pred, average='macro')
        losses = np.mean(losses)
        # auc = roc_auc_score(y_labels, y_pred, multi_class="ovr", average='macro')
        # p, r, f1, _ = precision_recall_fscore_support(y_labels, y_pred, average='macro')
        print(classification_report(y_labels, y_pred, target_names=['mild_severe', 'normal_moderate']))
        print('accuracy_v2:', acc_v2)
        print("recal", sensi)
        print("specificity", speci)
        print("f1", f1_)
        return acc_v2, sensi, speci, f1_, losses

    def _save_checkpoint(self, model, model_name, optimizer, lr_scheduler, score):
        output_dir = os.path.join("/EYE/cy/leiqibing/result/reorganize2/2cls_aug",
                                  '{}-{}_2cls'.format(model_name, score))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_model = {
            'net': model,  # 网络权重
            'opt': optimizer,  # 优化器参数
            'lr_scheduler': lr_scheduler}
        torch.save(save_model, os.path.join(output_dir, 'pytorch_model.h5'))
        #        torch.save(model, os.path.join(output_dir, 'pytorch_model.pth'))
        with open(os.path.join(output_dir, 'config.json'), "w", encoding="utf-8") as f:
            params = self.config
            json.dump(params, f, ensure_ascii=False)
        logger.info('Saving models checkpoint to %s', output_dir)


if __name__ == '__main__':
    e = Experiment(model_name='xception', data_root=r'/home/zcb/cy/UWF_6classes/data',
                   pretrained=False, config=model_config)
    e.train()
