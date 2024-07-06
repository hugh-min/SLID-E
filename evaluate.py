from data_loader import *
from sklearn.metrics import f1_score, recall_score, roc_auc_score, classification_report
from torch.utils.tensorboard import SummaryWriter
from utils import *

writer = SummaryWriter("logs")
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# 质量筛选模型评估数据路径：/EYE/cy/leiqibing/dataset/reorganize_data2/quality/val
# 疾病诊断raw data模型数据路径：/EYE/cy/leiqibing/dataset/reorganize_data2/diease/val
# 疾病诊断augumented data模型数据路径：/EYE/cy/leiqibing/dataset/reorganize_data2/augument/val_aug
def make_evaluate():
    checkpoint = torch.load(r"/EYE/leiqibing/checkpoints/diease/augued/bisenet_model.h5")
    model_2cls = checkpoint['net']
    model_2cls.to(device)
    model_2cls.eval()
    val_dataset = datasets.ImageFolder(root="/EYE/leiqibing/dataset/test",
                                       transform=data_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, num_workers=32, shuffle=True)
    losses = []
    y_pred = []
    y_labels = []
    criterion = nn.CrossEntropyLoss()
    for data, label in val_loader:
        with torch.no_grad():
            data = data.to(device)
            label = label.to(device)
            output = model_2cls(data)
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
    auc = roc_auc_score(y_labels, y_pred, average='macro')
    losses = np.mean(losses)
    # auc = roc_auc_score(y_labels, y_pred, multi_class="ovr", average='macro') '图片不符合标准', '图片符合标准'
    # p, r, f1, _ = precision_recall_fscore_support(y_labels, y_pred, average='macro')  'mild_severe', 'normal_moderate'
    print(classification_report(y_labels, y_pred, target_names=['mild_severe', 'normal_moderate']))
    print('accuracy:', acc_v2)
    print("sensitivity", sensi)
    print("specificity", speci)
    print("f1", f1_)
    print("auc", auc)
    return acc_v2, sensi, speci, auc, f1_, losses


def make_predict(img_path):
    reflect = {0: 'mild_severe', 1: 'normal_moderate'}
    checkpoint = torch.load(r"/EYE/cy/leiqibing/result/reorganize2/2cls_aug/GoogLeNet-0.9487179487179487_2cls/pytorch_model.h5")
    model_2cls = checkpoint['net']
    model_2cls.to(device)
    model_2cls.eval()
    img = Image.open(img_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0).to(device)
    cls = torch.softmax(model_2cls(img), dim=1)
    predict_cls = torch.max(cls, dim=1)[1].long().cpu().detach().numpy()[0]
    result = reflect[predict_cls]
    print(result)


make_evaluate()