import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from utils import Seed
import argparse
import os
from data.dataset import get_isic_dataset, get_gdr_dataset
from model.base import BaseModel

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='isic', help='isic or gdr')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--loss', type=str, default='entropy_loss', help='focal_loss or entropy_loss')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_dir', type=str, default='../dataset/', help='path to dataset')
parser.add_argument('--model', type=str, default='resnet18', help='resnet18 or resnet50')
parser.add_argument('--model_save_dir', type=str, default='saved/', help='path to save client model')
parser.add_argument('--client_idx', type=int, default=0, help='client index')

args = parser.parse_args()

print('args:')
for k, v in sorted(vars(args).items()):
    print("\t{}: {}".format(k, v))

device = 'cuda'
Seed(args.seed)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if args.dataset == 'isic':
    train_loaders, val_loaders, test_loaders, _ = get_isic_dataset(args.data_dir, args.batch_size, num_workers=4,
                                                                   val_rate=0.05)
elif args.dataset == 'gdr':
    train_loaders, val_loaders, test_loaders, _ = get_gdr_dataset(args.data_dir, args.batch_size, num_workers=4)
else:
    raise NotImplementedError

train_loader = train_loaders[args.client_idx]
val_loader = val_loaders[args.client_idx]
test_loader = test_loaders

class_num = 8 if args.dataset == 'isic' else 5

basemodel = BaseModel(num_classes=class_num, base=args.model).to(device)
optim = torch.optim.SGD(basemodel.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

if args.loss == 'focal_loss':
    criterion = FocalLoss()
elif args.loss == 'entropy_loss':
    criterion = nn.CrossEntropyLoss()
else:
    raise NotImplementedError

best_val_acc = 0.0
for epoch in range(args.epochs):
    basemodel.train()
    loss_sum = 0
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)
        output = basemodel(image)
        loss = criterion(output, label)
        loss_sum += loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

    print('Epoch: {}/{} Loss_avg: {:.4f}'.format(epoch, args.epochs, loss_sum / len(train_loader)))

    if epoch % 5 == 4:
        basemodel.eval()
        with torch.no_grad():
            softmax = torch.nn.Softmax(dim=1)
            loss = 0
            label_list = []
            pred_list = []
            output_list = []
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)
                output = basemodel(image)
                predict = softmax(output)
                loss += criterion(output, label).item()

                _, pred = torch.max(output, 1)
                output_sf = softmax(output)

                label_list.append(label.cpu().data.numpy())
                pred_list.append(pred.cpu().data.numpy())
                output_list.append(output_sf.cpu().data.numpy())

            label = [item for sublist in label_list for item in sublist]
            pred = [item for sublist in pred_list for item in sublist]

            acc = accuracy_score(label, pred)
            f1 = f1_score(label, pred, average='macro')
            kappa = cohen_kappa_score(label, pred)
            loss = loss / len(train_loader)

        print('>>> Val: Loss {:.4f}, Acc {:.4f}, F1 {:.4f}, Kappa {:.4f}'.format(loss, acc, f1, kappa))

        if acc > best_val_acc:
            best_val_acc = acc
            save_path = os.path.join(args.model_save_dir, 'client_{}'.format(args.client_idx))
            basemodel.save_model(save_path)
            print('Saved model with highest acc ...')

basemodel.renew_model(save_path)
basemodel.eval()
softmax = torch.nn.Softmax(dim=1)
loss = 0
label_list = []
pred_list = []
output_list = []

for image, label in test_loader:
    image, label = image.to(device), label.to(device)
    output = basemodel(image)
    predict = softmax(output)
    loss += criterion(output, label).item()

    _, pred = torch.max(output, 1)
    output_sf = softmax(output)

    label_list.append(label.cpu().data.numpy())
    pred_list.append(pred.cpu().data.numpy())
    output_list.append(output_sf.cpu().data.numpy())

label = [item for sublist in label_list for item in sublist]
pred = [item for sublist in pred_list for item in sublist]

acc = accuracy_score(label, pred)
f1 = f1_score(label, pred, average='macro')
kappa = cohen_kappa_score(label, pred)
loss = loss / len(train_loader)

print('>>> Test: Loss {:.4f}, Acc {:.4f}, F1 {:.4f}, Kappa {:.4f}'.format(loss, acc, f1, kappa))
