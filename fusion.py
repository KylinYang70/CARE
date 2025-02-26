from model.resnet import resnet18, resnet50
import random
import argparse
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, cohen_kappa_score
import numpy as np
import os
from dataset import *
from model.base import BaseModel
from loss import *

to_np = lambda x: x.data.cpu().numpy()


# set seed
def Seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ImageCalibrator(nn.Module):
    def __init__(self, num_clients, hidden_dim=64, img_feature_dim=128):
        super(ImageCalibrator, self).__init__()

        self.num_clients = num_clients
        self.img_feature_dim = img_feature_dim
        if args.model == 'resnet18':
            self.feature_projection = nn.Linear(512, img_feature_dim)
        elif args.model == 'resnet50':
            self.feature_projection = nn.Linear(2048, img_feature_dim)
        else:
            raise NotImplementedError
        self.fc1 = nn.Linear(int(num_clients * args.num_classes), hidden_dim)
        self.fc2 = nn.Linear(img_feature_dim, hidden_dim)
        self.cross_t1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cross_t2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_t1 = nn.Linear(hidden_dim, num_clients)
        self.fc_t2 = nn.Linear(hidden_dim, num_clients)

    def forward(self, inputs, img_feature, mode='train'):
        img_feature = img_feature.squeeze().detach()
        img_feature = self.feature_projection(img_feature)

        x1 = F.relu(nn.Dropout(p=0.5)(self.fc1(inputs)))
        x2 = F.relu(nn.Dropout(p=0.5)(self.fc2(img_feature)))

        cross_feat_t1 = torch.cat([x1, x2], dim=1)
        cross_feat_t2 = torch.cat([x2, x1], dim=1)

        x_t1 = F.relu(self.cross_t1(cross_feat_t1))
        x_t2 = F.relu(self.cross_t2(cross_feat_t2))

        t1 = nn.Softplus()(self.fc_t1(x_t1))
        t2 = nn.Softplus()(self.fc_t2(x_t2))

        inputs_chunks = torch.chunk(inputs, chunks=self.num_clients, dim=1)
        t1_chunks = torch.chunk(t1, chunks=self.num_clients, dim=1)
        t2_chunks = torch.chunk(t2, chunks=self.num_clients, dim=1)

        alphas = [p * F.softmax(i * j, dim=1) + 1 for i, j, p in zip(inputs_chunks, t1_chunks, t2_chunks)]
        x = DS_Combin(alphas, args.num_classes)

        if mode == 'train':
            return x, alphas

        return x, t1, t2, alphas


class SimpleClassifier(nn.Module):
    def __init__(self, num_clients, num_classes, hidden_dim=32):
        super(SimpleClassifier, self).__init__()

        self.fc1 = nn.Linear(int(num_clients * num_classes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, args.num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, model, num_classes):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = model.to(device)
        if args.model == 'resnet18':
            self.classifier = nn.Linear(512, num_classes)
        elif args.model == 'resnet50':
            self.classifier = nn.Linear(2048, num_classes)
        else:
            raise NotImplementedError

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out


def build_new_dataset(net_local_list, val_loader, batch_size, num_clients, mode='softmax'):
    print(len(val_loader.dataset))
    length = int(len(val_loader.dataset) / batch_size) * batch_size
    print(length)

    new_dataset = np.zeros((length, int(num_clients * args.num_classes)))
    new_labels = np.zeros(length)
    image_list = []
    for batch_idx, (image, label) in enumerate(val_loader):
        with torch.no_grad():
            image_list.append(image)
            image = image.to(device)
            client_shift_idx = 0
            for client_i in range(num_clients):
                net = net_local_list[client_i]
                output = net(image)
                if mode == 'softmax':
                    smax = to_np(F.softmax(output, dim=1))
                elif mode == 'logit':
                    smax = to_np(output)
                new_dataset[batch_size * batch_idx:batch_size * (batch_idx + 1),
                args.num_classes * client_shift_idx:args.num_classes * (client_shift_idx + 1)] = smax
                new_labels[batch_size * batch_idx:batch_size * (batch_idx + 1)] = label
                client_shift_idx += 1
    image_all = torch.cat(image_list, dim=0)
    calibrate_dataset = SimpleDataset(data=new_dataset, targets=new_labels, images=image_all)

    np.save('new_dataset.npy', np.array(new_dataset))
    np.save('new_labels.npy', np.array(new_labels))
    return calibrate_dataset


def cross_expert_calibrator(net_local_list, new_data_loader, model_extract, testloader_global, batch_size, num_clients):
    calibrator = ImageCalibrator(num_clients=num_clients).to(device)
    _new_calibrator_eps = 60
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_new_calibrator_eps)
    model_extract.eval()
    acc_max = 0.0
    best_state_dict = None
    for ep in range(_new_calibrator_eps):
        calibrator.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, images, idx) in enumerate(new_data_loader):
            images = images.to(device)
            inputs, targets = inputs.float().to(device), targets.long().to(device)
            image_features = model_extract(images)

            optimizer.zero_grad()
            outputs, alphas = calibrator(inputs, image_features, mode='train')
            loss = torch.mean(ce_loss(targets, outputs, args.num_classes, 1))
            for alpha in alphas:
                loss += 1 / num_clients * torch.mean(ce_loss(targets, alpha, args.num_classes, 1))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print("Ep: {}, Training: {}/{}, Loss: {:.3f} | Acc: {:.3f} ({}/{})".format(
                    ep, batch_idx, len(new_data_loader), train_loss / (batch_idx + 1),
                                                         100. * correct / total, correct, total))

        scheduler.step()
        with torch.no_grad():
            if (ep + 1) % 10 == 0:
                acc, kappa, f1 = global_calibration_based(net_local_list, testloader_global, calibrator=calibrator,
                                                          model_extract=model_extract, batch_size=batch_size,
                                                          num_clients=num_clients)
                if acc_max < acc:
                    acc_max = acc
                    best_state_dict = calibrator.state_dict()
                    torch.save(best_state_dict, "./calibrator_best.pth")

    print("training done!")
    print(acc_max)
    calibrator.load_state_dict(best_state_dict)
    calibrator.eval()
    with torch.no_grad():
        acc_best, kappa_best, f1_best = global_calibration_based(net_local_list, testloader_global,
                                                                 calibrator=calibrator,
                                                                 model_extract=model_extract,
                                                                 batch_size=batch_size,
                                                                 num_clients=num_clients)

    return calibrator, acc_best, kappa_best, f1_best


def global_calibration_based(net_local_list, test_loader, calibrator, model_extract, batch_size, num_clients):
    calibrator.eval()
    t1_list = []
    t2_list = []
    label_list = []
    output_list = []
    pred_list = []
    client_alpha_list = []
    model_extract.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            image, label = data
            image = image.to(device)
            image_features = model_extract(image)

            new_test = np.zeros((batch_size, int(num_clients * args.num_classes)))
            client_shift_idx = 0
            for net_local_index, net_local in enumerate(net_local_list):
                output = net_local(image)
                smax = to_np(output)
                new_test[:, args.num_classes * client_shift_idx:args.num_classes * (client_shift_idx + 1)] = smax
                client_shift_idx += 1
            new_test = torch.from_numpy(new_test).float().to(device)
            output, t1, t2, alphas = calibrator(new_test, image_features, mode='test')
            _, pred = torch.max(output, 1)

            t1_list.append(t1.cpu().numpy())
            t2_list.append(t2.cpu().numpy())
            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output.cpu().data.numpy())
            alphas_list = [item.cpu().numpy() for item in alphas]
            client_alpha_list.append(alphas_list)
        labels = np.concatenate(label_list)
        preds = np.concatenate(pred_list)
        alphas_np = np.array(client_alpha_list)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        kappa = cohen_kappa_score(labels, preds)
        print(classification_report(labels, preds))
        print('acc: {:.2f}%, kappa: {:.2f}%, F1: {:.2f}%.'.format(acc * 100, kappa * 100, f1 * 100))

        np.save('./output/t1_values.npy', np.array(t1_list))
        np.save('./output/t2_values.npy', np.array(t2_list))
        np.save('./output/output_values.npy', np.array(output_list))
        np.save('./output/label_values.npy', np.array(label_list))
        np.save('./output/alphas.npy', np.array(alphas_np))
    return acc, kappa, f1


def cross_expert_classifier(net_local_list, new_data_loader, testloader_global, batch_size, num_clients):
    MLP = SimpleClassifier(num_clients=num_clients, num_classes=args.num_classes).to(device)
    _new_classifier_eps = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(MLP.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_new_classifier_eps)

    MLP.train()
    for ep in range(_new_classifier_eps):
        targets_list = []
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, images, idx) in enumerate(new_data_loader):
            inputs, targets = inputs.float().to(device), targets.long().to(device)
            optimizer.zero_grad()
            outputs = MLP(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            targets_list.append(targets.cpu().data.numpy())

            if batch_idx % 10 == 0:
                print("Ep: {}, Training: {}/{}, Loss: {:.3f} | Acc: {:.3f} ({}/{})".format(
                    ep,
                    batch_idx, len(new_data_loader), train_loss / (batch_idx + 1),
                                                     100. * correct / total, correct, total))
        scheduler.step()
        if (ep + 1) % 10 == 0:
            acc, kappa, f1 = global_inference_learning_based(net_local_list, testloader_global, MLP=MLP,
                                                             batch_size=batch_size, num_clients=num_clients)

    return acc, kappa, f1


def global_inference_learning_based(net_local_list, test_loader, MLP, batch_size, num_clients):
    MLP.eval()
    label_list = []
    output_list = []
    pred_list = []

    with torch.no_grad():
        for batch_dix, data in enumerate(test_loader):
            image, label = data
            image = image.to(device)

            new_test = np.zeros((batch_size, int(num_clients * args.num_classes)))
            client_shift_idx = 0
            for net_local_index, net_local in enumerate(net_local_list):
                output = net_local(image)
                smax = to_np(F.softmax(output, dim=1))
                new_test[:, args.num_classes * client_shift_idx:args.num_classes * (client_shift_idx + 1)] = smax
                client_shift_idx += 1

            new_test = torch.from_numpy(new_test).float().to(device)
            output = MLP(new_test)
            _, pred = torch.max(output, 1)
            output_sf = F.softmax(output, dim=1)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
        labels = np.concatenate(label_list)
        preds = np.concatenate(pred_list)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        kappa = cohen_kappa_score(labels, preds)
        print(
            'acc: {:.2f}%, kappa: {:.2f}%, F1: {:.2f}%.'.format(acc * 100, kappa * 100, f1 * 100))
    return acc, kappa, f1


def global_inference_vanilla_ensemble(net_local_list, test_loader, batch_size, num_clients):
    with torch.no_grad():
        label_list = []
        output_list = []
        pred_list = []
        for batch_idx, data in enumerate(test_loader):
            image, label = data

            target = label.to(device)
            image = image.to(device)

            output_prob = torch.zeros(batch_size, args.num_classes)
            for net_local_index, net_local in enumerate(net_local_list):
                output = net_local(image)
                smax = to_np(F.softmax(output, dim=1))
                output_prob += smax

            output_prob /= num_clients
            _, pred = torch.max(output_prob, 1)

            label_list.append(target.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_prob.cpu().data.numpy())

        labels = np.concatenate(label_list)
        preds = np.concatenate(pred_list)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        kappa = cohen_kappa_score(labels, preds)
        print(classification_report(labels, preds))
        print(
            'acc: {:.2f}%, kappa: {:.2f}%, F1: {:.2f}%.'.format(acc * 100, kappa * 100, f1 * 100))

    return acc, kappa, f1


def global_inference_oracle_upper_bound(net_local_list, test_loader):
    label_list = []
    pred_list = []

    for model in net_local_list:
        model.eval()
        model.to(device)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            for i in range(len(images)):
                image = images[i].unsqueeze(0)  # [1, C, H, W]
                true_label = labels[i].item()

                model_preds = []
                for model in net_local_list:
                    output = model(image)
                    _, pred = torch.max(output, dim=1)
                    pred_label = pred.item()
                    model_preds.append(pred_label)

                if true_label in model_preds:
                    final_pred = true_label
                else:
                    final_pred = model_preds[0]

                label_list.append(true_label)
                pred_list.append(final_pred)

    labels = np.array(label_list)
    preds = np.array(pred_list)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    kappa = cohen_kappa_score(labels, preds)

    print('Upper Bound -> ACC: {:.2f}%, Kappa: {:.2f}%, F1: {:.2f}%'.format(
        acc * 100, kappa * 100, f1 * 100))

    return acc, kappa, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='isic')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='../isic/')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--model_save_dir', type=str, default='saved/isic/')
    parser.add_argument('--expert_list', type=int, nargs='+')

    args = parser.parse_args()

    if args.dataset == 'isic':
        args.num_classes = 8
    elif args.dataset == 'gdr':
        args.num_classes = 5

    # print setup
    print('args:')
    for k, v in sorted(vars(args).items()):
        print("\t{}: {}".format(k, v))

    Seed(args.seed)
    device = 'cuda'
    if args.dataset == 'isic':
        train_loaders, val_loaders, test_loader, val_all_expert_loader = get_isic_dataset(args.data_dir, args.batch_size,
                                                                                       num_workers=4, val_rate=0.1)
    elif args.dataset == 'gdr':
        train_loaders, val_loaders, test_loader, val_all_expert_loader = get_gdr_dataset(args.data_dir, args.batch_size,num_workers=4)
    else:
        raise NotImplementedError
    test_global_loader = test_loader
    num_clients = len(args.expert_list)
    net_local_list = [BaseModel(num_classes=args.num_classes, base=args.model).to(device) for client_index in
                      range(num_clients)]
    for local_id in range(num_clients):
        param_dir = os.path.join(args.model_save_dir, 'expert_{}'.format(args.expert_list[local_id]))
        net_local_list[local_id].renew_model(param_dir)
        net_local_list[local_id].eval()
    softmax_calibrate_dataset = build_new_dataset(net_local_list, val_all_expert_loader, args.batch_size, num_clients, mode='softmax')
    logit_calibrate_dataset = build_new_dataset(net_local_list, val_all_expert_loader, args.batch_size, num_clients, mode='logit')
    logit_test_dataset = build_new_dataset(net_local_list, test_global_loader, args.batch_size, num_clients, mode='logit')
    print("logit_test_dataset has been built.")
    softmax_new_data_loader = torch.utils.data.DataLoader(softmax_calibrate_dataset, batch_size=16, shuffle=True, num_workers=4)
    logit_new_data_loader = torch.utils.data.DataLoader(logit_calibrate_dataset, batch_size=16, shuffle=True, num_workers=4)
    if args.model == 'resnet18':
        extractor = FeatureExtractor(resnet18(pretrained=True).to(device), num_classes=args.num_classes)
    elif args.model == 'resnet50':
        extractor = FeatureExtractor(resnet50(pretrained=True).to(device), num_classes=args.num_classes)
    else:
        raise NotImplementedError
    model_extract = extractor.feature_extractor

    _, acc_evi, kappa_evi, f1_evi = cross_expert_calibrator(net_local_list, logit_new_data_loader, model_extract,
                                                            test_global_loader, args.batch_size, num_clients)
    acc_foe, kappa_foe, f1_foe = cross_expert_classifier(net_local_list, softmax_new_data_loader, test_global_loader,
                                                         args.batch_size, num_clients)
    acc_ens, kappa_ens, f1_ens = global_inference_vanilla_ensemble(net_local_list, test_global_loader, args.batch_size,
                                                                   num_clients)
    acc_upper, kappa_upper, f1_upper = global_inference_oracle_upper_bound(net_local_list, test_global_loader)

    print("_______________________________________________")
    print("Ours: acc: {:.2f}%, kappa: {:.2f}%, f1: {:.2f}%.".format(acc_evi * 100, kappa_evi * 100, f1_evi * 100))
    print("FOE: acc: {:.2f}%, kappa: {:.2f}%, f1: {:.2f}%.".format(acc_foe * 100, kappa_foe * 100, f1_foe * 100))
    print("Ensemble: acc: {:.2f}%, kappa: {:.2f}%, f1: {:.2f}%.".format(acc_ens * 100, kappa_ens * 100, f1_ens * 100))
    print("Upper bound: acc: {:.2f}%, kappa: {:.2f}%, f1: {:.2f}%.".format(acc_upper * 100, kappa_upper * 100, f1_upper * 100))
