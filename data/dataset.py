from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from isic import FedIsic2019
from GDRBench import GDRBench

def get_isic_dataset(data_root, batch_size=32, num_workers=4, val_rate=0.1):
    num_clients = 6
    train_loaders, val_loaders, test_loaders = [], [], []
    all_val_datasets = []

    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for center in range(num_clients):
        train_dataset = FedIsic2019(center=center, split='train', transform=train_transform, data_path=data_root,
                                    val_rate=val_rate)
        val_dataset = FedIsic2019(center=center, split='val', transform=test_transform, data_path=data_root,
                                  val_rate=val_rate)
        test_dataset = FedIsic2019(center=center, split='test', transform=test_transform, data_path=data_root)

        all_val_datasets.append(val_dataset)

        train_loaders.append(
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True))

        val_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers))

    combined_val_dataset = ConcatDataset(all_val_datasets)
    combined_val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loaders, val_loaders, test_loader, combined_val_loader


def get_gdr_dataset(data_root, batch_size=32, num_workers=4):
    domains = ['APTOS', 'DEEPDR', 'FGADR', 'IDRID', 'RLDR']
    train_loaders, val_loaders, test_loaders = [], [], []
    all_val_datasets = []
    all_test_datasets = []

    tra_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tra_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for domain in domains:
        train_dataset = GDRBench(domain=domain, mode='train', root=data_root, trans_basic=tra_train)
        val_dataset = GDRBench(domain=domain, mode='val', root=data_root, trans_basic=tra_test)
        test_dataset = GDRBench(domain=domain, mode='test', root=data_root, trans_basic=tra_test)

        all_val_datasets.append(val_dataset)
        all_test_datasets.append(test_dataset)
        train_loaders.append(
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True))
        val_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers))

    combined_val_dataset = ConcatDataset(all_val_datasets)
    combined_test_dataset = ConcatDataset(all_test_datasets)
    combined_val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loaders, val_loaders, combined_test_loader, combined_val_loader

class SimpleDataset(Dataset):
    def __init__(self, data, targets, images=None):
        self.data = data
        self.targets = targets
        self.images = images

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.images is not None:
            return self.data[idx, :], self.targets[idx], self.images[idx], idx

        img, target = self.data[idx, :], self.targets[idx]
        return img, target

