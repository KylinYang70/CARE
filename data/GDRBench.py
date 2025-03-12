import os.path as osp
from torch.utils.data.dataset import Dataset
from PIL import Image


# Dataset for fundus images including APTOS, DEEPDR, FGADR, IDRID, MESSIDOR, RLDR (and DDR, Eyepacs for ESDG)
class GDRBench(Dataset):
    def __init__(self, domain, mode, root, trans_basic=None):
        root = osp.abspath(osp.expanduser(root))
        self.mode = mode
        self.dataset_dir = osp.join(root, "images")
        self.split_dir = osp.join(root, "splits")

        self.data = []
        self.label = []

        self.trans_basic = trans_basic

        if mode == "train":
            self._read_data(domain, "train")
        elif mode == "val":
            self._read_data(domain, "calibrate")
        elif mode == "test":
            self._read_data(domain, "test")

    def _read_data(self, dname, split):
        file = osp.join(self.split_dir, dname + "_" + split + ".txt")
        impath_label_list = self._read_split(file)
        for impath, label in impath_label_list:
            self.data.append(impath)
            self.label.append(label)

    def _read_split(self, split_file):
        items = []
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                impath = osp.join(self.dataset_dir, impath)
                label = int(label)
                items.append((impath, label))

        return items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = Image.open(self.data[idx]).convert("RGB")
        label = self.label[idx]
        if self.trans_basic is not None:
            data = self.trans_basic(data)
        return data, label
