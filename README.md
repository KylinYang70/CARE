# CARE: A Calibration-Aware Cross-Institutional Collaboration Framework for Medical Image Classification


## 📂 Repository Structure

```
├── data/                 # Dataset processing scripts
├── model/                # ResNet models and base classifiers
├── saved/                # Folder for saving source models
├── loss.py               # Loss functions
├── train_client.py       # Source institution training script
├── fusion.py             # Fusion script
├── utils.py              # Utility functions
└── README.md             # Documentation
```

---

## 🛠 Setup

### **1️⃣ Install Dependencies**
Ensure **Python 3.8+** is installed, then run:

```bash
pip install -r requirements.txt
```

---

## 📥 Data Preparation

1. **Download the datasets**:
   
   - **ISIC dataset**: Part of the **FLamby** benchmark. Clone the repository:
     
     ```bash
     git clone https://github.com/owkin/FLamby.git
     ```
     Follow the dataset preparation instructions in the FLamby repository.
     
   - **GDRBench dataset**: Available at:
     
     ```bash
     git clone https://github.com/chehx/DGDR.git
     ```
     Prepare the dataset following the guidelines provided in the DGDR repository.


2. Organize the dataset as follows:
```
data_path/
 ├── isic/
 │   ├── downloaded_images/
 │   ├── label/
 ├── gdr/
 │   ├── images/
 │   ├── masks/
 │   ├── splits/
```

3. Modify dataset paths in `train_client.py` and `fusion.py` as needed.

---

## 🚀 Training Individual Clients

Each source institution trains its own model independently.

### **Train a Single Model**
```bash
python train_client.py --dataset 'isic' --data_dir 'your/dataset/path' --client_idx 0 --epochs 100 --model 'resnet18' --loss 'entropy_loss'
```

---

## 🔄 Model Fusion

Once all source models have trained their models, we aggregate them using the CARE framework.

### **Run Fusion.py**
```bash
python fusion.py --dataset 'isic' --client_list 0 1 2 3 4
```

After fusion, the script will output performance metrics:

- **Accuracy (`acc`)** 、 **Kappa Score (`kappa`) 、F1 Score (`f1`)**

---

## 📜 Citation (To be updated)
Due to the blind review process, the citation is currently omitted.
