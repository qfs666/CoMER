import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import cv2
from .transforms import ScaleAugmentation, ScaleToLimitRange
# from transforms import ScaleAugmentation, ScaleToLimitRange

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024


# class CROHMEDataset(Dataset):
#     def __init__(self, ds, is_train: bool, scale_aug: bool) -> None:
#         super().__init__()
#         self.ds = ds

#         trans_list = []
#         if is_train and scale_aug:
#             trans_list.append(ScaleAugmentation(K_MIN, K_MAX))

#         trans_list += [
#             ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
#             tr.ToTensor(),
#         ]
#         self.transform = tr.Compose(trans_list)

#     def __getitem__(self, idx):
#         fname, img, caption = self.ds[idx]
#         print(fname,img,caption)
#         img = [self.transform(im) for im in img]

#         return fname, img, caption

#     def __len__(self):
#         return len(self.ds)
    

class CROHMEDataset(Dataset):
    def __init__(self, ds, is_train: bool, scale_aug: bool) -> None:
        super().__init__()
        if is_train:
            img_path = os.path.join(ds,"train","img")
            captions = os.path.join(ds,"train","caption.txt")
            # img_path = os.path.join(ds,"2019","img")
            # captions = os.path.join(ds,"2019","caption.txt")
        else:
            img_path = os.path.join(ds,"2019","img")
            captions = os.path.join(ds,"2019","caption.txt")


        captions_labels = {}
        datas = open(captions,'r').readlines()
        for data in datas:
            name,labels = data.strip().split("\t")
            captions_labels[name+".bmp"]=labels.split(" ")
        self.datas = []
        for i in os.listdir(img_path):
            if i in captions_labels:
                self.datas.append([i,os.path.join(img_path,i),captions_labels[i]])       
        trans_list = []
        if is_train and scale_aug:
            trans_list.append(ScaleAugmentation(K_MIN, K_MAX))

        trans_list += [
            ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
            tr.ToTensor(),
        ]
        self.transform = tr.Compose(trans_list)

    def __getitem__(self, idx):
        fname, img, caption = self.datas[idx]
        img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
        # print(img.shape)
        img = self.transform(img)
        return fname, img, caption

    def __len__(self):
        return len(self.datas)


if __name__ =="__main__":
    path = "/mnt/pfs_l2/jieti_team/CV/qfs/project/server_project/std_project/CoMER-master/data"
    datasets = CROHMEDataset(path,is_train=True,scale_aug=True)
    for i in datasets:
        print(i)