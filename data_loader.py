import torch
import torchvision
import folders

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain

        if dataset == 'hsimagedataset':
            transform = transforms.Compose(
                [transforms.RandomCrop(size=patch_size), # use randomcrop and compare with cc
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                
            self.data = folders.CustomImageDataset(
                root = path + '/train/0_clean', index=img_indx, transform=transforms, patch_num=patch_num, label=2
            ) + 
            folders.CustomImageDataset(
                root = path + '/train/1_light', index=img_indx, transform=transforms, patch_num=patch_num, label=1
            ) + 
            folders.CustomImageDataset(
                root = path + '/train/2_heavy', index=img_indx, transform=transforms, patch_num=patch_num, label=0
            )
       
    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False)
        return dataloader