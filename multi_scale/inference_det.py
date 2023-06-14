import os
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn import metrics
from pytorch_msssim import ms_ssim, ssim
from multi_scale.ssim_module import *
# from multi_scale.ssim_module_256 import twoin1Generator as twoin1Generator256

from multi_scale.mvtec_data_loader import MvtecDataLoader
import torch
import torchvision

# import Helper

torch.cuda.empty_cache()

cuda_dev = "cuda:0"
device = torch.device(cuda_dev)
print(">> Device Info: {} is in use".format(torch.cuda.get_device_name(0)))

saver_count = 0

BATCH_SIZE = 1

############################ Parameters ############################
latent_dimension = 128

category = {
    1: "bottle",
    2: "hazelnut",
    3: "capsule",
    4: "metal_nut",
    5: "leather",
    6: "pill",
    7: "wood",
    8: "carpet",
    9: "tile",
    10: "grid",
    11: "cable",
    12: "transistor",
    13: "toothbrush",
    14: "screw",
    15: "zipper"
}

num_worker = 12
# ####################################################################
def load_train(train_path, sample_rate):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomChoice([
        #     torchvision.transforms.RandomApply([
        #         torchvision.transforms.RandomRotation(degrees=(90, 90)),
        #     ], p=0.5),
        #     # torchvision.transforms.RandomApply([
        #     #     torchvision.transforms.RandomRotation(degrees=(180, 180)),
        #     # ], p=0.5),
        #     torchvision.transforms.RandomApply([
        #         torchvision.transforms.RandomRotation(degrees=(-90, -90)),
        #     ], p=0.5),
        # ]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagenet_data = MvtecDataLoader(train_path, transform=transform, mode="train", sample_rate=sample_rate)
    # imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform)

    train_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=num_worker,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    return train_data_loader, imagenet_data.__len__()


def load_test(test_path, sample_rate):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagenet_data = MvtecDataLoader(test_path, transform=transform, mode="test", sample_rate=sample_rate)

    valid_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=num_worker,
                                                    pin_memory=True,
                                                    shuffle=False)
    return valid_data_loader


AUC_ALL = []


all_group = None


def extract_patch(data_tmp):
    data_tmp = data_tmp.permute(0, 2, 3, 1, 4, 5)
    tmp = data_tmp.reshape(-1, 3, 32, 32)
    return tmp


data_range = 2.1179 + 2.6400
for key in category:
    NORMAL_NUM = category[key]
    print('Current Item: {}'.format(NORMAL_NUM))

    train_path = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/train/'.format(NORMAL_NUM)
    vaild_path = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/train/vaild/'.format(NORMAL_NUM)
    test_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/test/'.format(NORMAL_NUM)
    gt_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/ground_truth/'.format(NORMAL_NUM)

    # recorder = Recorder(Experiment_name, 'CIFAR-10_No.{}'.format(str(NORMAL_NUM)))
    from p256.mvtec_module import twoin1Generator256 as twoin1Generator256
    from p32.ssim_module import twoin1Generator as twoin1Generator32
    generator256 = twoin1Generator256(64, latent_dimension=latent_dimension)
    generator32 = twoin1Generator32(64, latent_dimension=latent_dimension)

    exp_name = "SR-1.0"
    
    ckpt32 = torch.load('./p32/check_points/p32_{}/No.{}_32_g.pth'.format(exp_name, NORMAL_NUM), map_location=cuda_dev)
    generator32.load_state_dict(ckpt32['model'])
    generator32.c = ckpt32['c']
    generator32.sigma = ckpt32['sigma']


    ckpt256 = torch.load('./p256/check_points/IGD_wo_inter/p256_{}/No.{}_256_g.pth'.format(exp_name, NORMAL_NUM), map_location=cuda_dev)
    generator256.load_state_dict(ckpt256['model'])
    generator256.c = ckpt256['c']
    generator256.sigma = ckpt256['sigma']

    criterion = torch.nn.MSELoss()
    l1_criterion = torch.nn.L1Loss()
    # ent_criterion = NegEntropyLoss()

    generator32.to(device)
    generator256.to(device)
    # discriminator.to(device)

    generator32.eval()
    generator256.eval()
    # discriminator.eval()

    y = []
    y_pred = []
    score = []
    score_recon = []
    score_gsvdd = []
    normal_mse_loss = []
    abnormal_mse_loss = []
    train_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/train/'.format(NORMAL_NUM)
    test_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/test/'.format(NORMAL_NUM)
    list_test = os.listdir(test_root)

    p_stride = 16

    with torch.no_grad():
        for i in range(len(list_test)):
            current_defect = list_test[i]
            test_path = test_root + "{}".format(current_defect)
            valid_dataset_loader = load_test(test_path, sample_rate=1.)
            for index, (images, label) in enumerate(valid_dataset_loader):
                # img = images.to(device)
                img_whole = images.to(device)
                img_tmp = img_whole.unfold(2, 32, p_stride).unfold(3, 32, p_stride)
                img = extract_patch(img_tmp)
                ############################################################################################
                # 32
                ############################################################################################
                latent_z_32 = generator32.encoder(img)
                generate_result32 = generator32(img)
                weight = 0.85
                ms_ssim_batch_wise = 1 - ms_ssim(img, generate_result32, data_range=data_range,
                                                 size_average=False, win_size=3,
                                                 weights=[0.0516, 0.3295, 0.3463, 0.2726])
                l1_batch_wise = (img - generate_result32) / data_range
                l1_batch_wise = l1_batch_wise.mean(1).mean(1).mean(1)
                # else:
                #     l1_batch_wise = l1_criterion(img, generate_result32) / data_range
                score_recon32 = weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise
                diff = (latent_z_32 - generator32.c) ** 2
                dist = -1 * torch.sum(diff, dim=1) / generator32.sigma
                score_gsvdd32 = 1 - torch.exp(dist)
                anormaly_score32 = (0.9 * score_recon32 + 0.1 * score_gsvdd32).max().cpu().detach().numpy()
                ############################################################################################
                # 256
                ############################################################################################
                latent_z_256 = generator256.encoder(img_whole)
                generate_result256 = generator256(img_whole)
                weight = 0.85
                ms_ssim_batch_wise = 1 - ms_ssim(img_whole, generate_result256, data_range=data_range,
                                                 size_average=True, win_size=11,
                                                 weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
                l1_batch_wise = l1_criterion(img_whole, generate_result256) / data_range
                score_recon256 = weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise

                diff = (latent_z_256 - generator256.c) ** 2
                dist = -1 * torch.sum(diff, dim=1) / generator256.sigma
                score_gsvdd256 = 1 - torch.exp(dist)
                anormaly_score256 = float((0.1 * score_recon256 + 0.9 * score_gsvdd256).max().cpu().detach().numpy())

                beta = 0.5
                anormaly_score_final = beta * anormaly_score32 + (1 - beta) * anormaly_score256

                score.append(anormaly_score_final)
                if label[0] == "good":
                    # normal_gsvdd.append(float(guass_svdd_loss.cpu().detach().numpy()))
                    normal_mse_loss.append(anormaly_score_final)
                    y.append(0)
                else:
                    # abnormal_gsvdd.append(float(guass_svdd_loss.cpu().detach().numpy()))
                    abnormal_mse_loss.append(anormaly_score_final)
                    y.append(1)

    ###################################################
    # Helper.plot_2d_chart(x1=numpy.arange(0, len(normal_mse_loss)), y1=normal_mse_loss, label1='normal_loss',
    # x2=numpy.arange(len(normal_mse_loss), len(normal_mse_loss) + len(abnormal_mse_loss)),
    # y2=abnormal_mse_loss, label2='abnormal_loss',
    # title=NORMAL_NUM)

    fpr, tpr, thresholds = metrics.roc_curve(y, score, pos_label=1)
    auc_result = auc(fpr, tpr)
    print(auc_result)
    AUC_ALL.append(auc_result)

average_auc = sum(AUC_ALL) / len(AUC_ALL)
print('Average AUC:', average_auc)
