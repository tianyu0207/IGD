import os
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn import metrics
from pytorch_msssim_residual_map import ms_ssim, ssim

from ssim_module import *
from ssim_module_256 import twoin1Generator as twoin1Generator256

from mvtec_data_loader import MvtecDataLoader, DualDataLoader
import torch
import torchvision
# import Helper

torch.cuda.empty_cache()
cuda_dev = "cuda:1"
device = torch.device(cuda_dev)
print(">> Device Info: {} is in use".format(torch.cuda.get_device_name(0)))

saver_count = 0

BATCH_SIZE = 1
############################ Parameters ############################
latent_dimension = 128
criterion = torch.nn.MSELoss()
l1_criterion = torch.nn.L1Loss()

def rgb2gray(rgb):
    return numpy.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

class Upsample(nn.Module):
    def __init__(self,  scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)

def heat_map_printer(ms_ssim_l1, img_whole, mask, threshold=None, plot=False):
    t1 = ms_ssim_l1.reshape(1, 225, 3 * 32 * 32)
    t2 = t1.permute(0, 2, 1)
    img_assem = torch.nn.functional.fold(t2, 256, 32, stride=16)
    np_grid_image = img_assem.squeeze(0).cpu().detach().numpy()
    heat_img = numpy.transpose(np_grid_image, (1, 2, 0)).squeeze().astype(numpy.float32)
    heatmap = rgb2gray(heat_img)

    if plot:
        grid_image = torchvision.utils.make_grid(img_whole, normalize=True, nrow=1, padding=0)
        np_grid_image = grid_image.cpu().detach().numpy()
        org_img = numpy.transpose(np_grid_image, (1, 2, 0)).squeeze().astype(numpy.float32)
        plt.imshow(org_img)
        # plt.colorbar()
        plt.axis('off')
        plt.savefig("../HEATMAP/org.png")

        grid_image = torchvision.utils.make_grid(mask, normalize=True, nrow=1, padding=0)
        np_grid_image = grid_image.cpu().detach().numpy()
        heat_img = numpy.transpose(np_grid_image, (1, 2, 0)).squeeze().astype(numpy.float32)
        plt.imshow(heat_img)
        plt.axis('off')
        plt.savefig("../HEATMAP/mask.png")

    return heatmap





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


# ####################################################################

def load_train(train_path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagenet_data = MvtecDataLoader(train_path, transform=transform)
    # imagenet_data = torchvision.datasets.ImageFolder(test_path, transform=transform)

    train_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=6,
                                                    pin_memory=True)
    return train_data_loader


def load_test(test_path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagenet_data = MvtecDataLoader(test_path, transform=transform)
    # imagenet_data = torchvision.datasets.ImageFolder(test_path, transform=transform)

    test_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=6,
                                                    pin_memory=True)
    return test_data_loader


def load_test_and_mask(test_path, mask_path):
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.ToTensor(),
    ])
    imagenet_data = DualDataLoader(test_path, mask_path, test_transform, mask_transform)
    # imagenet_data = torchvision.datasets.ImageFolder(test_path, transform=transform)

    test_mask_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=0,
                                                    pin_memory=True)
    return test_mask_data_loader


def load_vaild(vaild_path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagenet_data = MvtecDataLoader(vaild_path, transform=transform)
    # imagenet_data = torchvision.datasets.ImageFolder(test_path, transform=transform)

    vaild_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=0,
                                                    pin_memory=True)
    return vaild_data_loader




AUC_ALL = []
# require_auc_plot = True
# if require_auc_plot:
#     # auc_plt = plt.figure()
#     # plt.title('MS-SSIM Baseline')#.format(mem_size, mem_lamb, NORMAL_NUM_LIST))
#     plt.plot([0, 1], [0, 1], linestyle='--')
#     plt.xlim([-0.005, 1.005])
#     plt.ylim([-0.005, 1.005])
#     plt.xlabel('FPR')
#     plt.ylabel('TPR')

all_group = None


def extract_patch(data_tmp):
    data_tmp = data_tmp.permute(0, 2, 3, 1, 4, 5)
    tmp = data_tmp.reshape(-1, 3, 32, 32)
    return tmp


def patch_score(img, generator32):
    latent_z_32 = generator32.encoder(img)
    generate_result32 = generator32(img)
    weight = 0.85
    ms_ssim_batch_wise = 1 - ms_ssim(img, generate_result32, data_range=data_range,
                                     size_average=True, win_size=3,
                                     weights=[0.0516, 0.3295, 0.3463, 0.2726])
    l1_batch_wise = l1_criterion(img, generate_result32) / data_range
    ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise

    diff = (latent_z_32 - generator32.c) ** 2
    dist = -1 * torch.sum(diff, dim=1) / generator32.sigma
    guass_svdd_loss = torch.mean(1 - torch.exp(dist))
    score_recon32 = float(ms_ssim_l1.cpu().detach().numpy())
    score_gsvdd32 = float(guass_svdd_loss.cpu().detach().numpy())
    anormaly_score32 = 0.9 * score_recon32 + 0.1 * score_gsvdd32
    return anormaly_score32


data_range = 2.1179 + 2.6400
overall_iou = []
for key in category:
    NORMAL_NUM = category[key]
    print('Current Item: {}'.format(NORMAL_NUM))

    train_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/train/'.format(NORMAL_NUM)
    test_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/test/'.format(NORMAL_NUM)
    gt_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/ground_truth/'.format(NORMAL_NUM)

    generator32 = twoin1Generator(64, latent_dimension=latent_dimension)

    from ssl_pretrain.resnet_cifar import *
    from ssl_pretrain.models import ContrastiveModel

    model = ContrastiveModel(resnet18(), mlp_number=2, cls_head_number=1)
    generator32.pretrain = model.backbone

    ckpt = torch.load('./check_points/p32/{}_32/No.{}_g.pth'.format(NORMAL_NUM, NORMAL_NUM), map_location=cuda_dev)
    generator32.load_state_dict(ckpt['model'])
    generator32.c = ckpt['c']
    generator32.sigma = ckpt['sigma']

    import sys
    sys.path.insert(0, '/media/user/T7 Touch/CVPR/')
    generator256 = torch.load('./check_points/p256/{}_256/No.{}_g.pth'.format(NORMAL_NUM, NORMAL_NUM), map_location=cuda_dev)

    # ent_criterion = NegEntropyLoss()

    generator32.to(device)
    generator256.to(device)

    generator32.eval()
    generator256.eval()

    y = []
    y_pred = []
    score = []
    score_recon = []
    score_gsvdd = []
    normal_mse_loss = []
    abnormal_mse_loss = []
    # train_root = '/home/user/Desktop/Deep Learning/public_data/MVTec_AD (with_vaild)/{}/train/'.format(NORMAL_NUM)
    # test_root = '/home/user/Desktop/Deep Learning/public_data/MVTec_AD (with_vaild)/{}/test/'.format(NORMAL_NUM)

    list_test = os.listdir(test_root)

    p_stride = 16
    with torch.no_grad():
        iou_list = []
        for i in range(len(list_test)):
            if list_test[i] == "good":
                continue
            current_defect = list_test[i]
            test_path = test_root + "{}".format(current_defect)
            mask_path = gt_root + "{}".format(current_defect)
            # valid_dataset_loader = load_test(test_path)
            test_mask_data_loader = load_test_and_mask(test_path, mask_path)
            height = 256
            width = 256
            # occ_size = 32
            # occ_stride = 16

            threshold = None
            ad_score_train = []

            for index, (images, img_label, mask, mask_label) in enumerate(test_mask_data_loader):
                # img = images.to(device)
                img_whole = images.to(device)
                img_tmp = img_whole.unfold(2, 32, p_stride).unfold(3, 32, p_stride)
                # img_tmp = img_whole.unfold(2, 32, 16).unfold(3, 32, 16)
                img = extract_patch(img_tmp)
                # heatmap = torch.zeros((1, 1, 256, 256))
                ############################################################################################
                # 32
                ############################################################################################
                # for index in range(img.shape[0]):
                latent_z_32 = generator32.encoder(img)
                generate_result32 = generator32(img)
                weight = 0.85
                _, residual_map = ms_ssim(img, generate_result32, data_range=data_range,
                                          size_average=True, win_size=3,
                                          weights=[0.0516, 0.3295, 0.3463, 0.2726])
                ms_ssim_batch_wise1 = 1 - F.interpolate(residual_map[0], size=32)
                ms_ssim_batch_wise2 = 1 - F.interpolate(residual_map[1], size=32)
                ms_ssim_batch_wise3 = 1 - F.interpolate(residual_map[2], size=32)
                ms_ssim_batch_wise4 = 1 - F.interpolate(residual_map[3], size=32)
                ms_ssim_batch_wise = ms_ssim_batch_wise1 * 0.0516 + \
                                     ms_ssim_batch_wise2 * 0.3295 + \
                                     ms_ssim_batch_wise3 * 0.3463 + \
                                     ms_ssim_batch_wise4 * 0.2726

                l1_batch_wise = torch.abs(img - generate_result32) / data_range
                ms_ssim_l1 = F.relu_(weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise)
                heatmap32 = heat_map_printer(ms_ssim_l1, img_whole, mask, plot=True)


                ############################################################################################
                # 256
                ############################################################################################
                latent_z_256 = generator256.encoder(img_whole)
                generate_result256 = generator256(img_whole)
                _, residual_map256 = ms_ssim(img_whole, generate_result256, data_range=data_range,
                                         size_average=True, win_size=11,
                                         weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
                ms_ssim_batch_wise1 = 1 - F.interpolate(residual_map256[0], size=256)
                ms_ssim_batch_wise2 = 1 - F.interpolate(residual_map256[1], size=256)
                ms_ssim_batch_wise3 = 1 - F.interpolate(residual_map256[2], size=256)
                ms_ssim_batch_wise4 = 1 - F.interpolate(residual_map256[3], size=256)
                ms_ssim_batch_wise5 = 1 - F.interpolate(residual_map256[4], size=256)
                ms_ssim_batch_wise = ms_ssim_batch_wise1 * 0.0448 +  \
                                     ms_ssim_batch_wise2 * 0.2856 + \
                                     ms_ssim_batch_wise3 * 0.3001 + \
                                     ms_ssim_batch_wise4 * 0.2363 + \
                                     ms_ssim_batch_wise5 * 0.1333

                l1_batch_wise = torch.abs(img_whole - generate_result256) / data_range
                heatmap256 = F.relu_(weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise)
                heatmap256 = heatmap256.cpu().detach().numpy().squeeze(0)
                heatmap256 = numpy.transpose(heatmap256, (1, 2, 0)).squeeze().astype(numpy.float32)
                heatmap256 = rgb2gray(heatmap256)
                # heatmap256 = heat_map_printer(ms_ssim_l1, img_whole, mask, threshold=0.5)

                ############################################################################################
                # MULTI-SCALE
                ############################################################################################
                heatmap = 0.5 * heatmap32 + 0.5 * heatmap256

                SHOW_HEATMAP = heatmap.copy()
                plt.imshow(SHOW_HEATMAP, cmap="jet", interpolation='none')
                plt.colorbar()
                plt.axis('off')
                plt.savefig("../HEATMAP/heatmap.png")

                gt_mask = mask.cpu().detach().numpy().squeeze(0)
                gt_mask = numpy.transpose(gt_mask, (1, 2, 0)).squeeze().astype(numpy.int)

                inter_sum = heatmap + gt_mask
                inter_sum[inter_sum != 2] = 0
                inter_sum[inter_sum == 2] = 1
                intersection = numpy.sum(inter_sum)

                union_sum = heatmap + gt_mask
                union_sum[union_sum != 0] = 1
                union = numpy.sum(union_sum)
                # iou = intersection / union

                auc = metrics.roc_auc_score(gt_mask.reshape(-1), heatmap.reshape(-1))
                iou_list.append(auc)

            avg_iou = sum(iou_list)/len(iou_list)
            a=1
    print("{} AUC:{}".format(NORMAL_NUM, avg_iou))
    overall_iou.append(avg_iou)
    a = 1

average_iou_all_class = sum(overall_iou) / len(overall_iou)
print("Overall PIXEL-LEVEL AUC: {}".format(average_iou_all_class))
