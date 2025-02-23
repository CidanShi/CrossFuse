from time import time
from network import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import os
import numpy as np
import cv2
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save, image_read_cv2
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path='./models/CrossFuse_RSM3.pth'
for dataset_name in ["RoadScene","TNO","MSRS"]:
    print("\n"*2+"="*80)
    model_name="CrossFuse    "
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('./test_image/',dataset_name)
    test_out_folder=os.path.join('./test_results/',dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    with torch.no_grad():
        sum = 0
        count = 0
        for img_name in os.listdir(os.path.join(test_folder,"ir")):

            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = cv2.split(image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='YCrCb'))[0][
                           np.newaxis, np.newaxis, ...] / 255.0

            # ycrcb, uint8
            data_VIS_BGR = cv2.imread(os.path.join(test_folder, "vi", img_name))
            _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))


            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            # add
            start_time = time()

            feature_V_L, feature_V_H, feature_V = Encoder(data_VIS)
            feature_I_L, feature_I_H, feature_I = Encoder(data_IR)
            feature_F_L = BaseFuseLayer(feature_V_L + feature_I_L)
            feature_F_H = DetailFuseLayer(feature_V_H + feature_I_H)

            Fuse, _ = Decoder(data_VIS, feature_F_L, feature_F_H)
            Fuse=(Fuse-torch.min(Fuse))/(torch.max(Fuse)-torch.min(Fuse))

            end_time = time()
            elapsed_time = end_time - start_time
            sum += elapsed_time
            count +=1

            fuse = np.squeeze((Fuse * 255).cpu().numpy())

            # float32 to uint8
            fuse = fuse.astype(np.uint8)
            # concatnate to get rgb results
            ycrcb_fuse = np.dstack((fuse, data_VIS_Cr, data_VIS_Cb))
            rgb_fuse = cv2.cvtColor(ycrcb_fuse, cv2.COLOR_YCrCb2RGB)
            img_save(rgb_fuse, img_name.split(sep='.')[0], test_out_folder)
        if count > 0:
            print(f'total processing images: {count}')
            print(f'total processing time: {round(sum, 4)} seconds')
            print(f'average processing time: {round(sum / count, 4)} seconds')
        else:
            print('no images processed')
