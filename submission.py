import argparse
import cv2
import math
from models import hsm
import numpy as np
import os
import pdb
import skimage.io
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
from models.submodule import *
from utils.eval import mkdir_p, save_pfm
from utils.preprocess import get_transform
#cudnn.benchmark = True
cudnn.benchmark = False

parser = argparse.ArgumentParser(description='HSM')
parser.add_argument('--datapath', default='./data-mbtest/',
                    help='test data path')
# parser.add_argument('--loadmodel', default='/root/high-res-stereo/logname/finetune_999.tar',
#                     help='model path')
parser.add_argument('--loadmodel', default='/disk2/users/M22_zhaoqinghao/high-res-stereo/logname/final-768px.tar',
                    help='model path')
parser.add_argument('--outdir', default='output',
                    help='output dir')
parser.add_argument('--clean', type=float, default=-1,
                    help='clean up output using entropy estimation')
parser.add_argument('--testres', type=float, default=0.5,
                    help='test time resolution ratio 0-x')
parser.add_argument('--max_disp', type=float, default=-1,
                    help='maximum disparity to search for')
parser.add_argument('--level', type=int, default=1,
                    help='output level of output, default is level 1 (stage 3),\
                          can also use level 2 (stage 2) or level 3 (stage 1)')
args = parser.parse_args()



# dataloader
from dataloader import listfiles as DA
test_left_img, test_right_img, _, _ = DA.dataloader(args.datapath)

# construct model，构建模型
model = hsm(128,args.clean,level=args.level) 
# model = nn.DataParallel(model, device_ids=[0])
model = nn.DataParallel(model, device_ids=[0, 1]) # use 2 GPUs
model.cuda() # use CUDA

# 将预训练模型的参数加载到神经网络模型中
# 以便在测试或使用模型时使用预训练的参数
if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    # 去除预训练模型中与视差有关的参数
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# dry run, 用于预热GPU，以便在测试时获得更准确的计时
multip = 48
# 四维矩阵，第一维表示图片数量，第二维表示通道数，第三维表示高度，第四维表示宽度？
imgL = np.zeros((1,3,24*multip,32*multip))
imgR = np.zeros((1,3,24*multip,32*multip))
# 图像转换为PyTorch张量，并将其移动到GPU上进行计算。
# 具体来说，`imgL`和`imgR`是两个包含图像数据的变量，
# 这些数据最初以NumPy数组的形式存储。
# 使用`torch.FloatTensor()`函数将这些数组转换为PyTorch张量。
# 使用`Variable()`函数将这些张量包装成PyTorch变量
# 使用`.cuda()`函数将这些变量移动到GPU上进行计算。
imgL = Variable(torch.FloatTensor(imgL).cuda())
imgR = Variable(torch.FloatTensor(imgR).cuda())
# 禁用梯度计算，以提高计算速度和效率
with torch.no_grad():
    model.eval() # 设置为评估模式
    pred_disp,entropy = model(imgL,imgR) # 视差图: pred_disp 不确定性图: entropy

def main():
    processed = get_transform()
    model.eval()
    # 遍历测试集中的每一张图片
    for inx in range(len(test_left_img)):
        # 打印当前处理的图片路径
        print(test_left_img[inx])
        # 读取图片，转为float32类型，获取图像的RGB通道
        imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))[:,:,:3]
        imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))[:,:,:3]

        # # mask 用于去除像素值为 0 的区域
        # mask = (imgL_o[:,:,0]!=0).astype(np.float32)
        # imgL_o[:,:,0] = imgL_o[:,:,0]*mask
        # imgL_o[:,:,1] = imgL_o[:,:,1]*mask
        # imgL_o[:,:,2] = imgL_o[:,:,2]*mask


        # 获取图像的尺寸
        imgsize = imgL_o.shape[:2]

        # 读取图片的最大视差
        if args.max_disp>0:
            if args.max_disp % 16 != 0:
                args.max_disp = 16 * math.floor(args.max_disp/16)
            max_disp = int(args.max_disp)
        else:
            with open(test_left_img[inx].replace('im0.png','calib.txt')) as f:
                lines = f.readlines()
                max_disp = int(int(lines[6].split('=')[-1]))

        # change max disp
        tmpdisp = int(max_disp*args.testres//64*64) # // 整数除法，64的倍数
        if (max_disp*args.testres/64*64) > tmpdisp:
            model.module.maxdisp = tmpdisp + 64
        else:
            model.module.maxdisp = tmpdisp
        if model.module.maxdisp ==64: model.module.maxdisp=128
        # 获取视差回归对象，将模型输出的视差图像转换为深度图像
        model.module.disp_reg8 =  disparityregression(model.module.maxdisp,16).cuda()
        model.module.disp_reg16 = disparityregression(model.module.maxdisp,16).cuda()
        model.module.disp_reg32 = disparityregression(model.module.maxdisp,32).cuda()
        model.module.disp_reg64 = disparityregression(model.module.maxdisp,64).cuda()
        print(model.module.maxdisp)
        
        # resize test img to testres
        imgL_o = cv2.resize(imgL_o,None,fx=args.testres,fy=args.testres,interpolation=cv2.INTER_CUBIC)
        imgR_o = cv2.resize(imgR_o,None,fx=args.testres,fy=args.testres,interpolation=cv2.INTER_CUBIC)
        # 归一化, processed = get_transform()
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()

        imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
        imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

        ##fast pad
        max_h = int(imgL.shape[2] // 64 * 64)   # 最大高度
        max_w = int(imgL.shape[3] // 64 * 64)   # 最大宽度
        if max_h < imgL.shape[2]: max_h += 64
        if max_w < imgL.shape[3]: max_w += 64

        top_pad = max_h-imgL.shape[2]
        left_pad = max_w-imgL.shape[3]
        imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

        # test
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        # 禁用梯度计算，以提高计算速度和效率
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            pred_disp,entropy = model(imgL,imgR) # 调用模型，获取视差图和不确定性图，model = hsm(128,args.clean,level=args.level) 
            torch.cuda.synchronize()
            ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )
        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad   = max_h-imgL_o.shape[0]
        left_pad  = max_w-imgL_o.shape[1]
        entropy = entropy[top_pad:,:pred_disp.shape[1]-left_pad].cpu().numpy()
        pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]

        # save predictions
        idxname = test_left_img[inx].split('/')[-2]
        if not os.path.exists('%s/%s'%(args.outdir,idxname)):
            os.makedirs('%s/%s'%(args.outdir,idxname))
        idxname = '%s/disp0HSM'%(idxname)

        # resize to highres
        pred_disp = cv2.resize(pred_disp/args.testres,(imgsize[1],imgsize[0]),interpolation=cv2.INTER_LINEAR)

        # clip while keep inf
        invalid = np.logical_or(pred_disp == np.inf,pred_disp!=pred_disp)
        pred_disp[invalid] = np.inf

        np.save('%s/%s-disp.npy'% (args.outdir, idxname.split('/')[0]),(pred_disp))
        np.save('%s/%s-ent.npy'% (args.outdir, idxname.split('/')[0]),(entropy))
        cv2.imwrite('%s/%s-disp.png'% (args.outdir, idxname.split('/')[0]),pred_disp/pred_disp[~invalid].max()*255)
        cv2.imwrite('%s/%s-ent.png'% (args.outdir, idxname.split('/')[0]),entropy/entropy.max()*255)

        with open('%s/%s.pfm'% (args.outdir, idxname),'w') as f:
            save_pfm(f,pred_disp[::-1,:])
        with open('%s/%s/timeHSM.txt'%(args.outdir,idxname.split('/')[0]),'w') as f:
             f.write(str(ttime))
            
        torch.cuda.empty_cache()

# code that will only be executed when the script is run as the main program
if __name__ == '__main__':
    main()

