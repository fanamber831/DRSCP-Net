import torch
import os
import SimpleITK as sitk 
import pickle
# import pydicom
import numpy as np
import datetime
import pandas as pd
import json 
import random

def mkdir(p, is_file=False):
    if is_file:                                               # 如果 is_file 为 True
        p, _ =  os.path.split(p)                              # 将路径 p 分割成目录和文件名，保留目录路径 p
    isExists = os.path.exists(p)                              # 检查路径 p 是否存在
    if isExists:                                              # 如果路径 p 已存在
        pass                                                  # 不做任何操作
    else:                                                     # 如果路径 p 不存在
        os.makedirs(p)                                        # 创建路径 p
        print("make directory successfully:{}".format(p))     # 打印创建目录成功的信息


class transformPET:
    def __init__(self, data_range, mode="linear"):
        self.r = data_range                                   # 保存数据范围
        self.m = mode                                         # 保存模式（默认为 "linear"）

    def cut(self,x):
        thresh=self.r                                         # 阈值为数据范围
        x[x>thresh]=thresh                                    # 将大于阈值的元素设为阈值
        x[x<=0]=0.0                                           # 将小于等于 0 的元素设为 0.0
        _, _, d,h,w=x.shape                                   # 获取 x 的形状
        x = x[:, :, 4:d-4, 16:h-16, 16:w-16]                  # 截取 x 的中心部分
        return x                                              # 返回处理后的 x
 
    def clip(self,x):

        x[x>self.r]=self.r                                    # 将大于数据范围的元素设为数据范围
        x[x<=0]=0.0                                           # 将小于等于 0 的元素设为 0.0
        return x 
    
    def get_error_mask(self, low, full, error_thresh=0.01):
        error = torch.abs(full-low)                           # 计算绝对误差
        mask = (error>error_thresh).type(torch.FloatTensor).to(full.device)   # 生成误差掩码（误差大于阈值的部分设为 1，其他部分设为 0）
        return mask                                           # 返回误差掩码
    
    def normalize(self, img, cut=False):
        img = self.clip(img)                                  # 调用 clip 方法处理 img
        if self.m == "exp":
            c = torch.log(torch.Tensor([self.r+1])).to(img.device)            # 计算对数基数
            img = torch.log(1+img)/c                          # 对 img 进行对数归一化
        else:
            img = img/self.r                                  # 进行线性归一化
        
        if cut:
            img = self.cut(img)                               # 如果 cut 为 True，调用 cut 方法处理 img
        return img
    def denormalize(self, img):
        if self.m == 'exp':
            c = torch.log(torch.tensor([self.r + 1.0])).to(img.device)       # 计算对数基数
            img = torch.exp(c * img) - 1                      # 对 img 进行对数反归一化
        else:
            img *= self.r                                     # 进行线性反归一化
        img = self.clip(img)                                  # 调用 clip 方法处理 img
        return img



class transformData: 
    
    '''
    all-in-one medical image data
    '''
    
    def __init__(self, data_range=None):
        self.r = data_range                                            # 保存数据范围
        self.data_range = { 
            "CT":[-1024.0, 3072.0],                                    # CT 图像的数据范围
            "PET":[0.0, 20.0],                                         # PET 图像的数据范围
            "low-resolution":[0.0, 2095.0],                            # MRI 图像的数据范围
            } 
        
        
        '''
        Abdomen CT image is truncated to [-160, 240] 
        reference: https://github.com/SSinyu/WGAN-VGG
        '''
        
        self.test_trucate_data_range = { 
            "CT":[-160.0, 240.0],                                      # 腹部 CT 图像的截断范围
            "PET":[0.0, 20.0],                                         # PET 图像的截断范围
            "low-resolution":[0.0, 2095.0],                            # MRI 图像的截断范围
            }


    def truncate(self,img, d_min, d_max):
        img[img>d_max]=d_max                                           # 将大于最大值的像素值设为最大值
        img[img<d_min]=d_min                                           # 将小于最小值的像素值设为最小值
        return img 
    
    def truncate_test(self, img, modality): 
        d_min, d_max = self.test_trucate_data_range[modality]          # 获取特定模态的截断范围
        img[img>d_max]=d_max                                           # 将大于最大值的像素值设为最大值
        img[img<d_min]=d_min                                           # 将小于最小值的像素值设为最小值
        return img
    
    def normalize(self, img, modality): 
        d_min, d_max = self.data_range[modality]                       # 获取特定模态的数据范围
        img = self.truncate(img, d_min, d_max)                         # 调用 truncate 方法对图像进行截断处理
        img = (img - d_min)/(d_max - d_min)                            # 对图像进行归一化处理
        return img
    
    def denormalize(self, img, modality): 
        d_min, d_max = self.data_range[modality]                       # 获取特定模态的数据范围
        img = img*(d_max - d_min) + d_min                              # 对图像进行反归一化处理
        img = self.truncate(img, d_min, d_max)                         # 调用 truncate 方法对图像进行截断处理
        return img

    def random_crop(self, tensor, patch_size):
        """
        从给定的图像张量中随机裁剪大小为patch_size的patch。
    
        参数:
        tensor: 形状为[B, C, H, W]的图像张量。
        patch_size: 裁剪patch的大小,格式为(H, W)。
    
        返回:
        裁剪后的patch张量。
        """
        B, C, H, W = tensor.shape
        patch_h, patch_w = patch_size
    
        # 确保裁剪尺寸不大于原图像尺寸
        if patch_h > H or patch_w > W:
            raise ValueError("裁剪尺寸应小于原始图像尺寸")
    
        # 随机选择裁剪的起始点
        top = random.randint(0, H - patch_h)
        left = random.randint(0, W - patch_w)
    
        # 裁剪patch
        patches = tensor[:, :, top:top + patch_h, left:left + patch_w]
        return patches


    def random_rotate_flip(self, tensor):
        """
        对形状为[B, C, H, W]的图像张量执行随机旋转或翻转。
    
        参数:
        tensor: 形状为[B, C, H, W]的图像张量。
    
        返回:
        经过随机旋转或翻转的图像张量。
        """
        B, C, H, W = tensor.shape
        processed = torch.empty_like(tensor)
    
        for i in range(B):
            img = tensor[i]
            operation = torch.randint(0, 6, (1,)).item()
    
            if operation == 1:
                # 水平翻转
                img = torch.flip(img, [2])
            elif operation == 2:
                # 垂直翻转
                img = torch.flip(img, [1])
            elif operation == 3:
                # 旋转90度
                img = img.transpose(1, 2).flip(2)
            elif operation == 4:
                # 旋转180度
                img = img.flip(1).flip(2)
            elif operation == 5:
                # 旋转270度
                img = img.transpose(1, 2).flip(1)
    
            # 不做改变的情况下，operation == 0
            processed[i] = img
    
        return processed


    def _add_gaussian_noise(self, clean_patch, sigma):
        # 将sigma从[0,255]范围转换到[0,1]范围
        sigma = sigma / 255.0                                                               # 将噪声标准差从 [0, 255] 范围转换到 [0, 1] 范围
        noise = torch.randn_like(clean_patch)                                               # 生成与 clean_patch 形状相同的随机噪声
        noisy_patch = torch.clamp(clean_patch + noise * sigma, 0, 1)                        # 将噪声添加到 clean_patch 上，并将结果限制在 [0, 1] 范围内
        return noisy_patch, clean_patch                                                     # 返回添加噪声后的图像和原始图像

    def _degrade_by_type(self, clean_patch, degrade_type):
        if degrade_type == 0:
            # denoise sigma=15
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 1:
            # denoise sigma=25
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 2:
            # denoise sigma=50
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)

        return degraded_patch, clean_patch

    def degrade(self, clean_patches, degrade_type=None):
        if degrade_type is None:
            degrade_type = random.randint(0, 2)                                             # 随机选择退化类型

        B, C, H, W = clean_patches.shape                                                    # 获取批量、通道、高度和宽度
        degraded_patches = torch.empty_like(clean_patches)                                  # 创建与 clean_patches 形状相同的空张量

        for i in range(B):
            degraded_patches[i], _ = self._degrade_by_type(clean_patches[i], degrade_type)  # 逐个处理每个图像

        return degraded_patches




@torch.no_grad()
def synthesisOneAxial(model, img, kernel_size=64, stride=32, crop_size = 3):
    model.eval()                                                                            # 将模型设置为评估模式
    B, C, D, H, W = img.shape                                                               # 获取图像的形状（批量大小、通道数、深度、高度、宽度）
    nz = int(D//stride)                                                                     # 计算沿深度方向的分块数
    nx = int(H//stride)-1                                                                   # 计算沿高度方向的分块数
    ny = int(W//stride)-1                                                                   # 计算沿宽度方向的分块数
    result = torch.zeros((B, C, D, H, W)).type(torch.FloatTensor).to(img.device)            # 创建一个与输入图像形状相同的空张量，用于存储结果
    flag=True                                                                               # 初始化标志位，用于处理最后一个块
    for k in range(nz):
        idz = 0 if k==0 else k*stride+kernel_size-stride-crop_size                          # 计算当前块在深度方向的起始索引
        if idz+crop_size+stride==D:                                                         # 如果当前块正好覆盖整个深度范围，跳出循环
            break
        elif idz+crop_size+stride>D:                                                        # 如果当前块超出深度范围，将标志位设置为 False
            flag=False
        x = img[:,:,k*stride:k*stride+kernel_size,:,:] if flag else img[:,:,D-kernel_size:,:,:]##Large patches along z axis # 提取沿 z 轴的大块
        patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)  # 将大块沿 z, y, x 轴分块
        patches=patches.reshape(-1, C, kernel_size, kernel_size, kernel_size)                                     # 重塑分块为 1D 列表
        ######
        #Synthesis
        ######
        G_patches = model(patches)                                                          # 使用模型处理分块
        for i in range(nx):
            idx = 0 if i==0 else i*stride+kernel_size-stride-crop_size                      # 计算当前块在高度方向的起始索引
            for j in range(ny):
                idy = 0 if j==0 else j*stride+kernel_size-stride-crop_size                  # 计算当前块在宽度方向的起始索引
                if flag:
                    result[:,:,idz:k*stride+kernel_size,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz-k*stride:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
                else:
                    result[:,:,idz:,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz+kernel_size-D:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
    return result

@torch.no_grad()
def merge_patch(model, img, kernel_size=64, stride=32,crop_size = 3):
    model.eval()                                                                            # 将模型设置为评估模式
    B, C, D, H, W = img.shape
    nz = int(D//stride)
    nx = int(H//stride)-1
    ny = int(W//stride)-1
    result = torch.zeros((B, C, D, H, W)).type(torch.FloatTensor).to(img.device)            # 创建一个与输入图像形状相同的空张量，用于存储结果
    flag=True                                                                               # 初始化标志位，用于处理最后一个块
    for k in range(nz):
        idz = 0 if k==0 else k*stride+kernel_size-stride-crop_size                          # 计算当前块在深度方向的起始索引
        if idz+crop_size+stride==D:                                                         # 如果当前块正好覆盖整个深度范围，跳出循环
            break
        elif idz+crop_size+stride>D:                                                        # 如果当前块超出深度范围，将标志位设置为 False
            flag=False
        x = img[:,:,k*stride:k*stride+kernel_size,:,:] if flag else img[:,:,D-kernel_size:,:,:]##Large patches along z axis # 提取沿 z 轴的大块
        patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)  # 将大块沿 z, y, x 轴分块
        patches=patches.reshape(-1, C, kernel_size, kernel_size, kernel_size)                                     # 重塑分块为 1D 列表
        #Synthesis
        ###### 
        G_patches = patches.clone().detach()                                                # 克隆并分离 patches，用于存储生成的块
        for i in range(len(patches)):
            G_patches[i] = model(patches[[i],:,:,:,:])                                      # 对每个分块进行处理
        for i in range(nx):
            idx = 0 if i==0 else i*stride+kernel_size-stride-crop_size                      # 计算当前块在高度方向的起始索引
            for j in range(ny):
                idy = 0 if j==0 else j*stride+kernel_size-stride-crop_size                  # 计算当前块在宽度方向的起始索引
                if flag:
                    result[:,:,idz:k*stride+kernel_size,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz-k*stride:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
                else:
                    result[:,:,idz:,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz+kernel_size-D:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
    return result

class dataIO:
    def __init__(self):
        self.reader = {
            '.img':self.load_itk,
            '.gz':self.load_itk, 
            '.nii':self.load_itk,
            '.bin':self.load_bin, 
            '.txt':self.load_txt, 
            '.json':self.load_json
            
            }
        self.writer = {
            '.img':self.save_itk, 
            '.gz':self.save_itk, 
            '.nii':self.save_itk,
            '.bin':self.save_bin,
            '.csv':self.save_csv,
            '.txt':self.save_txt, 
            '.txt':self.save_json
            }
    
    
    def save_itk(self, data, path, use_int=False): 
        if use_int: 
            data = np.around(data)                                 # 如果 use_int 为真，将数据取整
        sitk.WriteImage(sitk.GetImageFromArray(data), path)        # 将数据保存为 ITK 格式

    def save_bin(self, data, path, use_int=False): 
        with open(path, "wb") as f:
            pickle.dump(data, f)                                   # 将数据保存为二进制文件
        
    def load_itk(self,path):
        return sitk.GetArrayFromImage(sitk.ReadImage(path))        # 从 ITK 格式文件中加载数据
        
    def load_bin(self,path):
        with open(path, "rb") as f:
            data = pickle.load(f)                                  # 从二进制文件中加载数据
        return data 
    def load_txt(self, path):
        with open(path, "r") as f:
            data = f.read()                                        # 从文本文件中加载数据
        return data



    def save_json(self, data, path):
        with open(path, "w", encoding='utf8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)       # 将数据保存为 JSON 格式

    def load_json(self, path):
        with open(path, encoding='utf8') as f:
            data = json.load(f)                                    # 从 JSON 文件中加载数据
        return data 

    def save_csv(self, data_dict, path):
        result=pd.DataFrame({ key:pd.Series(value) for key, value in data_dict.items() })
        result.to_csv(path)                                        # 将数据字典保存为 CSV 文件
    
    def save_txt(self, s, path):
        with open(path,'w') as f:
            f.write(s)                                             # 将字符串保存为文本文件

        
    def getFileEX(self, s):
        _, tempfilename = os.path.split(s)
        _, ex = os.path.splitext(tempfilename)
        return ex                                                  # 获取文件扩展名
    
    def load(self, path):
        ex = self.getFileEX(path)                                  # 获取文件扩展名
        return self.reader[ex](path)                               # 根据扩展名选择合适的读取方法并加载文件
    
    def save(self, data, path): 
        mkdir(path, is_file=True)                                  # 确保目录存在
        ex = self.getFileEX(path)                                  # 获取文件扩展名
        return self.writer[ex](data, path)                         # 根据扩展名选择合适的保存方法并保存文件




