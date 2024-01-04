from torchvision.models import vgg19
from torch import nn
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import torch_mlu
import cv2
import numpy

import os

os.putenv('MLU_VISIBLE_DEVICES','0')

class COCODataSet(Dataset):

    def __init__(self):
        super(COCODataSet, self).__init__()
        self.zip_files = ZipFile('./data/train2014.zip')
        self.data_set = []
        for file_name in self.zip_files.namelist():
            if file_name.endswith('.jpg'):
                self.data_set.append(file_name)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        file_path = self.data_set[item]
        image = self.zip_files.read(file_path)
        image = numpy.asarray(bytearray(image), dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(numpy.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg19_pretrained = vgg19(pretrained=True).features  # 调用预训练的vgg19模型

        self.layer1 = vgg19_pretrained[:4]  # 提取预训练VGG-19模型的前4层，作为当前模型的第一层
        self.layer2 = vgg19_pretrained[4:9]  # 提取预训练VGG-19模型的第5至第9层，作为当前模型的第二层
        self.layer3 = vgg19_pretrained[9:18]  # 提取预训练VGG-19模型的第10至第18层，作为当前模型的第三层
        self.layer4 = vgg19_pretrained[18:27]  # 提取预训练VGG-19模型的第19至第27层，作为当前模型的第四层

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out1, out2, out3, out4


class ResBlock(nn.Module):

    def __init__(self, c):
        super(ResBlock, self).__init__()  # 调用父类的构造函数来初始化继承自父类的属性

        self.layer = nn.Sequential(
            # 第一层卷积，输入和输出通道数均为c，卷积核大小为3x3，步长为1，填充为1，无偏置
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),

            # 实例规范化层，针对c个输出通道，带有可学习的尺度和平移参数
            nn.InstanceNorm2d(c, affine=True),

            # ReLU激活函数，进行原地操作以节省内存
            nn.ReLU(inplace=True),

            # 第二层卷积，与第一层相同的参数设置
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),

            # 实例规范化层，与前面相同的参数设置
            nn.InstanceNorm2d(c, affine=True),
        )

    def forward(self, x):
        #TODO: 返回残差运算的结果
        return x + self.layer(x)



class TransNet(nn.Module):

    def __init__(self):
        super(TransNet, self).__init__()
        self.layer = nn.Sequential(
            
            ###################下采样层################


            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),

            nn.InstanceNorm2d(64, affine=True),

            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),

            nn.InstanceNorm2d(128, affine=True),

            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),

            nn.InstanceNorm2d(256, affine=True),

            nn.ReLU(inplace=True),

            ##################残差层##################
            ResBlock(256),  # 不要让残差层的数量太多

            ################上采样层##################

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),# 使用torch.nn.Upsample对特征图进行上采样，将特征图的尺寸放大两倍，使用双线性插值，不对角点进行对齐


            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),# 卷积层，将输入通道数从256减少到128，卷积核大小为3x3，步长为1，填充为1，无偏置


            nn.InstanceNorm2d(128, affine=True),# 实例规范化层，针对128个输出通道，带有可学习的尺度和平移参数


            nn.ReLU(inplace=True),# ReLU激活函数，进行原地操作以节省内存


            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),# 再次使用torch.nn.Upsample进行上采样，放大特征图尺寸


            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),# 另一个卷积层，将输入通道数从128减少到64


            nn.InstanceNorm2d(64, affine=True),# 实例规范化层，针对64个输出通道，带有可学习的尺度和平移参数


            nn.ReLU(inplace=True),# ReLU激活函数，进行原地操作以节省内存

            ###############输出层#####################
            # 执行卷积操作
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            # sigmoid激活函数
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)


def load_image(path):
    image = cv2.imread(path)  # 使用OpenCV读取指定路径的图像

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR颜色空间转换为RGB颜色空间

    image = cv2.resize(image, (512, 512))  # 调整图像大小到512x512像素

    image = image.astype(numpy.float32) / 255.0  # 将图像数据类型转换为浮点数，并归一化到0-1之间

    image = torch.from_numpy(image)  # 将NumPy数组转换为PyTorch张量
    
    image = image.permute(2, 0, 1).unsqueeze(0)  # 改变张量的维度顺序，并增加一个维度，从高度x宽度x通道数转换为1x通道数x高度x宽度

    return image


def get_gram_matrix(f_map):
    """
    获取格拉姆矩阵
    :param f_map:特征图
    :return:格拉姆矩阵，形状（通道数,通道数）
    """
    n, c, h, w = f_map.shape
    if n == 1:
        f_map = f_map.reshape(c, h * w)
        gram_matrix = torch.mm(f_map, f_map.t())
        return gram_matrix
    else:
        f_map = f_map.reshape(n, c, h * w)
        gram_matrix = torch.matmul(f_map, f_map.transpose(1, 2))
        return gram_matrix


if __name__ == '__main__':
    image_style = load_image('./data/udnie.jpg').cpu()
    #TODO: 将输入的风格图像加载到mlu设备上,得到mlu_iamge_style
    mlu_image_style = image_style.to('mlu')
    net = VGG19().cpu()
    g_net = TransNet().cpu()
    #TODO: 将特征网络加载到mlu得到mlu_g_net
    mlu_net = net.to('mlu')
    mlu_g_net = g_net.to('mlu')
    #TODO: 将图像转换网络加载到mlu得到mlu_net
    mlu_net = mlu_net.to('mlu')
    mlu_g_net = mlu_g_net.to('mlu')
    print("mlu_net build PASS!\n")
    #TODO: 使用adam优化器对mlu_g_net的参数进行优化
    optimizer = torch.optim.Adam(mlu_g_net.parameters(), lr=0.001)
    #TODO: 在cpu上计算均方误差损失函得到loss_func
    loss_func = nn.MSELoss()
    #TODO: 将损失函数加载到mlu上得到mlu_loss_func
    mlu_loss_func = loss_func.to('mlu')
    print("build loss PASS!\n")
    data_set = COCODataSet()
    print("load COCODataSet PASS!\n")
    batch_size = 1
    data_loader = DataLoader(data_set, batch_size, True, drop_last=True)
    #TODO：mlu_iamge_style经过特征提取网络mlu_net生成风格特征s1-s4
    with torch.no_grad():
        mlu_features = mlu_net(mlu_image_style)
    #TODO: 对风格特征s1-s4计算格拉姆矩阵并从当前计算图中分离下来，得到对应的s1-s4
    gram_matrices = [get_gram_matrix(feature).detach() for feature in mlu_features]
    s1, s2, s3, s4 = gram_matrices
    j = 0
    count = 0
    epochs = 1
    while j <= epochs:
        for i, image in enumerate(data_loader):
            image_c = image.cpu()
            #TODO: 将输入图像拷贝到mlu上得到mlu_image_c
            mlu_image_c = image_c.to('mlu')
            #TODO: 将mlu_image_c经过mlu_g_net输出生成图像mlu_imge_g
            mlu_image_g = mlu_g_net(mlu_image_c)
            #TODO: 利用特征提取网络mlu_net提取生成图像mlu_image_g的特征out1-out4
            with torch.no_grad():
                mlu_features_g = mlu_net(mlu_image_g)
            ##############计算风格损失#################
            #TODO: 对生成图像的特征out1-out4计算gram矩阵，并与风格图像的特征求损失，分别得到loss_s1-loss_s4

            gram_matrices_g = [get_gram_matrix(feature).detach() for feature in mlu_features_g]
            loss_s1 = mlu_loss_func(gram_matrices_g[0], s1)
            loss_s2 = mlu_loss_func(gram_matrices_g[1], s2)
            loss_s3 = mlu_loss_func(gram_matrices_g[2], s3)
            loss_s4 = mlu_loss_func(gram_matrices_g[3], s4)
            #TODO：loss_s1-loss_s4相加得到风格损失loss_s
            loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4

            ##############计算内容损失#################
            #TODO: 将图片mlu_image_c经过特征提取网络mlu_net得到内容特图像的特征c1-c4

            mlu_features_c = mlu_net(mlu_image_c)
            c2 = mlu_features_c[1]

             #TODO: 将内容图像特征c2从计算图中分离并与内容图像特征out2求内容损失loss_c2
            loss_c2 = mlu_loss_func(c2, mlu_features_g[1])
            loss_c = loss_c2

            ##############计算总损失#################
            loss = loss_c + 0.000000005 * loss_s

            ########清空梯度、计算梯度、更新参数######
            #TODO: 梯度初始化为零
            optimizer.zero_grad()
            #TODO: 反向传播求梯度
            loss.backward()
            #TODO: 更新所有参数
            optimizer.step()
            print('j:',j, 'i:',i, 'loss:',loss.item(), 'loss_c:',loss_c.item(), 'loss_s:',loss_s.item())
            count += 1
            mlu_image_g = mlu_image_g.cpu()
            mlu_image_c = mlu_image_c.cpu()
            if i % 5 == 0:
                #TODO: 将图像转换网络fst_train_mlu.pth的参数存储在models/文件夹下
                torch.save(mlu_g_net.state_dict(), 'models/fst_train_mlu.pth')
                #TODO: 利用save_image函数将tensor形式的生成图像mlu_image_g以及输入图像mlu_image_c以jpg左右拼接的形式保存在/out/train_mlu/文件夹下
                save_image(torch.cat([mlu_image_c, mlu_image_g], dim=3), f'./out/train_mlu/{count}.jpg')
        j += 1

print("MLU TRAIN RESULT PASS!\n")

