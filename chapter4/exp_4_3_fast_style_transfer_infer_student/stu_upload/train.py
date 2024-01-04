# coding=utf-8
from torchvision.models import vgg19
import numpy
from torch import nn
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import cv2
import torch.optim as optim



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
        # 从数据集中获取指定索引的文件路径
        file_path = self.data_set[item]
        # 从压缩文件中读取图像数据
        image = self.zip_files.read(file_path)
        # 将读取的字节转换为NumPy数组
        image = numpy.asarray(bytearray(image), dtype='uint8')

        image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # # 使用OpenCV解码图像数据
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(numpy.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
    


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        #TODO: 调用vgg19网络
        vgg19_pretrained = vgg19(pretrained=True).features  # 调用预训练的vgg19模型

        self.layer1 = nn.Sequential(*list(vgg19_pretrained.children())[:7])  # 提取VGG-19的前7层，作为当前模型的第一层
        self.layer2 = nn.Sequential(*list(vgg19_pretrained.children())[7:14])  # 提取VGG-19的第8到第14层，作为当前模型的第二层
        self.layer3 = nn.Sequential(*list(vgg19_pretrained.children())[14:27])  # 提取VGG-19的第15到第27层，作为当前模型的第三层
        self.layer4 = nn.Sequential(*list(vgg19_pretrained.children())[27:40])  # 提取VGG-19的第28到第40层，作为当前模型的第四层

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out1, out2, out3, out4


class ResBlock(nn.Module):

    def __init__(self, c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            # 进行卷积，卷积核为3*1*1
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
            # 执行实例归一化
            nn.InstanceNorm2d(c, affine=True),
            # 执行ReLU
            nn.ReLU(inplace=True),
            # 进行卷积，卷积核为3*1*1
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
            # 执行实例归一化
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

            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),# 第一层卷积，输入通道数为3，输出通道数为64，卷积核大小为3x3，步长为1，填充为1，无偏置


            nn.InstanceNorm2d(64, affine=True),  # 实例规范化层，针对64个输出通道，带有可学习的尺度和平移参数
            nn.ReLU(inplace=True),  # ReLU激活函数，进行原地操作以节省内存

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 第二层卷积，输入通道数为64，输出通道数为128，卷积核大小为3x3，步长为2，填充为1，无偏置


            nn.InstanceNorm2d(128, affine=True),  # 实例规范化层，针对128个输出通道，带有可学习的尺度和平移参数

            nn.ReLU(inplace=True),  # ReLU激活函数，进行原地操作以节省内存

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),# 第三层卷积，输入通道数为128，输出通道数为256，卷积核大小为3x3，步长为2，填充为1，无偏置


            nn.InstanceNorm2d(256, affine=True),  # 实例规范化层，针对256个输出通道，带有可学习的尺度和平移参数

            nn.ReLU(inplace=True),  # ReLU激活函数，进行原地操作以节省内存

            ##################残差层##################
            ResBlock(256),  # 不太太多残差层

            ################上采样层##################
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),  # 卷积

            nn.InstanceNorm2d(128, affine=True),  # 实例化

            nn.ReLU(inplace=True),  # RELU激活


            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 上采样

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 卷积层

            nn.InstanceNorm2d(64, affine=True),

            nn.ReLU(inplace=True),
            
            ###############输出层#####################
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),

            nn.Sigmoid()  # sigmoid激活函数
        )

    def forward(self, x):
        return self.layer(x)


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    image = image.astype(numpy.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
    return image


def get_gram_matrix(f_map):
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
    net = VGG19().cpu()
    g_net = TransNet().cpu()  # 图像转换网络
    print("g_net build PASS!\n")

    optimizer = optim.Adam(g_net.parameters(), lr=0.001)

    loss_func = nn.MSELoss()

    print("build loss PASS!\n")
    data_set = COCODataSet()
    print("load COCODataSet PASS!\n")
    batch_size = 1
    data_loader = DataLoader(data_set, batch_size, True, drop_last=True)

    s1, s2, s3, s4 = net(image_style)  # 将风格图像通过网络(net)传递，得到四个不同层的特征图(s1, s2, s3, s4)

    gram_s1 = get_gram_matrix(s1).detach()  # 计算第一层特征图的Gram矩阵，并从当前计算图中分离
    gram_s2 = get_gram_matrix(s2).detach()  # 计算第二层特征图的Gram矩阵，并从当前计算图中分离
    gram_s3 = get_gram_matrix(s3).detach()  # 计算第三层特征图的Gram矩阵，并从当前计算图中分离
    gram_s4 = get_gram_matrix(s4).detach()  # 计算第四层特征图的Gram矩阵，并从当前计算图中分离

    j = 0
    count = 0
    epochs = 1
    while j <= epochs:
        for i, image in enumerate(data_loader):
            image_c = image.cpu()
            # print("image_c:", image_c.size())
            image_g = g_net(image_c)
            # print("image_g: ",image_g.size())

            out1, out2, out3, out4 = net(image_g)

            ###############计算风格损失###################
            #TODO: 对生成图像的特征out1-out4计算gram矩阵，并与风格图像的特征s1-s4通过loss_func求损失，分别得到loss_s1-loss_s4
            gram_out1 = get_gram_matrix(out1)
            gram_out2 = get_gram_matrix(out2)
            gram_out3 = get_gram_matrix(out3)
            gram_out4 = get_gram_matrix(out4)

            loss_s1 = loss_func(gram_out1, gram_s1)
            loss_s2 = loss_func(gram_out2, gram_s2)
            loss_s3 = loss_func(gram_out3, gram_s3)
            loss_s4 = loss_func(gram_out4, gram_s4)
            #TODO：loss_s1-loss_s4相加得到风格损失loss_s
            loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4

            ###############计算内容损失###################
            c1, c2, c3, c4 = net(image_c)
            # print("c2:",c2.size(),"out2:",out2.detach().size())
            loss_c2 = loss_func(c2, out2.detach())  # 计算内容损失
            loss_c = loss_c2

            ###############计算总损失###################
            loss = loss_c + 0.000000005 * loss_s

            #######清空梯度、计算梯度、更新参数###########
            # TODO: 梯度初始化为零
            optimizer.zero_grad()
            # TODO: 反向传播求梯度
            loss.backward()
            # TODO: 更新所有参数
            optimizer.step()
            print('j:',j, 'i:',i, 'loss:',loss.item(), 'loss_c:',loss_c.item(), 'loss_s:',loss_s.item())
            count += 1
            if i % 5 == 0:

                #TODO: 将图像转换网络fst_train_mlu.pth的参数存储在models/文件夹下
                torch.save(g_net.state_dict(), './models/fst_train.pth')

                # TODO: 利用save_image函数将tensor形式的生成图像mlu_image_g以及输入图像mlu_image_c以jpg左右拼接的形式保存在/out/train_mlu/文件夹下
                save_image(torch.cat((image_g, image_c), dim=3), f'./out/train/result_{count}.jpg')
                break

        j += 1

    print("TRAIN RESULT PASS!\n")
