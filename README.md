基于卷积神经网络的手写
汉字识别系统操作说明
本次实验环境是在Ubuntu18.04下，显卡需要Nvidia1080ti，python版本是python3.6.8 ,tensorflow版本是1.12.0。
A．	数据集处理
1.数据集来自于中科院自动化研究所，下载命令:
wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip
下载到程序主目录，解压后发现是一些gnt文件（注意在HWDB1.1trn_gnt.zip解压后是alz文件，需要再次解压，ubuntu上有alz的解压工具，需要输入命令
unalz -d HWDB1.1trn_gnt/ HWDB1.1trn_gnt.alz进行解压)，然后运行python gnt_to_phg.py将所有文件都转化为对应label目录下的所有png的图片，将数据集保存到data目录下。
2.或者从百度云下载下来已经转化完的文件
数据集链接链接:
https://pan.baidu.com/s/194VOdBzCjrNhWxoBncnPJQ提取码: 4bg6,需要数据集解压到data目录下。
B．配置环境
一、	安装python库
1.	安装opencv
pip3 install opencv-python
pip3 install opencv-contrib-python
pip3 install matplotlib 
pip3 install numpy
pip3 install shutil 
sudo apt-get install python-imaging
2.	安装pyqt5   
1.	安装PyQt5包
pip3 install pyqt5 -i https://pypi.douban.com/simple
2.	安装Ubuntu下所需要的依赖
sudo apt install pyqt5*
二、安装tensorflow-gpu版本
1.	安装显卡驱动
可以按照这个教程安装显卡驱动，https://www.cnblogs.com/liangzp/p/9105294.html
2. 安装CUDA，CUDA是英伟达专门为GPU计算推出的计算平台
安装CUDA Toolkit 9.0
下载地址：
https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704&target_type=runfilelocal
选择17.10的版本，按照下图选择，安装Base Installer。
 
 
通过终端进入下载目录，输入下列命令进行安装。
sudo chmod +x cuda_9.0.176_384.81_linux.run
./cuda_9.0.176_384.81_linux.run –override
installing with an unsupported configuration？时选择yes
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81？为了稳定最好选择no
3. 安装CUDNN 7.0，cuDNN是英伟达为CUDA加速运算推出的加速库，用于在GPU上实现高性能现代并行计算；
安装CUDNN 7.0
下载地址：
https://developer.nvidia.com/rdp/cudnn-archive
选择下图这个版本
 
通过终端进入下载目录，输入下列命令进行安装。
# 解压
tar -zxvf cudnn-9.0-linux-x64-v7.tgz 
# 复制相应文件
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/
sudo cp  cuda/include/cudnn.h /usr/local/cuda-9.0/include/
# 所有用户可读
sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
4.安装libcupti
sudo apt-get install libcupti-dev
5. 配置
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}
6. 安装tensorflow-gpu
Pip3 install --upgrade tensorflow-gpu
7. 测试
 
按照上图输入，如果输出1.12.0说明安装成功
三、也可以安装CPU版的tensorflow
	sudo pip3 install tensorflow
C．运行程序(此步骤可以在CPU下运行，需要将开头代码os.environ["CUDA_VISIBLE_DEVICES"] = "-1"注释取消)
1.在程序主目录下，在终端下输入下列命令执行程序，需要等待一小段时间。
python main.py
2.点击打开文件从picture里选择图片进行汉字图片切割
 
3.点击切割字符，切割汉字图片
 
4.点击汉字预测按钮，进行汉字识别。
 
5.双击切割的汉字，可以显示预测汉字的概率，如点击“白”字，下图显示概率。
 
D．训练模型（建议使用GPU训练，需要将开头代码os.environ["CUDA_VISIBLE_DEVICES"] = "-1"注释）
1.在跟目录下，在终端下输入
（1）训练
python  chinese_character_recognition.py  --mode=train 
（2）接着以前的训练
python  chinese_character_recognition.py  --mode=train  --restore=True
（3）测试测试集的准确率
python  chinese_character_recognition.py  --mode=validation
（4）识别图片
需要将待识别的单字图片复制到tmp文件夹下，输入下列命令
python  chinese_character_recognition.py  --mode=inference
（5）chinese_labels文件是对应汉字的Unicode编码，data目录中放的是数据集，checkpoint里面放的是训练的模型，log里面放的是用tensorboard查看的日志文件,输入tensorboard --logdir=./log/train/查看训练集的图像，输入tensorboard --logdir=./log/val/查看测试集的图像
2.在Program文件夹下有三个卷积神经网络，需要将目录下的data文件夹分别复制到fc、fc_gap、gap这三个文件夹中分别是fc目录下包含两个全连接层的Model A，fc_gap目录下的用全局平均池化层替换第一层全连接层的Model B，gap目录下的把全连接层替换为全局平均池化层的Model C，源程序代码是chinese_character_recognition_bn.py，可以按照上面的步骤操作，将chinese_character_recognition.py替换为chinese_character_recognition_bn.py。

