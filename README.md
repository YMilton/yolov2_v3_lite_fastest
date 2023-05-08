# yolov2_v3_lite_fastest
yolov2、yolov3、yoloLite、yoloFastest detection and training


### 1. 自定义网络层
<p>(1) data_process 包：训练之前数据的预处理、损失函数的计算(一个 batch)等。<p>
<p>(2) layer 包：神经网络的不同层，yolo、maxpool、conv 包括前向与反向，其他暂时只有前向。<p>  
<p>(3) cfg 目录：网络配置文件*.cfg，训练得到的权重文件*.weights，目标对象名称文件等。<p>  
<p>(4) images：检测的图像。 <p> 
<p>(5) load_net.m：通过 cfg 加载网络。  <p>
<p>(6) my_network.m：串联所有层的网络。 <p> 
<p>(7) yolo_detection.m：目标检测。  <p>
<p>(8) yolo_train.m：网络训练。<p>
<br/>

### 2. Matlab算子的检测与训练
<p>(1) train_pack 包：包括神经网络的各层，加载网络与权重(适配 Matlab 深度学习算子)，检测与训练处理等。<p>  
 <p>&emsp;loadNet_matlab.m: 通过 cfg 文件、weights 文件加载网络(适配 Maltab 算子)。包括两种方式：<p> 
 <p>&emsp;&emsp;1.fun(cfg_file, weight_file)，适用于训练与检测。训练时 weight_file为主干网络预训练权重，检测时 weight_file 为某个网络训练好的权重;<p>
 <p>&emsp;&emsp;2.fun(cfg_file)只适用训练时，没有加载预训练权重，通过自定义权重初始化。<p>
 <p>&emsp;yoloDetection.m: 通过 gpu 检测一个目录下的所有图像。其中包括非极大值抑制等。<p>  
 <p>&emsp;yoloTrain.m: 通过 gpu 做训练。其中包括损失函数计算，反向传播及梯度更新等。<p>
<p>(2) gpu_detection: 使用 GPU 做检测的入口。<p>
<p>(3) gpu_train: 使用 GPU 做训练的入口。<p>
<br/>

### 3. 检测效果
##### 3.1  yolov2-tiny
![image](https://user-images.githubusercontent.com/27056069/236856236-3b11f02c-9a66-4d6c-b27f-997320487244.png)
##### 3.2 yolov2
![image](https://user-images.githubusercontent.com/27056069/236857671-ace72de7-e308-4d28-a72c-4c9adf50e657.png)
##### 3.3 yolov3-tiny
![image](https://user-images.githubusercontent.com/27056069/236858238-10f2c13b-7bb7-4e65-834e-e73a5b962669.png)
##### 3.4 yolov3
![image](https://user-images.githubusercontent.com/27056069/236858469-b777d3c8-1151-4a67-bd0c-134a7178ec26.png)
##### 3.5 yolo-fastest
![image](https://user-images.githubusercontent.com/27056069/236858678-efafecb2-656a-4b60-b684-3320a92b830d.png)
##### 3.6 yolo-lite
![image](https://user-images.githubusercontent.com/27056069/236858898-0e4d3311-11ce-4487-91ea-48ecaf2fccd1.png)
##### 3.7 yolov4-tiny(GPU)
![image](https://user-images.githubusercontent.com/27056069/236859144-d12cc162-3aac-4fbd-81c9-ff882cd425ea.png)
##### 3.8 yolov4(GPU)
![image](https://user-images.githubusercontent.com/27056069/236859259-9c87d381-b38d-4e28-a508-c2990dd04d02.png)

<p>yolo系列网络结构图https://blog.csdn.net/YMilton/article/details/120268832<p>
