clear,clc

% net load
net = load_net('cfg/lite/trial6.cfg');
% net = load_net('cfg/yolov3/yolov3-tiny.cfg');
net.mynet = load_net.set_status(net.mynet,'train');
% coco data load
DL = data_process.dataloader('cfg/coco.data',net);

% parfor
num_imgs = length(DL.train_path);
batch = DL.net_info.batch/DL.net_info.subdivisions;
if batch==1
    fprintf('The training batch is %d,the batch must be greater than 1\n',batch);
    return;
end

import data_process.batch_process
mynet = net.mynet;
for n=1:batch:num_imgs
    begin_idx = n;end_idx = n + batch - 1; 
    batch_imgs = DL.train_path(begin_idx:end_idx);
    fprintf('Batch %d\n',end_idx/batch)
    [lbox,lconf,lcls,L,mn] = batch_process.batch_loss(batch_imgs,mynet,DL.img_size);
    mynet = mn.mynet; % 更新mynet
end
save mynet

