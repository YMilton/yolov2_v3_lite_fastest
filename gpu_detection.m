clear,clc
close all

import train_pack.yoloDetection
import data_process.dataloader


% first one：load mat file
% data_loader = dataloader('dataSets/target/target.data');
% img_paths = data_loader.valid_path(1:20);
% data = load('save_model/lite_epoch300_target.mat');
% yoloDetection.detect_with_net(data.model, [], img_paths,0.5,0.75,0.5);


% second one：load cfg file and weight file(darknet)
% img_path = {'E:\DataSets\Private_DataSets\boat_airplane_tank\images\11589.bmp'};
data_loader = dataloader('dataSets/boat_airplane_tank/target2.data');
img_paths = data_loader.valid_path(5:10);
yd = yoloDetection(img_paths,'dataSets/boat_airplane_tank/lite2_replace_anchor.cfg','dataSets/boat_airplane_tank/lite2_repl.weights');
% data_loader.net_info = yd.loadNet.net_info;
yd.names_path = 'dataSets/boat_airplane_tank/target.names';
yd.detection(0.5,0.75,0.5);

% img_path = fullfile({dir('./images/*.jpg').folder},{dir('./images/*.jpg').name});
% yd = yoloDetection(img_path,'cfg/yolov3/yolov3.cfg','cfg/yolov3/yolov3.weights');
% yd.names_path = 'cfg/coco.names';
% yd.detection(0.5,0.75,0.5);