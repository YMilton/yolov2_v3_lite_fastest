clear,clc
close all

import train_pack.yoloTrain

% training vehicle data
% yt = yoloTrain('dataSets/vehicle/trial6.cfg','dataSets/vehicle/vehicle.data');
% yt.LR = 0.001; yt.num_epochs = 80; yt.warmup_period = 1000;

% training target data
yt = yoloTrain('dataSets/boat_airplane_tank/lite2.cfg','dataSets/boat_airplane_tank/target2.data');
yt.LR = 0.001; yt.num_epochs = 120; yt.warmup_period = 1000; % set the learning arguments
yt = yt.replace_anchors; % replace the network anchors using estimate acnhors
yt.train;
