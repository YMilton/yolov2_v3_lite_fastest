clear,clc
close all

% model evaluate
data_file = 'dataSets/boat_airplane_tank/target2.data';

mat_file = 'save_model/lite2_epoch120.mat';

import train_pack.modelEvaluate
me = modelEvaluate(data_file,mat_file);
me.conf_thresh = 0.5; me.nms_thresh = 0.75; me.score_thresh = 0.5;
[mAP,AP,F1,recall,precision] = me.evaluate;
disp("=============================================================");
disp("| AP="+AP+", F1="+F1+", Recall="+recall+", Precision="+precision+" |");
disp("| mAP="+mAP+" |");
disp("=============================================================");