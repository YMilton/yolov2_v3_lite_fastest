clear,clc

%% matlab的anchor评估函数
data = load('vehicleTrainingData.mat');
trainingData = data.vehicleTrainingData;
blds = boxLabelDatastore(trainingData(1:208,2:end));
anchorBoxes = sort(estimateAnchorBoxes(blds,5))

%% 直接调用kmeans
bboxes = cell2mat(blds.LabelData(:,1));
bboxes_wh = bboxes(:,3:4);
[idx,C]=kmeans(bboxes_wh,5);
C = floor(fliplr(C));
C = sort(C)

%% 自定义的anchor评估函数
[centers,meanIoU] = data_process.dataloader.kmeans_anchors(bboxes_wh,5);
centers = fliplr(centers);
centers = sort(centers)


% function anchors = sort_anchors(centers)
%     area = centers(:,1).*centers(:,2);
%     [~,idx] = sort(area,'ascend');
%     anchors = centers(idx,:); % 排序后的锚点框
% end