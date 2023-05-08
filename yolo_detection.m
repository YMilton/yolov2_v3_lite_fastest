clear,clc

tic

% img_path = '../images/zidane.jpg';
img_path = '../images/dog.jpg';
% img_path = '../images/field.jpg';
% img_path = '../images/horses.jpg';

% img_path = 'dataSets/target/images/valid/y3500000_320_1.jpg';

%================================================ yolov2 ================================================%
% mn = my_network(img_path,'onnx_net.mat');
% mn = my_network(img_path,'cfg/yolov2/yolov2.cfg','cfg/yolov2/yolov2.weights');
% mn = my_network(img_path,'cfg/yolov2/yolov2-tiny.cfg','cfg/yolov2/yolov2-tiny.weights');

%================================================ yolov3 ================================================%
mn = my_network(img_path,'cfg/yolov3/yolov3.cfg','cfg/yolov3/yolov3.weights');
% mn = my_network(img_path,'cfg/yolov3/yolov3-tiny.cfg','cfg/yolov3/yolov3-tiny.weights');

%============================================= yolo-fastest =============================================%
% mn = my_network(img_path,'cfg/fastest/yolo-fastest.cfg','cfg/fastest/yolo-fastest.weights');
% mn = my_network(img_path,'cfg/fastest/yolo-fastest-xl.cfg','cfg/fastest/yolo-fastest-xl.weights');
% mn = my_network(img_path,'cfg/fastest/yolo-fastest-1.1.cfg','cfg/fastest/yolo-fastest-1.1.weights');
% mn = my_network(img_path,'cfg/fastest/yolo-fastest-1.1-xl.cfg','cfg/fastest/yolo-fastest-1.1-xl.weights');

%============================================== yolov2-lite =============================================%
% mn = my_network(img_path,'cfg/lite/tiny-yolov2-trial3-noBatch.cfg','cfg/lite/tiny-yolov2-trial3-noBatch.weights');
% mn = my_network(img_path,'cfg/lite/trial6.cfg','cfg/lite/trial6_653550.weights');
% mn = my_network(img_path,'dataSets/target/lite.cfg','dataSets/target/lite.weights');

mn = mn.forward;

preds = []; % 保存所有yolo的预测值，整合在一起
for n=1:length(mn.yolos)
    yolo = mn.yolos{n};
    preds=[preds;yolo.output];
end

yolo.output = preds;
yolo.conf_thresh = 0.5;
yolo.nms_thresh = 0.75;
yolo.score_thresh = 0.5; % yolov2没有该参数
% predict_box不使用anchor
[bbox,score,label_names] = yolo.predict_box(img_path);

if ~isempty(bbox)
    bbox(:,3) = bbox(:,3) - bbox(:,1);
    bbox(:,4) = bbox(:,4) - bbox(:,2);
    
    imshow(imread(mn.img_path),[])
    hold on
    for k=1:size(bbox,1)
        c = rand(1,3);
        rectangle('Position',bbox(k,:),'EdgeColor',c,'LineWidth',1.5);
        str_label = label_names{k};
        disp(str_label)
        text(bbox(k,1),bbox(k,2)-8,str_label, 'Color',c,'LineWidth',1.5);
    end
end

toc