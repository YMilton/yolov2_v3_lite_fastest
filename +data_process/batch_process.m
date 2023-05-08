
classdef batch_process
    methods(Static)
        % 计算网络的损失
        function [lbox,lconf,lcls,L,new_mn] = batch_loss(img_paths,mynet,input_size)
            t1 = clock;
            import data_process.batch_process
            import data_process.loss
            import data_process.data_util
            
            [batch_imgs,batch_gt] = batch_process.get_batch_imgs(img_paths, input_size, 1);
            
            % (1) batch个图像的前向传播，预测值获取
            mn = my_network; mn.mynet = mynet; mn.input = batch_imgs;
            mn = mn.forward;
            pred_yolos = mn.yolos; % pred_yolos为元胞数组，yolov2只有一个yolo对象，yolov3不止一个yolo对象（2或3）
            
            yolo_deltas = {}; % 多个yolo输出
            for n=1:length(pred_yolos)
                gt = batch_gt;
                yolo = pred_yolos{n}; % box输出范围在[0,1]之间
                % 在feature grid上的预测框
                yolo_out = yolo.output;
                [map_h,map_w,~,batch] = size(yolo.input);
                yolo_out(:,[1,3],:) = yolo_out(:,[1,3],:)*map_w;
                yolo_out(:,[2,4],:) = yolo_out(:,[2,4],:)*map_h;
                preds = reshape(yolo_out,map_h,map_w,size(yolo.anchors,1),[],batch);
                preds = permute(preds,[5,1,2,3,4]); % size:[batch_size,grid_size,grid_size,anchor_size,5+classes]

                % (2) ground_truth值的处理
                gt(:,[3,5]) = gt(:,[3,5])*map_w; % gt投影到feature map上
                gt(:,[4,6]) = gt(:,[4,6])*map_h;
                anchors = yolo.anchors;
                
                for k=1:batch
                    batch_idxs = gt(:,1)==k;
                    whs = gt(batch_idxs,5:6); % 一幅图像的人工标注框wh
                    [best_iou,best_idx] = data_util.iou_anchors(whs,anchors);
                    gt(batch_idxs,7) = best_idx;
                    gt(batch_idxs,8) = best_iou;
                end   
                % batch中损失与梯度计算
                [lbox,lconf,lcls,L,delta] = loss.batch_loss(preds,gt);
                yolo_deltas{n} = delta;
                fprintf('YOLO %d: [IoU loss: %.3f, Object loss: %.3f, Class loss: %.3f, Loss: %.3f, Batch loss: %f]\n',...
                    n,lbox,lconf,lcls,L,L*batch);
            end
            t2 = clock;
            fprintf('Elapsed time: %.3fs.\n',etime(t2,t1));
            
            % (3)反向传播
            t1 = clock;
            mn.delta = yolo_deltas; % 传入梯度元胞数组
            mn = mn.backward;
            new_mn = mn;
            t2 = clock;
            fprintf('Backward elapsed time: %.3fs.\n',etime(t2,t1));
        end
        
        
        % 获取一个batch的图像矩阵与truth_bbox, size(batch_imgs) = [rows,cols,filters,batch]
        function [batch_imgs,batch_gt] = get_batch_imgs(img_paths, input_size, is_norm)
            import data_process.data_util
            batch_gt = []; batch_imgs = [];
            for k=1:length(img_paths)
                img_path = img_paths{k};
                I = imread(img_path); % 读取原始图像
                % 读取的图像通道与输入的通道保持一致
                if size(I,3)==1 && size(I,3)<input_size(3) %灰度图转>3通道输入
                    I = repmat(I,1,1,input_size(3));
                elseif size(I,3)==3 && input_size(3)==1 %彩色图转灰度图
                    I = im2gray(I);
                end
                
                % 随机旋转图像，bbox
                tform = randomAffine2d('Rotation',[-30,30]);
                [I_, rout] = imwarp(I,tform);
                
                truth_bbox = data_util.get_bbox(img_path); % 获取每幅图像的人工标注框
                box = ceil(truth_bbox(:,2:5).*[size(I,[2,1]), size(I,[2,1])]);
                box_ = bboxwarp(box,tform, rout); % 旋转的bbox
                if ~isempty(box_)
                    box_ = box_./[size(I_,[2,1]), size(I_,[2,1])]; % 归一化
                    truth_bbox(:,2:5) = box_; % 重新赋值到truth_bbox
                else
                    I_ = I; % bbox变换为空，则图像不变换
                end
                
                if is_norm % 图像是否归一化
                    img = single(imresize(I_,input_size(1:2)))/255; % 归一化，resize
                else
                    img = single(imresize(I_,input_size(1:2)));
                end
                batch_imgs(:,:,:,k) = img;
                
                truth_bbox = [ones(size(truth_bbox,1),1)*k,truth_bbox]; % 添加batch_idx
                batch_gt=[batch_gt;truth_bbox];
            end
        end
        
        
        % 获取图像对应的真实标注信息 [img_idx,cls_idx,box]
        function gt = get_gt_info(img_paths)
            import data_process.data_util
            
            gt = [];
            for k=1:length(img_paths)
                truth_bbox = data_util.get_bbox(img_paths{k}); % 获取每幅图像的人工标注框
                truth_bbox(:,1) = truth_bbox(:,1) + 1; % matlab从1开始
                
                img_path = img_paths{k};
                [h,w,~] = size(imread(img_path));
                truth_bbox(:,2:end) = truth_bbox(:,2:end).*[w,h,w,h]; % box转换到原始图像上
                
                % 添加图像索引
                truth_bbox = [ones(size(truth_bbox,1),1)*k,truth_bbox];
                gt = [gt;truth_bbox];
            end
        end
        
        
    end
end