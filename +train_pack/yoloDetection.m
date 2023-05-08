% 目标检测(预测)类
classdef yoloDetection
    properties
        img_paths
        cfg_file
        weight_file
        names_path % 目标类别名称
        
        loadNet
    end
    
    methods
        function obj = yoloDetection(varargin)
            import train_pack.loadNet_matlab
            
            if nargin~=3
                return;
            end
            
            obj.img_paths = varargin{1};
            obj.cfg_file = varargin{2};
            obj.weight_file = varargin{3};
            
            obj.loadNet = loadNet_matlab(obj.cfg_file,obj.weight_file);
        end
        
        
        % yolo层的forward结果
        function detection(this, conf_thresh, nms_thresh, score_thresh)
            tic
            % 添加目标类别名称文件
            output_idxs = this.loadNet.output_idxs;
            myNet = this.loadNet.myNet;
            for k=1:length(output_idxs)
                idx = output_idxs(k);
                layer = myNet.Layers(idx);
                layer.names_path = this.names_path;
                myNet = replaceLayer(myNet,layer.Name,layer);
            end
            
            % 转换成dlnetwork
            dlnet = dlnetwork(myNet);
            fprintf('load success! elapsed time: %.3fs\n\n',toc); % 加载网络与网络权重消耗的时间
            
            this.detect_with_net(dlnet, output_idxs, this.img_paths, conf_thresh, nms_thresh, score_thresh); % 多幅图像的目标检测结果
            
        end
        
    end
    
    
    methods(Static)
        % 目标检测与图绘制
        function detect_with_net(dlnet, output_idxs, img_paths, conf_thresh, nms_thresh, score_thresh)
            import train_pack.yoloDetection.detect_bboxes
            import train_pack.yoloDetection.object_in_image
            
            if isempty(output_idxs)
                output_idxs = [];
                for k=1:length(dlnet.Layers)
                    layer = dlnet.Layers(k);
                    if contains(layer.Name,'yolo')
                        output_idxs = [output_idxs, k];
                    end
                end
            end
            
            tic
            bbox_cells = detect_bboxes(dlnet, output_idxs, img_paths, conf_thresh, nms_thresh, score_thresh);
            % 绘制图像
            i=0; % 统计无目标的图像个数
            for k=1:length(bbox_cells)
                if ~isempty(bbox_cells{k})
                    figure(k)
                    img = object_in_image(img_paths{k},bbox_cells{k}{1},bbox_cells{k}{2});
                    imshow(img,[]);
                else
                    i = i + 1;
                end
            end
            fprintf('has object: %d, no object: %d, object detection success! elapsed time: %.3fs\n',length(bbox_cells) - i, i,toc); 
        end
        
        
        % 通过mat文件的网络检测多幅图像 bbox_cells:{bbox,labelName_score,[conf,cls_conf,cls_idx,score]}
        function bbox_cells = detect_bboxes(dlnet, output_idxs, img_paths, conf_thresh, nms_thresh, score_thresh)
            % 读取图像集
            img_mats = [];
            input_size = dlnet.Layers(1).InputSize;
            for k=1:length(img_paths)
                img = single(imread(img_paths{k}))/255; % 归一化
                % 灰度与彩色图像的转换
                if size(img,3)==1 && size(img,3)<input_size(3) % 灰度图转多通道图
                    img = repmat(img,1,1,input_size(3));
                elseif size(img,3)==3 && input_size(3)==1 % 彩色图转灰度图
                    img = im2gray(img);
                end
                img = imresize(img,input_size(1:2),'triangle');
                img_mats(:,:,:,k) = img;
            end
            
            % 图像集的目标预测
            dlX = dlarray(img_mats,'SSCB');
            if canUseGPU % 是否使用GPU
                dlX = gpuArray(dlX);
            end
            outputs = cell(length(dlnet.OutputNames),1);
            [outputs{:}] = predict(dlnet,dlX,'Outputs',dlnet.OutputNames);
            for k=1:length(outputs)
                if canUseGPU % GPU转CPU，dlarray转普通矩阵
                    outputs{k} = extractdata(gather(outputs{k}));
                else
                    outputs{k} = extractdata(outputs{k}); %dlarray转普通矩阵
                end
            end            
            
            yolo_outputs = outputs;
            % 查找yolo层并做数据变换处理(predict_process),然后concat在一起供predict_box调用
            process_outputs = []; 
            for k=1:length(output_idxs)
                idx = output_idxs(k);
                yolo_layer = dlnet.Layers(idx);
                yolo_layer = yolo_layer.predict_process(yolo_outputs{k});
                process_outputs = cat(1,process_outputs,yolo_layer.output);
            end
            
            % 通过最后一层yolo预测批图像的目标并绘制图像
            yolo_layer.output = process_outputs;
            yolo_layer.conf_thresh = conf_thresh;
            yolo_layer.nms_thresh = nms_thresh;
            yolo_layer.score_thresh = score_thresh;
            
            bbox_cells = yolo_layer.predict_box(img_paths); % bbox:[cx,cy,w,h]
        end
        
        
        % 绘制检测的图像
        function out_img = object_in_image(img_path,bboxes,label_names)
            if ~isempty(bboxes)
                % [cx,cy,w,h]转换成[x1,y1,w,h]
                bboxes(:,1) = bboxes(:,1) - bboxes(:,3)/2;
                bboxes(:,2) = bboxes(:,2) - bboxes(:,4)/2;
                
                img = imread(img_path);
                colors = rand(size(bboxes,1),3)*255;
                out_img = insertObjectAnnotation(img,'rectangle',bboxes,label_names,...
                                                    'Color',colors,'LineWidth',2);
            end
        end
    end
end