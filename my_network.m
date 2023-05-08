
classdef my_network   
    properties(Access=public)        
        name='my_network'
        img_path % 输入图像路径
        net_info % 网络信息
        input_size
        
        input
        yolos
        mynet % 数据类型为cell
        
        delta % 梯度
    end
    
% ================================================================================================================ % 
    methods(Static)
                
        % sigmoid函数
        function out = sigmoid(x)
            out = 1.0./(1.0+exp(-x));
        end
        
        %softmax函数
        function out = softmax(x)
            x = x - max(x,[],4); % 减去最后一个维度的最大值
            out = exp(x)./sum(exp(x),4);
        end
        
        % clamp函数
        function out = clamp(x,min_val,max_val)
            x(x<min_val) = min_val;
            x(x>max_val) = max_val;
            out = x;
        end
        
        % cxcywh->xyxy
        function box_xyxy = xywh2xyxy(box_xywh)
            box_xyxy(:,1) = box_xywh(:,1) - box_xywh(:,3)/2;
            box_xyxy(:,2) = box_xywh(:,2) - box_xywh(:,4)/2;
            box_xyxy(:,3) = box_xywh(:,1) + box_xywh(:,3)/2;
            box_xyxy(:,4) = box_xywh(:,2) + box_xywh(:,4)/2;
        end
        
        % 交并比计算 box:[cx,cy,w,h]
        function iou=bbox_iou(box1,box2)
            % xywh->xyxy
            b1 = my_network.xywh2xyxy(box1);
            b2 = my_network.xywh2xyxy(box2);
            
            % 公共区域左上坐标与右下坐标计算
            x1 = max(b1(:,1),b2(:,1)); y1 = max(b1(:,2),b2(:,2));
            x2 = min(b1(:,3),b2(:,3)); y2 = min(b1(:,4),b2(:,4));
            
            inter_rect = (x2-x1+1).*(y2-y1+1);
            b1_rect = (b1(:,3) - b1(:,1)+1).*(b1(:,4) - b1(:,2)+1);
            b2_rect = (b2(:,3) - b2(:,1)+1).*(b2(:,4) - b2(:,2)+1);
            
            iou = inter_rect./(b1_rect + b2_rect - inter_rect + 1e-15);
        end
        
        % pred是二维的，如四维[13,13,3,85]=>[507,85],
        % conf_thres:置信度阈值，nms_thres:非极大值阈值,score_thresh: 分数阈值限制
        function out=nms_yolov3(pred,conf_thres,nms_thres,score_thresh)
                        
            pred = pred(pred(:,5)>conf_thres,:); % 筛选conf大于conf_thres的行
            if ~isempty(pred)
                [vals,idxs]=max(pred(:,6:end),[],2);  % class_conf的最大值与索引
                score = pred(:,5).*vals; % conf阈值筛选的概率与最大class_conf相乘
                detections = cat(2,pred(:,1:5),vals,idxs,score);
                detections = sortrows(detections,8,'descend'); % 根据score降序排序
                % 剔除score阈值小于score_thresh的行
                detections = detections(detections(:,8)>score_thresh,:);
                
                out=[];
                while ~isempty(detections)
                    % 筛选出类别相同，并iou>nms_thres的检测框
                    iou_filter = my_network.bbox_iou(detections(1,1:4),detections(:,1:4)) > nms_thres;
                    class_idx_filter = detections(:,7)==detections(1,7);
                    filter = iou_filter & class_idx_filter;
                    
                    % 通过conf对box做加权平均
                    weights = detections(filter,5);
                    detections(1,1:4) = sum(weights.*detections(filter,1:4),1)/sum(weights);
                    out=[out;detections(1,:)];
                    % 删除非极大值抑制处理的行
                    detections = detections(~filter,:);
                end
            else
                out=[];%没有检测目标
            end
        end
        
        % yolov2版本的nms
        function out=nms_yolov2(pred,conf_thres,nms_thres)
            
            [vals,idxs]=max(pred(:,6:end),[],2);  % class_conf的最大值与索引
            score = pred(:,5).*vals; % conf概率与最大class_conf相乘
            
            conf_thres_filter = score>conf_thres;
            pred = pred(conf_thres_filter,:);
            vals = vals(conf_thres_filter);
            idxs = idxs(conf_thres_filter); % 根据conf_thres筛选类别标号
            score = score(conf_thres_filter);
            
            if ~isempty(pred)
                detections = cat(2,pred(:,1:5),vals, idxs, score);
                detections = sortrows(detections,8,'descend'); % 根据score降序排序
                
                out=[];
                while ~isempty(detections)
                    iou_filter = my_network.bbox_iou(detections(1,1:4),detections(:,1:4)) > nms_thres;
                    out = [out;detections(1,:)];
                    
                    detections = detections(~iou_filter,:);
                end
            else
                out=[];%没有检测目标
            end
        end
               
    end
    
% ========================================================================================================== %     
    methods(Access=public)
        % 赋值图像路径，神经网络层结构
        % my_network(img_path, matlab_net_layers) or my_network(img_path, cfg_file, weight_file)
        function obj = my_network(varargin)
            if nargin>3 || nargin<0 % 不能超过3个参数
                disp('Please input right parameters(my_network)!');
                return;
            end
            
            if nargin==0
                return;
            end
            
            if nargin<=2
                % 通过onnx加载网络
                load(varargin{2}, 'lgraph');
                obj.mynet = load_net.create_net_matlab(lgraph.Layers);
                obj.input_size = lgraph.Layers(1).InputSize(1:2);
            else
                myload = load_net(varargin{2},varargin{3});
                obj.mynet = myload.mynet;
                obj.input_size = [myload.net_struct.net.width, myload.net_struct.net.height];
                obj.net_info = myload.net_struct.net;
            end
            
            % 图像路径赋值
            obj.img_path = varargin{1};
            % 图像预处理,简单归一化
            %obj.input = double(preprocess_image(obj))/255;
            obj.input = double(imresize(imread(obj.img_path),obj.input_size))/255;
        end
        
        
        % 前向传播
        function obj = forward(this)
            % 添加网络层的类
            import layers.*
            
            if isempty(this.input) || isempty(this.mynet)
                disp('please give the input value network cell!');
            end
            
            % 存储过程变量(当前层输出作为下一层输入)
            tmp = this.input;
            yolo_num = 1;
            for k=1:length(this.mynet)
                layer = this.mynet{k};
%                 if k==5
%                     fprintf('%s %d\n',layer.name, k);
%                 end
                layer.input = tmp; % 赋值该层的input值
                
                if strcmp(layer.name,'conv') || strcmp(layer.name,'maxpool') || strcmp(layer.name,'dropout')...
                        || strcmp(layer.name,'reorg') || strcmp(layer.name,'upsample')
                    layer = layer.forward;
                
                elseif strcmp(layer.name,'shortcut') || strcmp(layer.name,'route')
                    layer = layer.forward(this.mynet,k); % 传入网络与当前层序号
                    
                elseif strcmp(layer.name,'yolo')
                    layer = layer.forward;
                    this.yolos{end+1,1} = layer;
                    
%                     disp("==========================================================")
%                     fprintf('THE %d YOLO FORWARD SUCCESS!\n',yolo_num);
%                     disp("==========================================================")
                    yolo_num = yolo_num + 1; 
                    
                else
                    fprintf('There is no %s net layer\n',layer.name);
                end
                tmp = layer.output; % 当前层的输出作为下一层的输入
                this.mynet{k} = layer; % 赋值mynet当前层layer
            end
            obj = this;
            % fprintf('forward success!\n\n');
        end
        
        
        % 网络的反向传播
        function obj = backward(this)
            % 添加网络层的类
            import layers.*
            
            % 找出yolo层在网络中的位置
            yolo_idxs = [];
            for i=1:length(this.mynet)
                layer = this.mynet{i};
                if strcmp(layer.name,'yolo')
                    yolo_idxs = [yolo_idxs,i];
                end
            end
            
            assert(length(yolo_idxs),length(this.delta)); % 保证长度相同
            
            for n=1:length(this.delta)
                delta_tmp = this.delta{n}; % 梯度
                yolo_idx = yolo_idxs(n);
                for k=yolo_idx:-1:1
                    layer = this.mynet{k};
                    layer.delta = delta_tmp; %赋值当前层的delta
                    % tic
                    layer = layer.backward; % 当前层反向传播
                    % fprintf(strcat(layer.name,num2str(k),' elapsed time %.3fs\n'),toc);
                    this.mynet{k} = layer; % 更新当前层
                    delta_tmp = layer.update_delta; % 更新的delta供上一层使用
                end
            end
            
            obj = this;
        end
        
        
        % 处理输入图像的尺寸,resize输入的图像
        function copy_img = preprocess_image(obj)
            img = imread(obj.img_path);
            [rows,cols,~] = size(img);
            copy_img = uint8(zeros(obj.input_size(2),obj.input_size(1),3));
            
            if rows>cols % 高大于宽
                scale = obj.input_size(2)/rows; % 缩放尺寸
                img_resize = imresize(img,scale,'bilinear');
                copy_x = floor((obj.input_size(1) - size(img_resize,2))/2);
                copy_img(:,copy_x:copy_x+size(img_resize,2)-1,:) = img_resize;
            else
                scale = obj.input_size(1)/cols;
                img_resize = imresize(img,scale,'bilinear');
                copy_y = floor((obj.input_size(2) - size(img_resize,1))/2);
                copy_img(copy_y:copy_y+size(img_resize,1)-1,:,:) = img_resize;
            end
        end
           
    end
    
end



