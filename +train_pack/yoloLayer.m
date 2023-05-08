classdef yoloLayer < nnet.layer.Layer
    properties(Access=public)
        version
        input_size % 网络传入的图像大小
        anchors % 锚点
        classes
        names_path % 目标检测类
        
        conf_thresh = 0.45 % 阈值
        nms_thresh = 0.75
        score_thresh = 0.5
        
        batch_size
        map_size % 特征宽高
        output % 变换后的数据
    end
    
    methods
        % 构造函数
        function obj = yoloLayer(name, version,anchors,classes)
            obj.Type = 'yoloLayer';
            obj.Description = 'yolo layer arguments';
            obj.Name = name;
            obj.version = version;
            obj.anchors = anchors;
            obj.classes = classes;
        end
        
        function Z=predict(~,X)
            X = dlarray(X);
            Z = X;
        end
        
        
        
        % 对数据做sigmoid处理、乘以anchor、变换处理 (比较花费时间)
        % yolov3 input: [rows,cols,(5+classes)*3,batch_size]
        function obj = predict_process(this,input)
            prediction = input;
            [map_h,map_w,~,this.batch_size] = size(input);
            this.map_size = [map_w,map_h]; % 特征图宽高赋值
            % size: [13,13,5,25], matlab是按列reshape, python是按行reshape
            prediction = reshape(prediction,size(prediction,1), size(prediction,2),...
                                    size(prediction,3)/size(this.anchors,1), size(this.anchors,1),size(prediction,4));
            % 维度交换[rows,cols,anchor_num,batch_size,5+classes]
            prediction = permute(prediction,[1,2,4,5,3]); % 把5+classes维度交换到最后一维
            [map_h,map_w,~,~,~] = size(prediction); % feature map的大小
            
            % feature map的网格坐标生成
            [grid_x,grid_y] = meshgrid(1:map_w,1:map_h);
            grid_x = grid_x - 1; grid_y = grid_y - 1; % matlab是从1开始，计算是需要从0开始
            % xy的计算。加上偏移量 bx=sigmoid(tx)+cx, by=sigmoid(ty)+cy
            prediction(:,:,:,:,1) = (my_network.sigmoid(prediction(:,:,:,:,1)) + grid_x)/map_w;
            prediction(:,:,:,:,2) = (my_network.sigmoid(prediction(:,:,:,:,2)) + grid_y)/map_h;
            % 如果锚点表示的图像像素，则重新计算anchors到feature map上
            if max(max(this.anchors))>20
                this.anchors(:,1) = this.anchors(:,1)/this.input_size(1)*map_w;
                this.anchors(:,2) = this.anchors(:,2)/this.input_size(2)*map_h;
            end
            % wh的计算。 bw=pw*exp(tw), bh=ph*exp(th)
            % this.anchors是相对于feature map上的锚点大小
            anchor_w = reshape(this.anchors(:,1),1,1,size(this.anchors,1),1);
            anchor_h = reshape(this.anchors(:,2),1,1,size(this.anchors,1),1);
            prediction(:,:,:,:,3) = exp(prediction(:,:,:,:,3)).*anchor_w/map_w;
            prediction(:,:,:,:,4) = exp(prediction(:,:,:,:,4)).*anchor_h/map_h;
            % conf做sigmoid计算
            prediction(:,:,:,:,5) = my_network.sigmoid(prediction(:,:,:,:,5));
            
            prediction(:,:,:,:,6:end) = my_network.sigmoid(prediction(:,:,:,:,6:end));
            
            % class_conf做softmax或sigmoid计算
%             if strcmp(this.version,'V2')
%                 prediction(:,:,:,:,6:end) = my_network.softmax(prediction(:,:,:,:,6:end));
%             else
%                 prediction(:,:,:,:,6:end) = my_network.sigmoid(prediction(:,:,:,:,6:end));
%             end
            
            % 转换矩阵5维矩阵为3维矩阵            
            this.output = reshape(prediction,[],size(prediction,4),size(prediction,5));
            this.output = permute(this.output,[1,3,2]); % [rows*cols*anchor_num,5+classes,batch_size]
            
            obj = this;
        end
        
        
        % 针对yolo层的输出数据计算检测框、标签、置信度
        % img_path：原始图像路径
        function bbox_cells = predict_box(this, img_paths)
            % output: 3维矩阵[rows*cols*anchor_num,5+classes,batch_size]
            import data_process.data_util.get_file_context;
            
            bbox_cells = cell(size(this.output,3),1);
            for b=1:size(this.output,3)
                % 非极大值抑制处理
                prediction = this.output(:,:,b);                
                if strcmp(this.version,'V2')
                    out = my_network.nms_yolov2(prediction,this.conf_thresh,this.nms_thresh);
                else
                    out = my_network.nms_yolov3(prediction,this.conf_thresh,this.nms_thresh,this.score_thresh);
                end
                
                % out = my_network.nms_yolov3(prediction,this.conf_thresh,this.nms_thresh,this.score_thresh);
                
                if isempty(out)
                    fprintf('image: %s no object detected!\n',img_paths{b});
                    bbox_cells{b} = [];
                    continue;
                end

                % 映射到原始图像上
                [h,w,~] = size(imread(img_paths{b}));
                out(:,[1,3]) = out(:,[1,3])*w; 
                out(:,[2,4]) = out(:,[2,4])*h;
                
                % out:[cx,cy,w,h,conf,cls_conf,cls_conf_max_idx,score]
                bbox = out(:,1:4);
                score = out(:,8); % conf*cls_conf
                label = out(:,7); % 标签标号

                % 检测目标的名称
                class_names = get_file_context(this.names_path); % 标签名称
                
                % 标签名与置信度拼接
                label_names = {};
                for k=1:size(bbox,1)
                    str_label = strcat(class_names{label(k,:)},': ',num2str(score(k,:)));
                    label_names{k,1} = str_label;
                end
                
                bbox_cells{b} = {bbox;label_names;out(:,5:8)}; % {bbox,labelName_score,[conf,cls_conf,cls_idx,score]}
            end
        end
        
    end
    
end