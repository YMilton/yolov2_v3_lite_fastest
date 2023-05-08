% yolo层的定义
classdef yolo_layer
    properties(Access=public)
        name='yolo'
        version % 版本
        status % 状态设置
        input_size % 网络传入的图像大小
        anchors % 锚点
        classes
        
        conf_thresh = 0.45 % 阈值
        nms_thresh = 0.75
        score_thresh = 0.5
        
        input % [grid_x,grid_y,anchor_size,5+classes],如果有batch，则input与output为最后一个维度
        output % 对input sigmoid与乘以anchor后,四维数据转换成二维数组，如yolov2 coco [13,13,5,85]=>[845,85]
        
        delta % 当前层梯度 size=[batch_size,grid_size,grid_size,anchor_size,5+clasess]
        update_delta % 更新的梯度，供上一层使用 size=[grid_size,grid_size,anchor_size*(5+clasess),batch_size]
    end
    
    
    methods(Access=public)
        function obj = forward(this)           
            import layers.yolo_layer.single_forward
            if length(size(this.input))==3
                this.output = single_forward(this.input,this.anchors,this.input_size,this.version); % [grid*grid*anchor_size,5+classes]
            end
            
            if length(size(this.input))==4 % 一个batch处理
                batch = size(this.input,4);
                input_ = this.input; anchors_ = this.anchors;
                input_size_ = this.input_size; version_ = this.version;
                parfor b = 1:batch
                    output_(:,:,b) = single_forward(input_(:,:,:,b),anchors_,input_size_,version_);
                end
                this.output = output_; % [grid*grid*anchor_size,5+classes,batch]
            end
            
            obj = this;
        end
        
        % 反向传播
        function obj=backward(this)
            delta_matrix = this.delta; % size=[batch_size,grid_size,grid_size,anchor_size,5+clasess]
            [b,rows,cols,a,f] = size(delta_matrix);
            input_x = reshape(this.input,rows,cols,a,f,b);
            input_x = permute(input_x,[5,1,2,3,4]);
            cls_vals = input_x(:,:,:,:,6:end);
            if strcmp(this.version,'v2')
                delta_matrix(:,:,:,:,6:end) = delta_matrix(:,:,:,:,6:end).*my_network.softmax(cls_vals).*(1 - my_network.softmax(cls_vals));
            else
                delta_matrix(:,:,:,:,6:end) = delta_matrix(:,:,:,:,6:end).*my_network.sigmoid(cls_vals).*(1 - my_network.sigmoid(cls_vals));
            end
            conf_vals = input_x(:,:,:,:,5);
            delta_matrix(:,:,:,:,5) = delta_matrix(:,:,:,:,5).*my_network.sigmoid(conf_vals).*(1 - my_network.sigmoid(conf_vals));
            
            [~,map_h,map_w,~,~] = size(delta_matrix);
            anchor_w = reshape(this.anchors(:,1),1,1,1,size(this.anchors,1));
            anchor_h = reshape(this.anchors(:,2),1,1,1,size(this.anchors,1));
            delta_matrix(:,:,:,:,3) = delta_matrix(:,:,:,:,3).*exp(input_x(:,:,:,:,3)).*anchor_w/map_w;
            delta_matrix(:,:,:,:,4) = delta_matrix(:,:,:,:,4).*exp(input_x(:,:,:,:,4)).*anchor_h/map_h;
            
            delta_matrix(:,:,:,:,1) = delta_matrix(:,:,:,:,1).*my_network.sigmoid(input_x(:,:,:,:,1)).*(1 - my_network.sigmoid(input_x(:,:,:,1)))/map_w;
            delta_matrix(:,:,:,:,2) = delta_matrix(:,:,:,:,2).*my_network.sigmoid(input_x(:,:,:,:,2)).*(1 - my_network.sigmoid(input_x(:,:,:,2)))/map_h;
            this.update_delta = reshape(delta_matrix,b,rows,cols,[]);% 需要转换成上一层输出的大小
            this.update_delta = permute(this.update_delta,[2,3,4,1]);
            obj = this;
        end
        
        % 针对yolo层的输出数据计算检测框、标签、置信度
        % img_path：原始图像路径
        function [bbox,score,label_names] = predict_box(this, img_path)
            % 非极大值抑制处理
            prediction = this.output; %二维矩阵
            if strcmp(this.version,'v2')
                out = my_network.nms_yolov2(prediction,this.conf_thresh,this.nms_thresh);
            else
                out = my_network.nms_yolov3(prediction,this.conf_thresh,this.nms_thresh,this.score_thresh);
            end
            if isempty(out)
                fprintf('No object detected!\n');
                bbox = [];
                score = [];
                label_names =[];
                return;
            end
            
            % 映射到原始图像上
            [h,w,~] = size(imread(img_path));
            out(:,[1,3]) = out(:,[1,3])*w; 
            out(:,[2,4]) = out(:,[2,4])*h; 
            % 把[cx,cy,w,h]转换成[x1,y1,x2,y2]
            xyxy = my_network.xywh2xyxy(out(:,1:4));
            out(:,[1,3]) = my_network.clamp(xyxy(:,[1,3]),0,w); % 限制像素范围
            out(:,[2,4]) = my_network.clamp(xyxy(:,[2,4]),0,h);
            
            bbox = out(:,1:4);
            score = out(:,8);
            label = out(:,7); % 标签标号
            
            % 检测目标的名称
            if this.classes==80
                class_names = this.get_class_names('cfg/coco.names'); % 标签名称
            elseif this.classes==20
                class_names = this.get_class_names('cfg/voc.names');
            else
                class_names = this.get_class_names('dataSets/target/target.names'); % 自定义目标类别名称文件
            end
            % 标签名与置信度拼接
            label_names = {};
            for k=1:size(bbox,1)
                str_label = strcat(class_names{label(k,:)},': ',num2str(score(k,:)));
                label_names{k,1} = str_label;
            end
        end
    end
    
    
    methods(Static)
        % 针对yolo一个输出做预测检测：yolo-fastest有2个yolo，yolov3有3个yolo;
        % yolo层的前向传播，对卷积后的数据做sigmoid处理、乘以anchor
        % size(output)=[grid*grid*anchor_num,5+classes],二维矩阵
        function output = single_forward(input,anchors,input_size,version)
            prediction = double(input);
            % size: [13,13,5,25], matlab是按列reshape, python是按行reshape
            prediction = reshape(prediction,size(prediction,1), size(prediction,2), size(prediction,3)/size(anchors,1), size(anchors,1));
            % 维度交换
            prediction = permute(prediction,[1,2,4,3]);
            [map_h,map_w,~,~] = size(prediction); % feature map的大小
            
            % feature map的网格坐标生成
            [grid_x,grid_y] = meshgrid(1:map_w,1:map_h);
            grid_x = grid_x - 1; grid_y = grid_y - 1; % matlab是从1开始，计算是需要从0开始
            % xy的计算。加上偏移量 bx=sigmoid(tx)+cx, by=sigmoid(ty)+cy
            prediction(:,:,:,1) = (my_network.sigmoid(prediction(:,:,:,1)) + grid_x)/map_w;
            prediction(:,:,:,2) = (my_network.sigmoid(prediction(:,:,:,2)) + grid_y)/map_h;
            % 如果锚点表示的图像像素，则重新计算anchors到feature map上
            if max(max(anchors))>20
                anchors(:,1) = anchors(:,1)/input_size(1)*map_w;
                anchors(:,2) = anchors(:,2)/input_size(2)*map_h;
            end
            % wh的计算。 bw=pw*exp(tw), bh=ph*exp(th)
            % this.anchors是相对于feature map上的锚点大小
            anchor_w = reshape(anchors(:,1),1,1,size(anchors,1));
            anchor_h = reshape(anchors(:,2),1,1,size(anchors,1));
            prediction(:,:,:,3) = exp(prediction(:,:,:,3)).*anchor_w/map_w;
            prediction(:,:,:,4) = exp(prediction(:,:,:,4)).*anchor_h/map_h;
            % conf做sigmoid计算
            prediction(:,:,:,5) = my_network.sigmoid(prediction(:,:,:,5));
            % class_conf做softmax或sigmoid计算
            if strcmp(version,'v2')
                prediction(:,:,:,6:end) = my_network.softmax(prediction(:,:,:,6:end));
            else
                prediction(:,:,:,6:end) = my_network.sigmoid(prediction(:,:,:,6:end));
            end
            
            % 转换矩阵四维矩阵为二维矩阵            
            output = reshape(prediction,[],size(prediction,4));
        end
        
        % 获取检测目标的名称
        function class_names = get_class_names(file)
            import data_process.data_util.get_file_context;      
            class_names = get_file_context(file);
        end
    end
    
end