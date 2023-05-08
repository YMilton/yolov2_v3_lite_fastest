% 图库数据的加载
classdef dataloader
    properties(Access=public)
        train_path % 训练数据路径
        valid_path % 验证数据路径
        
        names %目标名称文件
        backup
        
        img_size % 图像尺寸
        net_info % cfg文件中[net]的属性
        
        data_path
    end
    
    methods(Access=public)
        % dataloader(data_path) or dataloader(data_path,mynet)    
        function obj = dataloader(varargin)
            if nargin>2 || nargin<=0
                return;
            end
            
            if nargin==2
                obj.net_info = varargin{2}.net_struct.net;
                obj.img_size = [obj.net_info.width, obj.net_info.height];
            end
            obj.data_path = varargin{1};
            [train_txt, valid_txt, names, backup] = load_path(obj);
            obj.names = names;
            obj.backup = backup;
            % 获取txt文件的路径内容
            fid = fopen(train_txt);
            obj.train_path = textscan(fid,'%s','delimiter','\n');
            obj.train_path = obj.train_path{1};
            
            fid = fopen(valid_txt);
            obj.valid_path = textscan(fid,'%s','delimiter','\n');
            obj.valid_path = obj.valid_path{1};
            fclose(fid);
            
            % 选择少量数据训练
%             obj.train_path = obj.train_path(1:500);
%             obj.valid_path = obj.valid_path(1:100);
        end
        
        % 获取coco.data中的路径并读取内容
        function [train_txt,valid_txt, names, backup] = load_path(this)
            fid = fopen(this.data_path);
            context = textscan(fid,'%s','delimiter','\n');
            context = context{1};
            fclose(fid);
            for k=1:length(context)
                if contains(context{k},'train') % 获取训练图片txt文件
                    key_val = split(context{k},'=');
                    train_txt = strtrim(key_val{2}); % 去除左右空格
                end
                
                if contains(context{k},'valid') % 获取验证图片txt文件
                    key_val = split(context{k},'=');
                    valid_txt = strtrim(key_val{2});
                end
                
                if contains(context{k},'names') % 获取数据集包含的目标名称
                    key_val = split(context{k},'=');
                    names = strtrim(key_val{2});
                end
                
                if contains(context{k},'backup') % 训练权重保存的位置
                    key_val = split(context{k},'=');
                    backup = strtrim(key_val{2});
                end
            end
        end
        
        
        % 获取训练集的在当前网络的anchors,anchors是归一化的结果[0,1]
        function [anchors,meanIoU] = estimate_anchors(this, num_anchors, is_show)
            % num_anchors指定聚类的anchor个数
            import data_process.data_util
                   
            bboxes = [];
            for k=1:length(this.train_path)
                file_path = this.train_path{k};
                bbox = data_util.get_bbox(file_path); % 归一化数据
                bboxes = [bboxes;bbox];
            end
            bboxes_whs = bboxes(:,4:5);
            bboxes_whs = bboxes_whs.*[this.net_info.width,this.net_info.height]; % 映射到输入上
            
            [anchors, meanIoU] = this.kmeans_anchors(bboxes_whs,num_anchors);
            
            if is_show % 绘制锚点框
                this.show_anchors([this.net_info.width,this.net_info.height],anchors);
            end
        end
        
        
        % 绘制ground_truth的框
        function draw_gt_bbox(this,num)
            import data_process.data_util
            
            file_path = this.train_path{num};
            bbox = data_util.get_bbox(file_path);
            class_names = data_util.get_file_context(this.names);
            % img = imresize(imread(file_path),[this.net_info.width,this.net_info.height]);
            img = imread(file_path);
            [h,w,~] = size(img);
            bbox(:,2:5) = bbox(:,2:5).*[w,h,w,h]; % 映射到原图
            bbox(:,2:3) = bbox(:,2:3) - bbox(:,4:5)/2;
            
            for k=1:size(bbox,1)
                labels{k} = class_names{bbox(k,1)+1};
            end
            colors = rand(size(bbox,1),3)*255;
            out_img = insertObjectAnnotation(img,'rectangle',bbox(:,2:5),labels, 'Color',colors,'LineWidth',2);
            imshow(out_img);
        end

    end
    
    methods(Static)
        % 绘制anchor图
        function show_anchors(input_size,anchors)
            background = ones(input_size);
            [w,h] = size(background);
            anchor_boxes(:,3:4) = anchors;
            anchor_boxes(:,1:2) = ones(size(anchors,1),2).*[w/2,h/2];
            anchor_boxes(:,1:2) = anchor_boxes(:,1:2) - anchor_boxes(:,3:4)/2;
            out_img = insertObjectAnnotation(background,'rectangle',anchor_boxes,'anchor',...
                                                'Color','r','LineWidth',2);
            imshow(out_img);
        end
        
        
        % ====聚类操作====%
        function [centers,meanIoU] = kmeans_anchors(bboxes_whs,num_anchors)
            import data_process.data_util
            
            % 1. kmeans++选取聚类中心
            rand_idx = randi([1,size(bboxes_whs,1)],1);
            centers = bboxes_whs(rand_idx,:);
            for n = 1:num_anchors-1 
                for k=1:size(bboxes_whs,1)
                    distance = 1 - data_util.iou_bbox_wh(bboxes_whs(k,:),centers);
                    D(k) = min(distance);
                end
                
                [~,max_idx] = max(D,[],2);
                centers = [centers;bboxes_whs(max_idx,:)];
                
%                 total = sum(D)*rand();
%                 for k=1:length(D)
%                     total = total - D(k);
%                     if total>0
%                         continue;
%                     end
%                     centers = [centers;bboxes_whs(k,:)];
%                     break;
%                 end
            end
            
%             % 随机选择num_anchors聚类中心
%             center_idxs = randi([1,size(bboxes_whs,1)],1,num_anchors);
%             centers = bboxes_whs(center_idxs,:); % 随机选取簇中心点

            % 2. 根据选择的中心点聚类
            while true
                cluster_cell = mat2cell(centers,ones(1,num_anchors)); % 存储cluster
                for k=1:size(bboxes_whs,1)
                    ious = data_util.iou_bbox_wh(bboxes_whs(k,:),centers);
                    [max_iou,~] = max(ious,[],1);
                    meanIoUs(k) = max_iou;
                    
                    distance = 1 - ious;
                    [~,min_idx] = min(distance,[],1);
                    
                    cluster_cell{min_idx} = [cluster_cell{min_idx};bboxes_whs(k,:)];
                end
                % 求每个簇的均值
                centers_ = cellfun(@(x) mean(x,1),cluster_cell,'UniformOutput',false);
                centers_  = cell2mat(centers_);
                if isequal(centers,centers_)
                    meanIoU= mean(meanIoUs); % 计算meanIoU
                    break;
                end
                centers = centers_;                
            end
            centers = floor(centers);
        end
        
        
        % 绘制ground_truth的框 (static)
        function draw_bbox(img,bboxes,class_names)
            [h,w,~] = size(img);
            bboxes(:,2:5) = bboxes(:,2:5).*[w,h,w,h]; % 映射到原图
            bboxes(:,2:3) = bboxes(:,2:3) - bboxes(:,4:5)/2;
            
            labels = [];
            for k=1:size(bboxes,1)
                labels{k} = class_names{bboxes(k,1)+1};
            end
            
            out_img = insertObjectAnnotation(img,'rectangle',bboxes(:,2:5),labels, 'Color','r','LineWidth',2);
            imshow(out_img);
        end
        
        
        % 获取人工标注框信息
        function [train_bbox,valid_bbox] = get_truth_bbox(train_path,valid_path)
            train_bbox = cell(length(train_path),1);
            valid_bbox = cell(length(valid_path),1);
            
            h = waitbar(0,'开始读取...');
            for k=1:length(train_path)
                txt_file = train_path{k};
                txt_file = replace(txt_file,'images','labels');
                txt_file = replace(txt_file,'jpg','txt');
                fid = fopen(txt_file,'r');
                train_bbox{k,1} = cell2mat(textscan(fid,'%f %f %f %f %f','delimiter','\n'));
                fclose(fid);
                
                % 进度条
                waitbar(k/length(train_path),h,strcat('读取完成度：',num2str(k/length(train_path)*100),'%'));
            end
            
            for k=1:length(valid_path)
                txt_file = valid_path{k};
                txt_file = replace(txt_file,'images','labels');
                txt_file = replace(txt_file,'jpg','txt');
                fid = fopen(txt_file,'r');
                valid_bbox{k,1} = cell2mat(textscan(fid,'%f %f %f %f %f','delimiter','\n'));
                fclose(fid);
            end
            
        end
    end
end
