% 模型评估
classdef modelEvaluate
    
    properties(Access=public)
        valid_paths % 验证数据集路径
        model % dlnetwork
        
        conf_thresh = 0.5
        nms_thresh = 0.75
        score_thresh = 0.5
        
        show_PR = true % 是否显示P-R曲线
        step = 200 % bbox_cells一次放在GPU计算的图片数量，防止内存溢出
        class_names
        
        output_idxs % 网络输出层位置
        
        overlap_thresh = 0.5 % 真实框与预测框的IoU
    end
    
    methods(Access=public)
        % obj = modelEvaluate(valid_dir, model_mat)
        function obj = modelEvaluate(varargin)
            import data_process.dataloader
            
            if nargin==0
                return;
            end
            if nargin>2
                disp('Input arguments error!');
                return;
            end
            % 加载目录中的图片路径
            data_file = varargin{1};
            data_loader = dataloader(data_file);
            obj.valid_paths = data_loader.valid_path;
            
            % 加载mat中的模型
            model_mat = varargin{2};
            data = load(model_mat);
            obj.model = data.model;
            
            if isfield(data,'output_idxs') % mat文件中是否存在output_idxs
                obj.output_idxs = data.output_idxs;
            else
                obj.output_idxs = [];
                for k=1:length(obj.model.Layers)
                    layer = obj.model.Layers(k);
                    if contains(layer.Name,'yolo')
                        obj.output_idxs = [obj.output_idxs, k];
                    end
                end
            end
            
            % 获取名称
            yolo_layer = data.model.Layers(end);
            import data_process.data_util.get_file_context
            obj.class_names = get_file_context(yolo_layer.names_path); % 标签名称
        end
        
        
        % 评估模型
        function [mAP,AP,F1,recall,precision]=evaluate(this)
            import train_pack.yoloDetection.detect_bboxes
            import data_process.batch_process.get_gt_info
            import train_pack.modelEvaluate.get_metrics
            
            % 预测的元胞数组：包括bbox坐标[x1,y1,x2,y2]，标签名称+score, [conf,cls_conf,cls_idx,score]
            bbox_cells = [];
            for k = 1:this.step:length(this.valid_paths)
                begin_idx = k;
                end_idx = min(k + this.step - 1,length(this.valid_paths));
                cut_bbox_cells = detect_bboxes(this.model,this.output_idxs, this.valid_paths(begin_idx:end_idx),...
                                                this.conf_thresh, this.nms_thresh, this.score_thresh);
                bbox_cells = [bbox_cells;cut_bbox_cells];
            end
            
            i=0; % 统计有目标的图像个数
            for k=1:length(bbox_cells)
                if ~isempty(bbox_cells{k})
                    i = i + 1;
                end
            end
            fprintf('has object: %d, no object: %d\n', i, length(bbox_cells) - i);

            % 读取验证集真实数据[img_idx,cls_idx,box]
            gt = get_gt_info(this.valid_paths);
            
            metrics = get_metrics(bbox_cells, gt, this.overlap_thresh); % [tp,conf,cls_idx]
            metrics = sortrows(metrics,2,'descend'); % 按照置信度降序排列
            
            % 类别编号(不重复）
            unique_cls = unique(gt(:,2));
            AP = []; precision = []; recall = [];
            for k = 1:length(unique_cls)
                idxs = metrics(:,3)==unique_cls(k); % 找出某个类别代号在metrics的索引
                
                bboxes_c = gt(gt(:,2)==unique_cls(k),:); % 某个类别的真实框信息
                num_gt = size(bboxes_c,1);
                num_p = sum(idxs);
                
                if num_p==0 && num_gt==0
                    continue;
                elseif num_p==0 || num_gt==0
                    AP(k,1) = 0;
                    precision(k,1) = 0;
                    recall(k,1) = 0;
                else
                    % 累加计算FP与TP
                    Tps = metrics(idxs,1);
                    tpc = cumsum(Tps);
                    fpc = cumsum(1 - Tps);
                    
                    % 召回率计算，TP/(TP + FN) TP+FN: 正例预测为正例+正例预测为负例
                    recall_curve = tpc/(num_gt + 1e-16);
                    recall(k,1) = recall_curve(end);
                    
                    % 精确率计算，TP/(TP+FP)
                    precision_curve = tpc./(tpc + fpc);
                    precision(k,1) = precision_curve(end);
                    
                    % 计算AP
                    mrec = [0.05;recall_curve;1.0];
                    mpre = [1.0;precision_curve;0.05];
                    
                    delta_ws = mrec(2:end,:) - mrec(1:end-1,:);
                    delta_hs = max(mpre(2:end,:),mpre(1:end-1,:));
                    AP(k,1) = sum(delta_ws.*delta_hs); 
                end
                
                if this.show_PR % 绘制P-R曲线
                    figure
                    plot([recall_curve;1],[1;precision_curve],'-');
                    title(['className = ',this.class_names{k},', AP = ',num2str(AP(k))]);
                end
            end
            
            % 计算F1
            F1 = 2*precision.*recall./(precision + recall + 1e-16);
            mAP = mean(AP);
        end
    end
    
    
    
    methods(Static)
        
        % 统计TP、置信度、类别代号 [tp,conf,cls_idx]
        function metrics = get_metrics(bbox_cells, gt, overlap_thresh)
            metrics = [];
            for k = 1:length(bbox_cells)                
                if ~isempty(bbox_cells{k})
                    bboxes = bbox_cells{k}{1};
                    out = bbox_cells{k}{3}; % [conf,cls_conf,cls_idx,score]
                    gt_cls = gt(gt(:,1)==k,2); % 当前图像类别代号
                    gt_bboxes = gt(gt(:,1)==k,3:end); % 当前图像真实框
                    tp = zeros(size(bboxes,1),1);
                    
                    detected_idxs = []; %存储出现的类别
                    for n = 1:size(bboxes,1) % 遍历这幅图像的预测框
                        if length(detected_idxs) == size(gt_bboxes,1) % 预测目标数目与真实目标数目相同，则退出
                            break;
                        end
                        if ~ismember(out(n,3),gt_cls) %某个框的预测类别代号不在gt_cls中
                            continue;
                        end
                        
                        ious = my_network.bbox_iou(bboxes(n,:),gt_bboxes); % 计算某个预测框与当前图像所有框的IoU
                        [max_iou,max_idx] = max(ious,[],1); % 找出与真实值交并比最大值与位置
                        if max_iou>overlap_thresh  && gt_cls(max_idx)==out(n,3) && ~ismember(max_idx,detected_idxs)
                            tp(n) = 1;
                            detected_idxs = [detected_idxs;max_idx];
                        end
                       
                    end
                    metrics = [metrics;[tp,out(:,[1,3])]];
                else
                    metrics = [metrics;[0,0,0]]; % 无目标的metrics
                end
            end  
        end
        
        
    end
end