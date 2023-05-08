% 损失计算
classdef loss

    methods(Static)  
        % 计算一个batch的损失, pred_yolo:预测结果(yolo类)，ground_truth:人工标注框
        % size(delta) = [batch_size,grid_h,grid_w,anchor_size,feature_size]
        function [lbox,lconf,lcls,L,delta] = batch_loss(pred, gts)
            
            % 找出人工标注框对应的预测值，ps与gt_matrix的尺寸相同
            bs = gts(:,1); % batch_idx
            as = gts(:,7); % anchor_num
            grid_xs = floor(gts(:,3)) + 1; % grid_x向下取整数,其中matlab数组索引从1开始
            grid_ys = floor(gts(:,4)) + 1; % grid_y
            import data_process.loss
            ps = loss.pred_subset(pred,bs,as,grid_xs,grid_ys);
            
            n = size(ps,1); % batch中目标框个数
            % (1)预测框损失计算lbox（MSE）
            lbox = 1/(2*n)*sum((ps(:,1) - gts(:,3)).^2 + (ps(:,2) - gts(:,4)).^2 ...
                    + (ps(:,3) - gts(:,5)).^2 + (ps(:,4) - gts(:,6)).^2);
                
            % (2)置信度损失计算lconf（BCE）
            [batch_size,grid_h,grid_w,anchor_size,feature_size] = size(pred);
            obj_idx = zeros(batch_size,grid_h,grid_w,anchor_size);
            noobj_idx = ones(batch_size,grid_h,grid_w,anchor_size);
            obj_idx = logical(loss.pad_mask(obj_idx,bs,as,grid_xs,grid_ys,1));
            best_ious = gts(:,8); % 通过阈值筛选，大于某个阈值表示有目标，反之无目标
            filter = best_ious>0.5;
            noobj_idx = logical(loss.pad_mask(noobj_idx,bs(filter),as(filter),grid_xs(filter),grid_ys(filter),0));
            pred_conf = pred(:,:,:,:,5); % 预测置信度
            gt_conf = obj_idx;
            % BCE计算
            lconf_obj = -sum(gt_conf(obj_idx).*log(pred_conf(obj_idx)) + (1-gt_conf(obj_idx)).*log(1-pred_conf(obj_idx)));
            lconf_noobj = -sum(gt_conf(noobj_idx).*log(pred_conf(noobj_idx)) + (1-gt_conf(noobj_idx)).*log(1-pred_conf(noobj_idx)));
            lconf = 1/n * (lconf_obj + 0.5*lconf_noobj);
            
            % (3)类别损失计算lcls
            labels = gts(:,2) + 1; % label,matlab从1开始
            gt_cls_conf = loss.pad_cls_conf(pred,bs,as,grid_xs,grid_ys,labels);
            cls_obj_idx = logical(gt_cls_conf);
            pred_cls_conf = pred(:,:,:,:,6:end);
            lcls = 1/n * -sum(gt_cls_conf(cls_obj_idx).*log(pred_cls_conf(cls_obj_idx)) + (1-gt_cls_conf(cls_obj_idx)).*log(1-pred_cls_conf(cls_obj_idx)));
            
            L = lbox + lconf + lcls;
            
            % ======损失函数到yolo输出的偏导====== %
            delta = zeros(batch_size,grid_h,grid_w,anchor_size,feature_size);
            % delta_xywh
            for k=1:size(ps,1)
                delta(bs(k,1),grid_ys(k,1),grid_xs(k,1),as(k,1),1) = 1/n * (ps(k,1) - gts(k,3));
                delta(bs(k,1),grid_ys(k,1),grid_xs(k,1),as(k,1),2) = 1/n * (ps(k,2) - gts(k,4));
                delta(bs(k,1),grid_ys(k,1),grid_xs(k,1),as(k,1),3) = 1/n * (ps(k,3) - gts(k,5));
                delta(bs(k,1),grid_ys(k,1),grid_xs(k,1),as(k,1),4) = 1/n * (ps(k,4) - gts(k,6));
            end
            % delta_conf:
            num = batch_size*grid_h*grid_w*anchor_size;
            obj_idx_ = obj_idx(:); noobj_idx_ = noobj_idx(:);
            pred_conf_ = pred_conf(:); gt_conf_ = gt_conf(:);
            delta_conf = zeros(num,1);
            for k=1:num
                if obj_idx_(k)==1 % 如果为1，后面项为0(因为gt_conf=1)
                    % delta_conf(k,1) = 1/n*(1/pred_conf_(k));
                    delta_conf(k,1) = 1/n*((pred_conf_(k)-gt_conf_(k))/(pred_conf_(k)*(1-pred_conf_(k))));
                end
                if noobj_idx_(k)==1 % gt_conf基本为0,也有可能为1
                    delta_conf(k,1) = 0.5/n*((pred_conf_(k)-gt_conf_(k))/(pred_conf_(k)*(1-pred_conf_(k))));
                end
            end
            delta(:,:,:,:,5) = reshape(delta_conf,batch_size,grid_h,grid_w,anchor_size);
            % delta_cls_conf
            num = batch_size*grid_h*grid_w*anchor_size*(feature_size-5);
            gt_cls_conf_ = gt_cls_conf(:);cls_obj_idx_ = cls_obj_idx(:);
            pred_cls_conf_ = pred_cls_conf(:);
            delta_cls_conf = zeros(num,1);
            for k=1:num
                if cls_obj_idx_(k)==1
                    delta_cls_conf(k,1) = 1/n*((pred_cls_conf_(k)-gt_cls_conf_(k))/(pred_cls_conf_(k)*(1-pred_cls_conf_(k))));
                end
            end
            delta(:,:,:,:,6:end) = reshape(delta_cls_conf,batch_size,grid_h,grid_w,anchor_size,feature_size-5);
        end
          
        
        % 填充目标类置信度
        function gt_cls_conf = pad_cls_conf(pred,bs,as,grid_xs,grid_ys,labels)
            [b,rows,cols,a,cc] = size(pred);
            gt_cls_conf = zeros(b,rows,cols,a,cc-5); % cc-5为目标类别数目
            for k=1:size(bs,1)
                gt_cls_conf(bs(k,1),grid_ys(k,1),grid_xs(k,1),as(k,1),labels(k,1)) = 1;
            end
        end
        
        
        % 填充mask
        function out_mask = pad_mask(mask,bs,as,grid_xs,grid_ys,val)
            for k=1:size(bs,1)
                mask(bs(k,1),grid_ys(k,1),grid_xs(k,1),as(k,1)) = val;
            end
            out_mask = mask;
        end
        
        
        % 根据条件筛选预测结果
        function pred_rows = pred_subset(pred,bs,as,grid_xs,grid_ys)
            for k=1:size(bs,1)
                row = pred(bs(k,1),grid_ys(k,1),grid_xs(k,1),as(k,1),:);
                row = row(:)';
                pred_rows(k,:) = row;
            end
        end
    end
    
end