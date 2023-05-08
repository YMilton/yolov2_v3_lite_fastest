
% 训练时的处理
classdef yoloTrain
    
    properties
        cfg_file 
        dlnet % 网络
        data_loader % 存储训练数据的对象
        execute_environment = 'auto' % 在cpu还是gpu上执行
        
        gpu=0;
        yolo_idxs; % 网络输出层的索引
        
        % 学习参数
        LR = 0.001 % 学习率
        num_epochs = 80 % 迭代周期
        warmup_period = 1000 % 学习率增加结束时期
        
        overlap_thresh = 0.01 % 计算真实框与所有预测框的交并比阈值
    end
    
    
    methods
        % yoloTrain(cfg_file,train_data_file,weight_file)
        function obj = yoloTrain(cfg_file,train_data_file,varargin)
            import train_pack.loadNet_matlab
            import data_process.dataloader
            
            obj.cfg_file = cfg_file;
            tic
            if nargin==1 % 如果有预处理权重文件
                Net = loadNet_matlab(cfg_file,varargin{1});
            else
                Net = loadNet_matlab(cfg_file);
            end
            rng(0) % 指定随机生成器种子
            obj.dlnet = dlnetwork(Net.myNet);
            
            % 找出网络中输出层的索引
            obj.yolo_idxs = Net.output_idxs;
            obj.data_loader = dataloader(train_data_file);
            obj.data_loader.net_info = Net.net_info; % 赋值cfg文件内容中[net]的属性
            fprintf('load net and init parm elapsed time: %.3fs.\n',toc);
        end
        
        
        % 替换网络中的anchors为训练anchors
        function obj = replace_anchors(this)
            num_anchors = 0;
            % 网络中anchors替换成聚类的anchors
            for k=1:length(this.yolo_idxs)
                yolo_idx = this.yolo_idxs(k);
                yolo_anchor_size = size(this.dlnet.Layers(yolo_idx).anchors,1);
                num_anchors = num_anchors + yolo_anchor_size;
                anchor_sizes(k) = yolo_anchor_size;
            end
            
            % 聚类训练集，获取训练集的anchors(基于数据图像映射的锚点)
            rng(0);
            [anchors,meanIoU] = this.data_loader.estimate_anchors(num_anchors,0);
            disp("meanIoU: "+meanIoU);
            % 根据面积降序排列anchors
            area = anchors(:,1).*anchors(:,2);
            [~,idx] = sort(area,'ascend');
            anchors = anchors(idx,:); % 排序后的锚点框
            
            % 给网络的yolo层赋值训练的锚点框
            net = layerGraph(this.dlnet);
            anchor_idxs = cumsum(anchor_sizes); % 累加
            for k=1:length(this.yolo_idxs)
                yolo_idx = this.yolo_idxs(k);
                yolo_layer = net.Layers(yolo_idx);
                yolo_layer.anchors = anchors(anchor_idxs(k)-anchor_sizes(k)+1:anchor_idxs(k),:);
                yolo_layer.names_path = this.data_loader.names;
                net = replaceLayer(net,yolo_layer.Name,yolo_layer);
            end
            this.dlnet = dlnetwork(net);
            obj = this;
        end
        
        
        % 模型的训练
        function train(this)
            import data_process.batch_process.get_batch_imgs
            import train_pack.yoloTrain.*
            
            % 训练图绘制
            figure;
            ax1 = subplot(231); ylabel('lbox');xlabel('iteration'); lbox_ploter = animatedline(ax1,'Color','r'); axis([0,Inf,0,Inf]); title('lbox'); 
            ax2 = subplot(232); ylabel('lconf');xlabel('iteration'); lconf_ploter = animatedline(ax2,'Color','g'); axis([0,Inf,0,Inf]); title('lconf'); 
            ax3 = subplot(233); ylabel('lcls');xlabel('iteration'); lcls_ploter = animatedline(ax3,'Color','b'); axis([0,Inf,0,Inf]); title('lcls'); 
            ax4 = subplot(234); ylabel('loss');xlabel('iteration'); loss_ploter = animatedline(ax4,'Color','c'); axis([0,Inf,0,Inf]); title('loss');
            ax5 = subplot(235); ylabel('learningRate');xlabel('iteration'); lr_ploter = animatedline(ax5,'Color','k'); axis([0,Inf,0,Inf]); title('learningRate'); 
            
            
            dl = this.data_loader;
            train_img_num = length(dl.train_path);
            net_info = dl.net_info;
            input_size = [net_info.width,net_info.height,net_info.channels]; % 网络输入尺寸
            batch = net_info.batch/net_info.subdivisions;
            if batch==1 % 如果batch=1修改batch
                batch = 8;
            end
            
            vel = [];
            learningRate = this.LR;
            nEpochs = this.num_epochs;
            warmupPeriod = this.warmup_period;
            iscut = false;
            for epoch = 1:nEpochs
                % shuffle训练的图像数据
                dl.train_path = this.shuffle_train_path(dl.train_path);
                for k=1:batch:train_img_num
                    tic
                    begin_idx = k;
                    end_idx = min(k + batch - 1,train_img_num);
                    batch_img_paths = dl.train_path(begin_idx:end_idx);
                    [batch_imgs,batch_gt] = get_batch_imgs(batch_img_paths, input_size, 1);
                    Xtrain = dlarray(batch_imgs,'SSCB');
                    % 采用cpu还是gpu，strcpmi忽略大小写
                    if (strcmpi(this.execute_environment,'auto') && canUseGPU) || strcmpi(this.execute_environment,'gpu')
                        this.gpu=1;
                        Xtrain = gpuArray(Xtrain);
                    end

                    [lbox,lconf,lcls,deltas,state] = dlfeval(@compute_gradients,this.dlnet,Xtrain,this.yolo_idxs, batch_gt, this.overlap_thresh);
                    
                    % 不恰当的学习率
                    if lbox==0 && lconf==0 && lcls==0 && deltas==0
                        disp("Please reset learning rate! currentLR="+currentLR);
                        iscut = true;
                        break;
                    end 
                    
                    iteration = (end_idx+(epoch-1)*train_img_num)/batch;
                    currentLR = this.warmup_lr(iteration, epoch, learningRate, warmupPeriod, nEpochs); % 学习率改变策略
                    
                    % 判断计算的梯度是否正常
                    if ~is_abnormal(deltas)
                        deltas = dlupdate(@(g,w) g + 0.0005*w, deltas, this.dlnet.Learnables);

                        [this.dlnet, vel] = sgdmupdate(this.dlnet, deltas, vel, currentLR); % SGDM更新模型的学习参数

                        this.dlnet.State = state; % 更新网络中的state参数
                    else
                        disp("The gradient extinction or explosion! currentLR=" + currentLR);
                        iscut = true;
                        break;
                    end
 
                    total_loss = lbox + lconf + lcls;
                    disp("Epoch: "+epoch+" | batch/total_imgs: ["+end_idx+"/"+train_img_num+"] | LR: "+currentLR+...
                        " | Box loss: "+lbox+" | Object loss: "+lconf+" | Class loss: "+lcls+" | Loss: "+total_loss);
                    
                    % 剩余时间计算
                    batch_time = toc;
                    total_time = batch_time*((train_img_num - end_idx)/batch)+batch_time*(train_img_num/batch)*(nEpochs-epoch); %时间为秒
                    day = floor(total_time/(24*60*60)); hours = floor(mod(total_time,24*60*60)/3600);
                    minutes = floor(mod(mod(total_time,24*60*60),3600)/60); seconds = mod(mod(mod(total_time,24*60*60),3600),60);
                    fprintf('Time left: %ddays %dhours %dminutes %.3fseconds.\n',day,hours,minutes,seconds);
                    
                    % 绘制动态图
                    x = (end_idx+(epoch-1)*train_img_num)/batch;
                    addpoints(lbox_ploter,x,lbox);
                    addpoints(lconf_ploter,x,lconf); 
                    addpoints(lcls_ploter,x,lcls); 
                    addpoints(loss_ploter,x,total_loss);
                    addpoints(lr_ploter,x,currentLR); 
                    drawnow;
                end
                
                % 保存模型
                if((epoch>40 && mod(epoch,20)==0) || epoch==nEpochs || iscut)
                    [~,name,~] = fileparts(this.cfg_file);
                    path = './save_model';
                    if ~exist(path,'dir')
                        mkdir(path);
                    end
                    matlabModel = fullfile(path,[name,'_epoch',num2str(epoch),'.mat']);
                    model = this.dlnet; output_idxs = this.yolo_idxs;
                    save(matlabModel,'model','output_idxs');
                    if iscut % 模型训练出现NaN，保存网络并结束迭代
                        break;
                    end
                end
            end
            saveas(gcf,'train_fig.jpg');
        end
    end
    
    
    methods(Static)
        
        % 学习率更新
        function current_lr = warmup_lr(iter,epoch,lr,warmupPeriod,nEpochs)
            persistent warmUpEpoch;
            
            if iter<warmupPeriod % 增加学习率
                current_lr = lr * ((iter/warmupPeriod)^4);
                warmUpEpoch = epoch; % 记录学习率增加结束的时期epoch
                
            elseif iter >= warmupPeriod && epoch < warmUpEpoch+floor(0.6*(nEpochs-warmUpEpoch))
                current_lr = lr;
                
            elseif epoch >= warmUpEpoch + floor(0.6*(nEpochs-warmUpEpoch)) && epoch < warmUpEpoch+floor(0.9*(nEpochs-warmUpEpoch))
                current_lr = lr*0.1;
                
            else
                current_lr = lr*0.01;
                
            end
        end
        
        
        % 计算损失，梯度与batch_norm中的滑动均值与方差
        function [lbox,lconf,lcls,deltas,state] = compute_gradients(dlnet, dlX, yolo_idxs, batch_gt,overlap_thresh)
            % batch_imgs=[rows,cols,channels,batch_num]; 
            % batch_gt cols:[batch_idx,cls_idx,ccx,cy,w,h],normlization
            import train_pack.yoloTrain.*
            
            yolo_outputs = cell(length(dlnet.OutputNames),1);
            [yolo_outputs{:},state] = forward(dlnet,dlX,'Outputs',dlnet.OutputNames); % 前向传播获取特征
            
            assert(length(yolo_outputs)==length(yolo_idxs),'yolo_idxs must be equal to yolo_outputs!');
            
            % 损失计算(包括三层yolo)
            lbox = dlarray(0); lconf = dlarray(0); lcls = dlarray(0);
            for n = 1:length(yolo_outputs)
                yolo_layer = dlnet.Layers(yolo_idxs(n));
                pred = yolo_outputs{n};
                [rows,cols,anchor_feature,batch_size] = size(pred);
                % %%%%%matlab的reshap是按列进行的，一定注意。如果reshape为[13,13,5,8,6]就数错误的%%%%%%
                pred = reshape(pred,rows,cols,anchor_feature/size(yolo_layer.anchors,1),size(yolo_layer.anchors,1),batch_size); 
                pred = permute(pred,[1,2,4,5,3]);
                
                % 对tx，ty, conf, cls_conf做sigmoid，tx,ty映射为偏移量,conf,cls_conf映射为目标置信度、类别置信度
                pred(:,:,:,:,1:2) = my_network.sigmoid(pred(:,:,:,:,1:2)); 
                pred(:,:,:,:,5:end) = my_network.sigmoid(pred(:,:,:,:,5:end));
                yolo_layer.map_size = [cols,rows]; % [w,h]
                yolo_layer.batch_size = batch_size;
                % 如果锚点表示的图像像素，则重新计算anchors到feature map上
                if max(max(yolo_layer.anchors))>20
                    yolo_layer.anchors = yolo_layer.anchors./yolo_layer.input_size(1:2).*yolo_layer.map_size;
                end
                
                % 映射到特征上
                anchors = yolo_layer.anchors; % 映射到特征上的anchor
                [xs_pos,ys_pos] = ndgrid(0:cols-1,0:rows-1,1:size(anchors,1),1:batch_size);
                pred_bbox(:,:,:,:,1) = extractdata(pred(:,:,:,:,1)) + xs_pos;
                pred_bbox(:,:,:,:,2) = extractdata(pred(:,:,:,:,2)) + ys_pos;
                reshape_anch = reshape(anchors,1,1,size(anchors,1),1,size(anchors,2));
                pred_bbox(:,:,:,:,3:4) = exp(extractdata(pred(:,:,:,:,3:4))).*reshape_anch; %%
                
                
                % 找出标注框在precess_pred中的位置
%                 gts = build_targets(yolo_layer,batch_gt);
%                 [lbox_,lconf_,lcls_] = compute_loss(pred,gts,yolo_layer.anchors,pred_bbox,overlap_thresh);
                                              
                [tcls,tbox,indices,anchor_grids] = build_targets2(yolo_layer,batch_gt);
                % pred纯预测值，pred_bbox在pred上sigmoid、exp后的值，tcls、tbox：目标真实值，anchor_grids:被选中的anchor,overlap_thresh：iou阈值(预测框与真实框)
                [lbox_,lconf_,lcls_] = compute_loss2(pred,tcls,tbox,indices,anchor_grids,pred_bbox, overlap_thresh);

                fprintf('real loss: lbox=%.3f, lconf=%.3f, lcls=%.3f\n',lbox_, lconf_, lcls_);
                if ~isnan(lbox_) && ~isnan(lconf_) && ~isnan(lcls_) && ~isinf(lbox_) && ~isinf(lconf_) && ~isinf(lcls_)
                    lbox = lbox + lbox_; lconf = lconf + lconf_; lcls = lcls + lcls_;
                end
            end
            
            % 梯度的计算
            total_loss = lbox + lconf + lcls;
            try
                deltas = dlgradient(total_loss,dlnet.Learnables);
            catch E % total_loss=0抛出异常
                deltas = 0;
                disp(E.message);
            end
            
            lbox = double(gather(extractdata(lbox)));
            lconf = double(gather(extractdata(lconf)));
            lcls = double(gather(extractdata(lcls)));
        end
        
        
        %============================================================================================%
        function [lbox,lconf,lcls] = compute_loss2(pred, tcls, tbox, indices, anchor_grids, pred_bbox, overlap_thresh)
            import train_pack.yoloTrain.filter_by_iou
            
            [grid_h,grid_w,anchor_size,batch_size,feature_size] = size(pred);
            bs = indices(:,1); % batch_pos
            as = indices(:,2); % anchor_pos
            grid_xs = indices(:,3); 
            grid_ys = indices(:,4);
            
            tbox_ = tbox; % 映射在特征图上的的box
            % 把真实值xywh映射为sigmoid(tx) sigmoid(ty) tw th
            tbox(:,1:2) = tbox(:,1:2) - floor(tbox(:,1:2)); % xy的偏移量
            tbox(:,3:4) = log(tbox(:,3:4)./anchor_grids); % tw th
            
            % 根据预测box与真实目标box计算iou,除indices位置外剔除iou>threshold的位置
            conf_mask = ones(grid_h,grid_w,anchor_size,batch_size,'like',pred);
            conf_mask = filter_by_iou(conf_mask,pred_bbox,[bs,tbox_], overlap_thresh);
            
            % 提取与gts对应的预测值
            tobj = zeros(grid_h,grid_w,anchor_size,batch_size,'like',pred);
            tcls_ = zeros(grid_h,grid_w,anchor_size,batch_size,feature_size-5,'like',pred);
            pred_rows = zeros(length(bs),feature_size,'like',pred);
            for k=1:length(bs)
                row = pred(grid_ys(k),grid_xs(k),as(k),bs(k),:);
                row = squeeze(row);
                pred_rows(k,:) = row';
                tobj(grid_ys(k),grid_xs(k),as(k),bs(k)) = 1.0;
                conf_mask(grid_ys(k),grid_xs(k),as(k),bs(k)) = 1.0;
                tcls_(grid_ys(k),grid_xs(k),as(k),bs(k),tcls(k)+1) = 1.0; % label,matlab从1开始
            end
            cls_mask = tcls_;
            
            % (1)预测框损失计算lbox(MSE)
            scales = 2 - (tbox_(:,3)/grid_w).*(tbox_(:,4)/grid_h); % 小目标误差系数
            pbbox = pred_rows(:,1:4);
            lbox = mse(pbbox.*scales,tbox.*scales,'DataFormat','SS');

            % (2) 置信度损失计算lconf (BCE)
            pred_obj = pred(:,:,:,:,5);
            lconf = crossentropy(pred_obj.*conf_mask,tobj.*conf_mask,'DataFormat','SSCB','TargetCategories','independent'); % TargetCategories:independent多标签分类

            % (3)类别损失计算lcls (BCE)
            pred_cls = pred(:,:,:,:,6:end);
            if numel(size(pred_cls))==4
                lcls = crossentropy(pred_cls.*cls_mask,tcls_.*cls_mask,'DataFormat','SSCB','TargetCategories','independent');
            else
                lcls = crossentropy(pred_cls.*cls_mask,tcls_.*cls_mask,'DataFormat','SSCBU','TargetCategories','independent');
            end
        end
        
        function [tcls,tbox,indices,anchor_grids] = build_targets2(yolo_layer,batch_gt)
            import data_process.data_util.iou_anchors 
            
            map_size = yolo_layer.map_size;
            % 将batch_gt的box投影到特征图上
            gts = batch_gt;
            gts(:,3:6) = gts(:,3:6).*[map_size,map_size];
            
            % 标注框与锚点计算交并比，获取最大iou值与max_iou时anchors中的索引
            anchors = yolo_layer.anchors;
            for b=1:yolo_layer.batch_size
                whs = gts(gts(:,1)==b,5:6); % 提取第b幅图像所有人工标注框的wh
                % 每个人工标注框与anchors计算最大iou与索引
                [best_iou, best_idx] = iou_anchors(whs,anchors);
                % 增加到gt的7、8列中
                gts(gts(:,1)==b,7) = best_idx;
                gts(gts(:,1)==b,8) = best_iou;
            end
            
            tcls = gts(:,2);
            tbox = gts(:,3:6); % 映射到特征图上的box
            % [batch_pos,anchor_pos,xs_pos,ys_pos]
            indices = [gts(:,1),gts(:,7),floor(gts(:,3)) + 1,floor(gts(:,4)) + 1]; % matlab数组索引从1开始
            % anchor_pos对应的wh
            for k=1:length(tcls)
                anchor_grids(k,:) = anchors(gts(k,7),:);
            end
        end
        
        
        %=================================================================%
        % 计算某个yolo输出与ground_truth的损失 pred size=[rows,cols,anchor_num,batch_size,5+classes]
        function [lbox,lconf,lcls] = compute_loss(pred, gts, anchors, pred_bbox, overlap_thresh)
            import train_pack.yoloTrain.pad_mask
            import train_pack.yoloTrain.filter_by_iou
            
            % 保留原始gt为gt_
            gts_ = gts;
            [grid_h,grid_w,anchor_size,batch_size,feature_size] = size(pred);
            
            % 位置获取
            bs = gts(:,1); % batch_pos
            as = gts(:,7); % anchor_pos
            grid_xs = floor(gts(:,3)) + 1; % matlab数组索引从1开始
            grid_ys = floor(gts(:,4)) + 1;
            
            % 转换xywh到没有sigmoid与exp前的数据 [batch_num,class_num,sigmoid(tx),sigmoid(ty),tw,th,anchor_num,max_iou]
            gts(:,3:4) = gts(:,3:4) - floor(gts(:,3:4)); % gt_sigmoid(tx)
            gts(:,5:6) = log(gts(:,5:6)./anchors(gts(:,7),:));
            
%             best_ious = gts(:,8); % 通过阈值筛选，筛选出IOU>阈值的位置作为目标，算损失
%             filter = best_ious>overlap_thresh;
%             if sum(filter)==0
%                 lbox=0;lconf=0;lcls=0;
%                 disp('all best_ious less than 0.5!');
%                 return;
%             end
%             bs = bs(filter); as = as(filter); grid_xs = grid_xs(filter); grid_ys = grid_ys(filter);
%             gts = gts(filter,:);
            % 提取与gts对应的预测值
            pred_rows = zeros(length(bs),size(pred,5),'like',pred);
            for k=1:length(bs)
                row = pred(grid_ys(k),grid_xs(k),as(k),bs(k),:);
                row = squeeze(row);
                pred_rows(k,:) = row';
            end
            
            % (1) 预测框损失计算lbox(MSE)
            n = size(pred_rows,1); % 批集合中目标个数
            scales = 2 - (gts_(:,5)/grid_w) .* (gts_(:,6)/grid_h); % 目标误差系数，针对小目标
%             scales = scales(filter,:); % 过滤后与pred_row,gts维度保持一致
            lbox = 1/(2*n)*sum(scales.*(pred_rows(:,1) - gts(:,3)).^2 + scales.*(pred_rows(:,2) - gts(:,4)).^2 ...
                    + scales.*(pred_rows(:,3) - gts(:,5)).^2 + scales.*(pred_rows(:,4) - gts(:,6)).^2);

            
            % (2) 置信度损失计算lconf (BCE)
            conf_mask = ones(grid_h,grid_w,anchor_size,batch_size);
            % 对训练过程中预测框与真实框IoU大于阈值的位置，置信度置为0，剔除干扰项
            conf_mask = filter_by_iou(conf_mask,pred_bbox,gts_(:,[1,3:6]),overlap_thresh);  % overlap_thresh真实框与预测框交并比阈值
            conf_mask = pad_mask(conf_mask,grid_ys,grid_xs,as,bs,1);

            gt_conf = zeros(grid_h,grid_w,anchor_size,batch_size);
            gt_conf = pad_mask(gt_conf,grid_ys,grid_xs,as,bs,1);
            
            % 为了保证训练时不出现INF与NAN,即-Inf=log(0),NaN = 0*log(0),预测值加上一个很小的值
            pred_conf = pred(:,:,:,:,5).*conf_mask; % 预测置信度
            % 消除矩阵中出现的0与1，防止log(val)与log(1-val)存在INF
            pred_conf(pred_conf==0) = 1e-15; pred_conf(pred_conf==1) = 1 - 1e-15; 
            gt_conf = gt_conf.*conf_mask;
            % BCE计算
            lconf = 1/(2*n) * -sum(gt_conf.*log(pred_conf) + (1-gt_conf).*log(1-pred_conf),'all'); % 包括有目标与无目标的计算

            
            % (3) 类别损失计算lcls (BCE)
            labels = gts(:,2) + 1; % label,matlab从1开始
            gt_cls_conf = zeros(grid_h,grid_w,anchor_size,batch_size,feature_size-5,'like',pred);
            for k=1:size(bs,1)
                gt_cls_conf(grid_ys(k),grid_xs(k),as(k),bs(k),labels(k)) = 1;
            end
            cls_mask = logical(gt_cls_conf);
            pred_cls_conf = pred(:,:,:,:,6:end);
            pred_cls_conf(pred_cls_conf==0) = 1e-15; pred_cls_conf(pred_cls_conf==1) = 1 - 1e-15;
            lcls = 1/(2*n) * -sum(gt_cls_conf(cls_mask).*log(pred_cls_conf(cls_mask)) + (1-gt_cls_conf(cls_mask)).*log(1-pred_cls_conf(cls_mask)));
            
            if isnan(lbox) || isnan(lconf) || isnan(lcls) || isinf(lbox) || isinf(lconf) || isinf(lcls)
                 disp('INF or NAN!');
            end
        end
        
        % 对标准化的ground_truth数据做处理
        function gts = build_targets(yolo_layer,batch_gt)
            % return: [batch_num,class_num,cx,cy,w,h,anchor_num,max_iou]
            % 其中cx,cy,w,h为归一化的数据
            import data_process.data_util.iou_anchors 
            
            map_size = yolo_layer.map_size;
            % 将batch_gt的box投影到特征图上，batch_gt的bbox是归一化数据
            gts = batch_gt; 
            gts(:,3:6) = gts(:,3:6).*[map_size,map_size];
            
            % 标注框与锚点计算交并比，获取最大iou值与max_iou时anchors中的索引
            anchors = yolo_layer.anchors;
            for b=1:yolo_layer.batch_size
                whs = gts(gts(:,1)==b,5:6); % 提取第b幅图像所有人工标注框的wh
                % 每个人工标注框与anchors计算最大iou与索引
                [best_iou, best_idx] = iou_anchors(whs,anchors);
                % 增加到gt的7、8列中
                gts(gts(:,1)==b,7) = best_idx;
                gts(gts(:,1)==b,8) = best_iou;
            end
        end
        
        % 通过阈值过滤掉除真实预测位置以外(grid_xs,grid_ys,as,bs)，其他大于阈值的位置设置为0，无目标,防止干扰
        function conf_mask = filter_by_iou(mask,pred_box,gt_box,threshold)
            pred_box = gather(pred_box);
            [grid_r,grid_c,anchor_num,batch_num,~] = size(pred_box);
            pred_box = reshape(pred_box,[],batch_num,4);
            pred_box = permute(pred_box,[1,3,2]);
            for b=1:batch_num
                batch_box = pred_box(:,:,b);
                batch_gt_box = gt_box(gt_box(:,1)==b,2:end);
                % 每个预测框与真实框计算IOU，取最大的IOU
%                 batch_ious = [];
%                 for k=1:size(batch_gt_box,1)
%                     tmp_ious = my_network.bbox_iou(batch_box, batch_gt_box(k,:));
%                     batch_ious = [batch_ious, tmp_ious];
%                 end
                batch_ious = bboxOverlapRatio(batch_box,batch_gt_box);
                batch_ious = max(batch_ious,[],2); % 每一行求最大值
                batch_ious = reshape(batch_ious,grid_r,grid_c,anchor_num);
                ious(:,:,:,b) = batch_ious;
            end
            mask(ious>threshold) = 0;
            conf_mask = mask;
        end
        
        %=================================================================%
        % 填充mask
        function out_mask = pad_mask(mask,ys,xs,as,bs,val)
            for k=1:size(bs,1)
                mask(ys(k),xs(k),as(k),bs(k)) = val;
            end
            out_mask = mask;
        end
        
        
        % 随机打乱训练样本顺序
        function shuffle_path = shuffle_train_path(train_path)
            shuffle_path = train_path;
            idxs = randperm(length(train_path));
            for k=1:length(train_path)
                shuffle_path{k} = train_path{idxs(k)};
            end
        end
        
        
        % 限制数据范围
        function val = set_range(val,min_val,max_val)
            val(val>max_val) = max_val;
            val(val<min_val) = min_val;
        end
        
        
        % 判断deltas中是否存在NaN与Inf
        function flag = is_abnormal(deltas)
            flag = false;
            vals = deltas.Value;
            for k=1:length(vals)
                sum_val = sum(vals{k},'all');
                if isnan(sum_val) || isinf(sum_val)
                    flag = true;
                    break;
                end
            end
        end
    end
    
end