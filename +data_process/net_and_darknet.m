
classdef net_and_darknet
    
    methods(Static)
        % 转换Matlab的网络权重为darknet格式的权重,并替换配置文件中的anchors
        function save_weight(mat_file, save_file, cfg_file)
            anchors = [];
            data = load(mat_file);
            model = data.model; % 加载数据中的model
            
            fid = fopen(save_file,'wb'); % 打开写权重的文件
            head = zeros(1,4); % darknet权重文件前4个值
            fwrite(fid,head,'int');
            for k=1:length(model.Layers)
                layer = model.Layers(k);
                if contains(class(layer),'Convolution2D') % 卷积层
                    % 是否存在batchNorm
                    if contains(class(model.Layers(k+1)),'batchNormalization') && k+1<length(model.Layers)
                        batchNorm = model.Layers(k+1);
                        offset = reshape(batchNorm.Offset,1,[]); % Offset
                        fwrite(fid, offset, 'float');
                        scale = reshape(batchNorm.Scale,1,[]); % Scale
                        fwrite(fid, scale, 'float');
                        trainedMean = reshape(batchNorm.TrainedMean,1,[]); % TrainedMean
                        fwrite(fid, trainedMean, 'float');
                        trainedVar = reshape(batchNorm.TrainedVariance,1,[]); % TrainedVariance
                        fwrite(fid, trainedVar, 'float');
                    else %如果不存在batchNorm层，写bias
                        B = layer.Bias;
                        B = reshape(B,1,[]);
                        fwrite(fid,B,'float');
                    end
                    
                    % 写入Weights
                    W = layer.Weights;
                    W = permute(W,[2,1,3,4]); % matlab是按照列拼凑一起
                    W = reshape(W,1,[]); % 转换成[1,N]
                    fwrite(fid,W,'float');
                end
                
                if contains(layer.Name,'yolo') %获取anchors
                    anchors=[anchors;layer.anchors];
                end
            end
            fclose(fid);
            import data_process.net_and_darknet.replace_anchors
            replace_anchors(cfg_file,anchors); %替换anchors
            disp('darknet weight wirte success!');
        end
        
        
        % 替换cfg文件中的anchors
        function replace_anchors(cfg_file,anchors)
            import data_process.data_util.get_file_context
            
            anchors = sort(anchors); % 排序
            tmp = anchors';
            tmp = tmp(:);
            anchors_str = "";
            for i=1:length(tmp) % 构建str
                if i~=length(tmp)
                    anchors_str = strcat(anchors_str,num2str(tmp(i)),", ");
                else
                    anchors_str = strcat(anchors_str,num2str(tmp(i)));
                end
            end
            
            context = get_file_context(cfg_file);
            split_name = split(cfg_file,'.');
            new_file_name = strcat(split_name{1},'_replace_anchor','.cfg');
            fid = fopen(new_file_name,'wt');
            for k=1:length(context)
                line_str = context{k};
                if contains(line_str,'anchors')
                    line_str = strcat("anchors =  ", anchors_str);
                end
                fprintf(fid,'%s\n',line_str);
            end
            fclose(fid);
            disp('replace anchors success!');
        end
             
        
        % 加载darknet格式的网络为matlab框架结构的net
        function  model = weight_load_net(cfg_file,weight_file,names_file)

            import train_pack.loadNet_matlab
            Net = loadNet_matlab(cfg_file,weight_file);

            for k=1:length(Net.myNet.Layers)
                if contains(Net.myNet.Layers(k).Name,'yolo')
                    yolo_layer = Net.myNet.Layers(k);
                    yolo_layer.names_path = names_file;
                    Net.myNet = replaceLayer(Net.myNet,yolo_layer.Name,yolo_layer);
                end
            end

            model = dlnetwork(Net.myNet);
        end    
    end
    
end
