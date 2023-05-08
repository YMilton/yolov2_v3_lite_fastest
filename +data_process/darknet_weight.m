% 转换Matlab的网络权重为darknet格式的权重
classdef darknet_weight
    
    methods(Static)
        
        function save_weight(mat_file, save_file)
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
            end
            fclose(fid);
            disp('darknet weight wirte success!');
        end
        
    end
    
end
