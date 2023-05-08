
% 使用matlab自带的网络层加载网络
classdef loadNet_matlab
    
    properties(Access=public)
        cfg_file
        weight_file
        net_info % cfg文件中[net]属性
        
        myNet
        output_idxs % 网络输出idx
    end
    
    
    methods(Access=public)
        function obj = loadNet_matlab(varargin)
            import train_pack.loadNet_matlab.make_network
            import train_pack.loadNet_matlab.net_load_weight
            
            if nargin<=0 && nargin>2
                return;
            end
            obj.cfg_file = varargin{1};
            if nargin==2
                obj.weight_file = varargin{2};
            end
            % 加载网络与初始化网络权重
            ln = load_net; ln.cfg_file = varargin{1};
            net_struct = ln.read_file2struct;
            obj.net_info = net_struct.net; % 赋值cfg文件内容中[net]的属性
            % lgraph_layer_names保存网络的名称与索引，同cfg文件的网络层对应
            [obj.myNet, lgraph_layer_names, obj.output_idxs] = make_network(net_struct);
            
            % 加载权重
            obj.myNet = net_load_weight(obj.myNet,lgraph_layer_names,net_struct, obj.weight_file);
        end
    end
    
    
    methods(Static)
        
        % 加载网络的权重
        function myNet = net_load_weight(makeNet, lgraph_layer_names, net_struct, weight_file)
            if isempty(weight_file) % 如果没有预处理权重
                myNet = makeNet; % 使用初始权重，在make_network中设置
                return;
            end
            
            cutoff=0; %初始化值
            if contains(weight_file,'darknet')  % 判断权重是否为主干网络
                split_str = split(weight_file,'.');
                cutoff = str2double(split_str{end}) + 1; % 切断权重加载的位置
            end
            
            fid = fopen(weight_file,'rb');
            if fid<0
                disp('can not open weight file!');
                myNet = makeNet;
                return;
            end
            
            % 加载权重文件的前几个数据
            head = fread(fid,3,'int'); % 根据前3个值判断读取第4个值的字节长度
            if (head(1)*10 + head(2))>=2
                iseen = fread(fid,2,'int');
            else
                iseen = fread(fid,1,'int');
            end
            fprintf('The number of training image: %d\n',iseen(1));
            
            % 加载conv与batch_norm的权重
            layer_names = net_struct.layer_name;
            for k = 1:length(layer_names)
                if k==cutoff
                    break;
                end
                
                if contains(layer_names{k},'conv') % 卷积层处理
                    idx = lgraph_layer_names{k}{1};
                    conv = makeNet.Layers(idx);
                    conv_struct = net_struct.(layer_names{k});
                    
                    % 如果存在batch_norm,则加载batch_norm的权重值
                    if isfield(conv_struct,'batch_normalize') && conv_struct.batch_normalize == 1                     
                        batch_norm = makeNet.Layers(idx+1);
                        % 加载batchnorm层的参数  
                        batch_norm.Offset = reshape(fread(fid,conv_struct.filters,'float'),1,1,conv_struct.filters);
                        batch_norm.Scale = reshape(fread(fid,conv_struct.filters,'float'),1,1,conv_struct.filters);
                        batch_norm.TrainedMean = reshape(fread(fid,conv_struct.filters,'float'),1,1,conv_struct.filters);
                        data = fread(fid,conv_struct.filters,'float'); % abs防止接近0的数为负数，方差不为负数
                        batch_norm.TrainedVariance = reshape(abs(data),1,1,conv_struct.filters);
                        makeNet = replaceLayer(makeNet,batch_norm.Name,batch_norm); % 赋值的batch_norm放回去
                    else
                        conv.Bias = reshape(fread(fid,conv_struct.filters,'float'),1,1,conv_struct.filters);  % linear的conv
                    end
                    
                    try
                        groups = conv.NumGroups;
                    catch
                        groups = 1; % 不是分组卷积
                    end
                    
                    % 加载卷积层的权重,考虑卷积核分组情况
                    if groups==1
                        c = lgraph_layer_names{k}{2};
                        filters = conv.NumFilters;
                        num = conv.FilterSize(1)*conv.FilterSize(2)*c*conv.NumFilters;
                        data = fread(fid,num,'float');
                        weight = reshape(data,conv.FilterSize(1),conv.FilterSize(2),c,filters);
                        conv.Weights = permute(weight,[2,1,3,4]); % matlab按照每列填充3x3
                    else
                        NumChannelPerGroup = lgraph_layer_names{k}{2}/conv.NumGroups;
                        num = conv.FilterSize(1)*conv.FilterSize(2)*NumChannelPerGroup*conv.NumFiltersPerGroup*conv.NumGroups;
                        data = fread(fid, num, 'float');
                        weight = reshape(data,conv.FilterSize(1),conv.FilterSize(2),NumChannelPerGroup,conv.NumFiltersPerGroup,conv.NumGroups);
                        conv.Weights = permute(weight,[2,1,3,4,5]);
                    end
 
                    makeNet = replaceLayer(makeNet, conv.Name, conv);
                end
            end
            fclose(fid);
            myNet = makeNet;
            fprintf('load weights success!\n');
        end
        
        
        % 创建网络
        function [myNet,lgraph_layer_names,output_idxs] = make_network(net_struct)
            import train_pack.*
            
            % 创建网络的输入层
            net_info = net_struct.net;
            lgraph = layerGraph();
            input_size = [net_info.height,net_info.width,net_info.channels];
            input_layer = imageInputLayer(input_size,'Normalization','none','Name','Input');
            lgraph = addLayers(lgraph,input_layer);
            lastName = input_layer.Name; % 添加的网络最后一层名称
            currentChannel = input_size(3); 
            layerChannels = [];
            output_idxs = [];  % 记录输出层位置
            
            % 其他层的创建
            lgraph_layer_names = {}; % 加载的层(卷积层由conv、batch_norm与leaky组成)
            layer_names = net_struct.layer_name;
            layer_idx = 1; % 记录位置
            for k = 1:length(layer_names)
                
                layer_idx = layer_idx + 1; % 记录cfg层在lgraph中的位置
                if contains(layer_names{k},'conv') % 卷积层处理
                    bn_layer=[];activation_layer=[];
                    conv_struct = net_struct.(layer_names{k});
                    k_size = conv_struct.size; stride = conv_struct.stride;
                    pad = conv_struct.pad; filters = conv_struct.filters;
                    if stride==1
                        pad = 'same';
                    end
                    % 添加卷积层，是否存在分组情况
                    if isfield(conv_struct,'groups') 
                        groups = conv_struct.groups;
                        % channels_per_group = nextLayerChannel / groups;
                        filters_per_group = filters / groups;
                        % WeightsInitializer初始化权重
                        conv_layer = groupedConvolution2dLayer(k_size, filters_per_group, groups, 'Name',['conv_group',num2str(k)],...
                                                                'Stride',stride,'Padding',pad, 'WeightsInitializer','he');  % weight_init @(sz) normrnd(0,0.02,sz)
                    else
                        conv_layer = convolution2dLayer(k_size,filters,'Name',['conv_',num2str(k)],'Stride',stride,'Padding',pad,...
                                                               'NumChannels',currentChannel,'WeightsInitializer','glorot'); % @(sz) normrnd(0,0.02,sz) glorot he
                    end
                    
                    % 添加batch_norm，ScaleInitializer初始化权重
                    if isfield(conv_struct,'batch_normalize') && conv_struct.batch_normalize==1
                        bn_layer = batchNormalizationLayer('Name',['batch_norm',num2str(k)],'ScaleInitializer','ones'); % ScaleInitializer @(sz) normrnd(1,0.02,sz)
                    end
                    
                    % 添加激活函数
                    if strcmp(conv_struct.activation,'leaky')
                        activation_layer = leakyReluLayer(0.1,'Name',['leaky_',num2str(k)]);
                    elseif strcmp(conv_struct.activation,'mish')
                        activation_layer = mishLayer(['mish_',num2str(k)]);
                    elseif strcmp(conv_struct.activation,'relu')
                        activation_layer = reluLayer('Name',['relu_',num2str(k)]);
                    end
                    % 待添加的网络层
                    lgraph = addLayers(lgraph,[conv_layer; bn_layer; activation_layer]);
                    lgraph = connectLayers(lgraph, lastName, conv_layer.Name);
                    % 记录该层添加网络层的名称
                    conv_names = {layer_idx, currentChannel, conv_layer.Name}; % 第一个元素存储conv在网络中的位置
                    if ~isempty(bn_layer)
                        conv_names{end+1} = bn_layer.Name;
                        layer_idx = layer_idx + 1;
                    end
                    if ~isempty(activation_layer)
                        conv_names{end+1} = activation_layer.Name;
                        layer_idx = layer_idx + 1;
                    end
                    lgraph_layer_names{k} = conv_names;
                    lastName = conv_names{end}; 
%                     lastChannel = conv_layer.NumFilters; % 添加层后，记录最后一层的名称与输出通道
                    
                    currentChannel = filters; % 更新当前层的通道数
                    
                elseif contains(layer_names{k},'maxpool') % 最大池化层处理
                    maxpool_struct = net_struct.(layer_names{k});
                    stride = maxpool_struct.stride;
                    if stride==1
                        pad = 'same';
                        maxpool_layer = maxPooling2dLayer(maxpool_struct.size,'Padding',pad,'Stride',stride,'Name',['maxpool_',num2str(k)]);
                    else
                        maxpool_layer = maxPooling2dLayer(maxpool_struct.size,'Stride',stride,'Name',['maxpool_',num2str(k)]);
                    end
                    
                    lgraph = addLayers(lgraph,maxpool_layer);
                    lgraph = connectLayers(lgraph, lastName, maxpool_layer.Name);
                    lgraph_layer_names{k} = {layer_idx, maxpool_layer.Name};
                    lastName = maxpool_layer.Name; 
                    
                elseif contains(layer_names{k},'reorg') % yolov2的reorg层
                    reorg_struct = net_struct.(layer_names{k});
                    reorg_layer = reorgLayer(['reorg_',num2str(k)],reorg_struct.stride);
                    % YOLOv2中的reorgLayer对应Matlab中的spaceToDepthLayer
%                     reorg_layer = spaceToDepthLayer([reorg_struct.stride reorg_struct.stride],'Name',['reorg_',num2str(k)]);  % spaceToDepthLayer  yolov2ReorgLayer
                    lgraph = addLayers(lgraph,reorg_layer);
                    lgraph = connectLayers(lgraph, lastName, reorg_layer.Name);
                    lgraph_layer_names{k} = {layer_idx, reorg_layer.Name};
                    currentChannel = currentChannel*reorg_struct.stride^2;
                    lastName = reorg_layer.Name;
                    
                elseif contains(layer_names{k},'region') % yolov2之前的特征输出层
                    yolo_struct = net_struct.(layer_names{k});
                    classes = yolo_struct.classes; % 能够检测目标的类别数目
                    anchors = str2num(yolo_struct.anchors);
                    anchors = reshape(anchors,2,[])';
                    net_info = net_struct.net;
                    input_size = [net_info.width, net_info.height, net_info.channels]; % 赋值输入网络的图像尺寸
                    
                    region_layer = yoloLayer(['yolo_',num2str(k)],'V2',anchors,classes);
                    region_layer.input_size = input_size;
                    lgraph = addLayers(lgraph,region_layer);
                    lgraph = connectLayers(lgraph, lastName, region_layer.Name);
                    lgraph_layer_names{k} = {layer_idx, region_layer.Name};
                    lastName = region_layer.Name;
                    output_idxs = [output_idxs,length(lgraph.Layers)];
                    
                elseif contains(layer_names{k},'shortcut')
                    shortcut = net_struct.(layer_names{k});
                    shortcut_layer = additionLayer(2,'Name',['shortcut_',num2str(k)]);
                    lgraph = addLayers(lgraph,shortcut_layer);
                    from = shortcut.from;
                    lgraph = connectLayers(lgraph, lgraph_layer_names{k+from}{end},[shortcut_layer.Name,'/in1']);  % 连接from的位置
                    lgraph = connectLayers(lgraph, lastName, [shortcut_layer.Name,'/in2']);
                    lgraph_layer_names{k} = {layer_idx, shortcut_layer.Name};
                    lastName = shortcut_layer.Name;
                    
                elseif contains(layer_names{k},'dropout')
                    dropout = net_struct.(layer_names{k});
                    dl = dropoutLayer(dropout.probability,'Name',['dropout_',num2str(k)]);
                    lgraph = addLayers(lgraph,dl);
                    lgraph = connectLayers(lgraph,lastName, dl.Name);
                    lgraph_layer_names{k} = {layer_idx, dl.Name};
                    lastName = dl.Name;
                    
                 elseif contains(layer_names{k},'upsample')
                    upsample = net_struct.(layer_names{k});
                    ul = upsampleLayer(['upsample_',num2str(k)],upsample.stride);
                    lgraph = addLayers(lgraph,ul);
                    lgraph = connectLayers(lgraph,lastName,ul.Name);
                    lgraph_layer_names{k} = {layer_idx, ul.Name};
                    lastName = ul.Name;
                    
                elseif contains(layer_names{k},'route') % route时上一层的channel会改变，并不是上一层conv的filters
                    route = net_struct.(layer_names{k});
                    route_idx = str2num(route.layers);
                    if length(route_idx)==1 % 一层路由
                        rl = route1Layer(['route_',num2str(k)],route_idx);
                        if isfield(route,'groups') % 存在group的情况
                            rl.groups = route.groups;
                            rl.group_id = route.group_id + 1;
                        end
                        lgraph = addLayers(lgraph,rl);
                        if route_idx<0
                            lgraph = connectLayers(lgraph,lgraph_layer_names{k+route_idx}{end},rl.Name);
                            currentChannel = layerChannels(k+route_idx);
                        else
                            % matlab数组从1开始计数
                            lgraph = connectLayers(lgraph,lgraph_layer_names{route_idx+1}{end},rl.Name); 
                            currentChannel = layerChannels(route_idx+1);
                        end
                        if isfield(route,'groups') % 下一层的通道数
                            currentChannel = (currentChannel/route.groups)*length(route.group_id);
                        end
                    else % 存在两层路由 
                        rl = depthConcatenationLayer(length(route_idx),'Name',['route_',num2str(k)]);
                        lgraph = addLayers(lgraph,rl);
                        currentChannel = 0; % 更新上一层输出的通道数
                        for n=1:length(route_idx)
                            if route_idx(n)<0
                                lgraph = connectLayers(lgraph, lgraph_layer_names{k+route_idx(n)}{end},[rl.Name,'/in',num2str(n)]);
                                currentChannel = currentChannel + layerChannels(k+route_idx(n));
                            else
                                lgraph = connectLayers(lgraph,lgraph_layer_names{route_idx(n)+1}{end},[rl.Name,'/in',num2str(n)]); % route_idx(n)+1, matlab是从1开始的
                                currentChannel = currentChannel + layerChannels(route_idx(n)+1);
                            end
                        end
                    end
                    lgraph_layer_names{k} = {layer_idx, rl.Name};
                    lastName = rl.Name;
    
                elseif contains(layer_names{k},'yolo') % yolo输出
                    yolo_struct = net_struct.(layer_names{k});
                    net_info = net_struct.net;
                    input_size = [net_info.width, net_info.height, net_info.channels]; % 赋值输入网络的图像尺寸
                    classes = yolo_struct.classes; % 能够检测目标的类别数目
                    anchors = str2num(yolo_struct.anchors);
                    anchors = reshape(anchors,2,[])';
                    anchors = anchors(str2num(yolo_struct.mask)+1,:);
                    
                    yolo_layer = yoloLayer(['yolo_',num2str(k)],'V3|V4',anchors,classes);
                    yolo_layer.input_size = input_size;
                    lgraph = addLayers(lgraph,yolo_layer);
                    lgraph = connectLayers(lgraph,lastName,yolo_layer.Name);
                    lgraph_layer_names{k} = {layer_idx, yolo_layer.Name};
                    lastName = yolo_layer.Name;
                    output_idxs = [output_idxs,length(lgraph.Layers)];
                    
                end
                layerChannels(k) = currentChannel; % 记录每一层的输出通道
            end
            layerChannels = layerChannels';
            lgraph_layer_names = lgraph_layer_names';
            myNet = lgraph;
            fprintf('load net success!\n');
        end
    end
end