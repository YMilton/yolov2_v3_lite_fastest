% 网络结构与权重加载
classdef load_net
    
    properties(Access=public)
        cfg_file
        weight_file
       
        net_struct
        mynet
    end    
    
    methods(Access=public)
        
        % 输入格式：load_net(cfg_file,weight_file) or load_net(cfg_file):创建权重
        function obj = load_net(varargin)
            if nargin>2 || nargin<=0
                if nargin==0 % 单纯创建类
                    return;
                end
                disp('Please input right parameters(load_net)!');
                return;
            end
            
            import layers.* % 添加网络的不同层

            obj.cfg_file = varargin{1};
            if nargin==2 % 存在权重文件时加载
                obj.weight_file = varargin{2};
            end
            obj.net_struct = read_file2struct(obj); % 转换cfg为struct

            % 读取权重
            obj.mynet = net_load_weights(obj);
            
        end
           
        
        % 读取cfg文件，并转化成struct
        function net_struct=read_file2struct(this)
            fid = fopen(this.cfg_file);
            if fid<0
                disp('can not open cfg file!');
                net_struct = [];
                return;
            else
                i = 1; layer_names = {}; layer_names_no_num = {};
                while 1
                    tline = fgetl(fid);
                    if ~ischar(tline)
                        fclose(fid);
                        break;
                    end
                    
                    if contains(tline,'[')
                        if contains(tline,'net')
                            name = strrep(tline(2:end-1),' ',''); % 删除字符串所有空格
                        else
                            name_without_num = strrep(tline(2:end-1),' ','');
                            layer_names_no_num(end+1) = {name_without_num};
                            name = strcat(name_without_num,num2str(i));
                            layer_names(end+1) = {name};
                            i = i + 1;
                        end
                    else
                        if  ~contains(tline,'#') && contains(tline,'=')
                            key_val = split(tline,'=');
                            % 等号后面有逗号或属性为layers，保留字符串
                            if contains(key_val{2},',') || strcmp(strtrim(key_val{1}),'layers')
                                net_struct.(name).(strtrim(key_val{1})) = strtrim(key_val{2});
                                continue;
                            end
                            
                            val = str2double(key_val{2});
                            if isnan(val) % 无法转换成数字
                                val = strtrim(key_val{2}); % 去除左右空格
                            end
                            net_struct.(name).(strtrim(key_val{1})) = val;            
                        end
                    end
                end
                net_struct.layer_name = layer_names';
            end
        end
        
        
        % 网络加载权重包括：训练后的权重、主干网络权重(darknet19、darknet53)
        function mynet = net_load_weights(this)
            import layers.batch_norm_layer

            mynet = [];
            if isempty(this.net_struct)
                disp('read cfg file first!');
                return;
            end
            
            % 构建网络并初始化conv与batch_normd权重与偏置
            mynet = load_net.make_network(this.net_struct); 
            if isempty(this.weight_file) % 如果没有预处理权重
                return;
            end
            cutoff=0; %初始化值
            if contains(this.weight_file,'darknet')  % 判断权重是否为主干网络
                split_str = split(this.weight_file,'.');
                cutoff = str2double(split_str{end}) + 1; % 切断权重加载的位置
            end
            
            fid = fopen(this.weight_file,'rb');
            if fid<0
                disp('can not open weight file!');
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
            layer_names = this.net_struct.layer_name;
            for k = 1:length(layer_names)
%                 if k==36
%                     disp(layer_names{k});
%                 end
                if k==cutoff
                    break;
                end
                if contains(layer_names{k},'conv') % 卷积层处理
                    conv = mynet{k};
                    conv_struct = this.net_struct.(layer_names{k});
                    
                    % 如果存在batch_norm,则加载batch_norm的权重值
                    if isfield(conv_struct,'batch_normalize') && conv_struct.batch_normalize == 1                     
                        batch_norm = batch_norm_layer(conv.filters);
                        % 加载batchnorm层的参数  
                        batch_norm.beta = reshape(fread(fid,conv.filters,'float'),1,1,conv.filters);
                        batch_norm.gamma = reshape(fread(fid,conv.filters,'float'),1,1,conv.filters);
                        batch_norm.rolling_mean = reshape(fread(fid,conv.filters,'float'),1,1,conv.filters);
                        batch_norm.rolling_var = reshape(fread(fid,conv.filters,'float'),1,1,conv.filters);
                        conv.batch_norm = batch_norm; % 添加batch_norm层到卷积层中
                    else
                        conv.bias = reshape(fread(fid,conv.filters,'float'),1,1,conv.filters);  % linear的conv
                    end
                    
                    % 加载卷积层的权重,考虑卷积核分组情况
                    c = size(conv.input,3)/conv.groups;
                    num = conv.k_size*conv.k_size*c*conv.filters;
                    data = fread(fid,num,'float');
                    weight = reshape(data,conv.k_size,conv.k_size,c,conv.filters);
                    conv.kernels = permute(weight,[2,1,3,4]); % matlab按照每列填充3x3
                    
                    %conv = conv.fuse_conv_batchnorm; % 融合batch_norm到conv
                    
                    mynet{k} = conv;
                end
            end
            fclose(fid);
            fprintf('load weights success!\n\n');
        end
        
    end
    
    
    methods(Static)
        
        % 把matlab的网络转换成自定义网络,赋值相关参数
        function mynet = create_net_matlab(matlab_net_layers)
            import layers.*
            
            mynet = {}; % 元胞数组初始化
            net_layers = matlab_net_layers;
            for k=1:length(net_layers)
                if contains(net_layers(k).Name,'convolution')
                    CL = conv_layer([],net_layers(k).NumFilters,net_layers(k).FilterSize(1),net_layers(k).Stride(1));
                    CL.kernels = net_layers(k).Weights;
                    CL.bias = net_layers(k).Bias;
                    mynet(k,1) = {CL};
                    % 合并conv、batchnorm、activation
                    if k~=length(net_layers)
                        if contains(net_layers(k+1).Name,'batchnorm')
                            % batch_norm创建
                            BN = batch_norm_layer;
                            BN.rolling_mean = net_layers(k+1).TrainedMean;
                            BN.rolling_var = net_layers(k+1).TrainedVariance;
                            BN.gamma = net_layers(k+1).Scale;
                            BN.beta = net_layers(k+1).Offset;
                            
                            CL.batch_norm = BN;
                        end
                        
                        if contains(net_layers(k+2).Name,'activation')
                            CL.activate_fun = 'leaky';
                            conv = CL.fuse_conv_batchnorm;
                            mynet(k,1) = {conv};
                        end
                    end
                
                elseif contains(net_layers(k).Name,'pooling')
                    MPL = maxpool_layer([],net_layers(k).PoolSize(1),net_layers(k).Stride(1));
                    mynet(k,1) = {MPL};
                else
                    mynet(k,1) = {[]};
                end
            end
            
            mynet(cellfun(@isempty,mynet)) = []; % 删除为空的元胞
            
            % 添加yolo_layer层
            yolo = yolo_layer();
            yolo.version='v2';
            yolo.classes = 20; % 能够检测目标的类别数目（VOC)
            yolo.anchors = [1.08,1.19; 3.42,4.41; 6.63,11.38; 9.42,5.11; 16.62,10.52];
            mynet{end+1,1} = yolo;
            
            fprintf('load net success!\n\n');
        end
        
        
        % 创建初始化权重的神经网络
        function mynet=make_network(net_struct)
            import layers.*
            
            % 遍历net_struct
            layer_names = net_struct.layer_name;
            net_info = net_struct.net;
            input = zeros(net_info.height,net_info.width,net_info.channels);
            mynet={};
            
            for k = 1:length(layer_names)
%                 if 7==7
%                     fprintf('%d, %s\n',k, layer_names{k});
%                 end
                if contains(layer_names{k},'conv') % 卷积层处理
                    conv_struct = net_struct.(layer_names{k});
                    conv = conv_layer(input,conv_struct.filters,conv_struct.size,conv_struct.stride);
                    % 设置卷积的output尺寸
                    conv.output = zeros(size(input,1)/conv_struct.stride, size(input,2)/conv_struct.stride, conv_struct.filters);
                    conv.padding = conv_struct.pad;
                    conv.activate_fun = conv_struct.activation;
                    if isfield(conv_struct,'groups')
                        conv.groups = conv_struct.groups;
                    end
                    % 设置batch_norm
                    if isfield(conv_struct,'batch_normalize') && conv_struct.batch_normalize==1
                        bn = batch_norm_layer(conv_struct.filters);
                        conv.batch_norm = bn;
                    end
                    
                    mynet(end+1,1) = {conv};

                elseif contains(layer_names{k},'maxpool') % 最大池化层处理
                    maxpool_struct = net_struct.(layer_names{k});
                    maxpool = maxpool_layer(input,maxpool_struct.size,maxpool_struct.stride);
                    maxpool.output = zeros(size(input,1)/maxpool_struct.stride, size(input,2)/maxpool_struct.stride, size(input,3));
                    mynet{end+1,1} = maxpool;
                    
                elseif contains(layer_names{k},'reorg') % yolov2的reorg层
                    reorg = reorg_layer();
                    reorg_struct = net_struct.(layer_names{k});
                    reorg.stride = reorg_struct.stride;
                    reorg.input = input;
                    reorg.output = zeros(size(input,1)/reorg.stride,size(input,2)/reorg.stride,reorg.stride^2*size(input,3));
                    mynet{end+1,1} = reorg;
                    
                elseif contains(layer_names{k},'region') % yolov2之前的特征输出层
                    yolo = yolo_layer();
                    yolo.version = 'v2'; % yolov2
                    yolo_struct = net_struct.(layer_names{k});
                    
                    yolo.input_size = [net_struct.net.width, net_struct.net.height]; % 赋值输入网络的图像尺寸
                    yolo.classes = yolo_struct.classes; % 能够检测目标的类别数目
                    anchors = str2num(yolo_struct.anchors);
                    anchors = reshape(anchors,2,[])';
                    yolo.anchors = anchors;
                    mynet{end+1,1} = yolo;
                    
                elseif contains(layer_names{k},'shortcut')
                    shortcut = net_struct.(layer_names{k});
                    sl = shortcut_layer(shortcut.from,shortcut.activation);
                    sl.input = input;
                    sl.output = zeros(size(input));
                    mynet{end+1,1} = sl;
                    
                elseif contains(layer_names{k},'dropout')
                    dropout = net_struct.(layer_names{k});
                    dl = dropout_layer(dropout.probability);
                    dl.mode = 'test';
                    dl.input = input;
                    dl.output = zeros(size(input));
                    mynet{end+1,1} = dl;
                    
                 elseif contains(layer_names{k},'upsample')
                    upsample = net_struct.(layer_names{k});
                    ul = upsample_layer(upsample.stride);
                    ul.input = input;
                    ul.output = zeros(upsample.stride*size(input,1),upsample.stride*size(input,2),size(input,3));
                    mynet{end+1,1} = ul;    
                    
                elseif contains(layer_names{k},'route') % route时上一层的channel会改变，并不是上一层conv的filters
                    route = net_struct.(layer_names{k});
                    route_idx = str2num(route.layers);
                    
                    if isfield(route,'groups') % route中是否存在分组
                        rl = route_layer(route.layers, route.groups, route.group_id);
                    else
                        rl = route_layer(route.layers);
                    end
                    
                    if length(route_idx)==1 % 一层路由
                        if route_idx<0
                            [rows,cols,c] = size(mynet{k + route_idx,1}.output);
                        else
                            [rows,cols,c] = size(mynet{route_idx,1}.output);
                        end
                        route_filters = c;
                        if rl.groups==2 && rl.group_id==1
                            route_filters = c/2;
                        end
                    else % 存在两层路由 
                        route_filters = 0;
                        for n=1:length(route_idx)
                            if route_idx(n)<0
                                [rows,cols,c] = size(mynet{k + route_idx(n),1}.output);
                                route_filters = route_filters + c;
                            else
                                [rows,cols,c] = size(mynet{route_idx(n)+1,1}.output); % route_idx(n)+1, matlab是从1开始的
                                route_filters = route_filters + c;
                            end
                        end
                    end
                    rl.output = zeros(rows,cols,route_filters); 
                    mynet{end+1,1} = rl;  % 添加route层参数
                    
                elseif contains(layer_names{k},'yolo') % yolo输出
                    yolo_struct = net_struct.(layer_names{k});
                    net_info = net_struct.net;
                    disp(yolo_struct);
                    
                    yolo = yolo_layer();
                    yolo.input_size = [net_info.width, net_info.height]; % 赋值输入网络的图像尺寸
                    yolo.classes = yolo_struct.classes; % 能够检测目标的类别数目
                    anchors = str2num(yolo_struct.anchors);
                    anchors = reshape(anchors,2,[])';
                    yolo.anchors = anchors(str2num(yolo_struct.mask)+1,:);
                    mynet{end+1,1} = yolo;
                else % 其他层
                    % disp(layer_names{k});
                end
                
                input = mynet{end,1}.output; % 更新下一层的输入
            end
            fprintf('load net success!\n\n');
        end
                
        
        % 设置网络是训练还是测试状态
        function new_net=set_status(net, status)
            for k=1:length(net)
                if strcmp(net{k}.name,'yolo')
                    net{k}.status=status;
                end
            end
            new_net = net;
        end
    end
    
end