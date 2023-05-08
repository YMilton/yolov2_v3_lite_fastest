
% matlab的卷积(带GPU的): toolbox\nnet\deep\@gpuArray\internal_dlconv.m
% X = nnet.internal.cnngpu.convolveForwardND(X, weights, paddingStart, [], stride, dilation, numGroups) + bias;
% 函数dlconv

% 卷积层类
classdef conv_layer
    properties(Access=public)
        name='conv';
        input % 输入[rows,cols,c,b]
        groups=1 % 分组参数，默认为1
        filters % 输出通道
        k_size = 3 % 卷积核大小
        stride=1 % 卷积核步长
        
        padding = 1
        kernels % 内核参数
        bias % 偏置项
        activate_fun = 'linear' % 激活函数
        batch_norm  % batch_norm对象
        output % 输出
        
        dkernels % 内核梯度计算
        dbias
        
        delta % 当前层梯度 [rows,cols,anchor_size*(5+clasess),batch_size]
        update_delta % 更新的梯度，供上一层使用 [rows,cols,anchor_size*(5+clasess),batch_size]
    end
    
    methods(Access=public)
        % 构造函数 conv_layer(input, filters, k_size, stride) or conv_layer()
        function this = conv_layer(varargin)
            if nargin==0
                return;
            end
            this.input = varargin{1};
            this.filters = varargin{2};
            this.k_size = varargin{3};
            this.padding = floor(this.k_size/2);
            this.stride = varargin{4};
            
            % 初始化权重与偏置
            if ~isempty(this.input)  
                this.kernels = normrnd(0, 0.02, [this.k_size, this.k_size, size(this.input,3)/this.groups, this.filters]);
                this.bias = zeros(1,1,this.filters);
            end
        end
        
        
        % 3维矩阵的卷积,input:[rows,cols,3],kernel:[s,s,3]
        function obj = forward(this)
            import layers.conv_layer.convolve3d
            import layers.conv_layer.convolve_matlab
            % 单幅图像卷积
            if length(size(this.input))<=3
                % 融合batch_norm到conv
                this = this.fuse_conv_batchnorm; % weight与bias为融合batch_norm层的结果
                this.output = convolve3d(this.input,this.padding,this.stride,this.groups,this.kernels,this.bias); %自定义卷积              
                
                % (input,padding,stride,weight,bias)
                % this.output = convolve_matlab(this.input,this.padding,this.stride,this.groups,this.kernels,this.bias);
            end
            % 多幅图像卷积(训练时) size(this.input)=[rows,cols,filters,batch]
            if length(size(this.input))==4
                batch = size(this.input,4);
                padding_ = this.padding; bias_=this.bias;stride_ = this.stride;
                groups_ = this.groups;kernels_ = this.kernels;input_ = this.input;
                parfor b=1:batch % 并行运算
                    output_(:,:,:,b) = convolve3d(input_(:,:,:,b),padding_,stride_,groups_,kernels_,bias_);
                end
                if ~isempty(this.batch_norm) % 存储在batch_norm
                    bn = this.batch_norm;
                    bn.mode = 'train'; % 设置为训练模式
                    bn.input = output_;
                    bn = bn.forward;
                    this.batch_norm = bn;
                    this.output = output_;
                else
                    this.output = output_ + reshape(this.bias,1,1,this.filters,1); % size(output_)=[rows,cols,filters,batch_size]
                end
            end

            % 激活函数
            import layers.activation_layer
            if contains(this.activate_fun,'leaky')
                this.output = activation_layer.leaky(this.output);
            end
            
            obj = this;
            
%             [h,w,c,b] = size(obj.output);
%             fprintf('conv_layer complete! conv_size [w,h,c,b]: [%d,%d,%d,%d]\n', w,h,c,b);

        end
        
        
        % 卷积层的反向传播
        function obj = backward(this)
            import layers.conv_layer.single_dkernel_convolve
            import layers.activation_layer
            
            % 激活函数反向传播
            if strcmp(this.activate_fun,'linear')
                this.dbias = sum(this.delta,[1,2,4]);
                this.bias = this.bias - this.dbias;
            end
            if strcmp(this.activate_fun,'leaky')
                this.delta = activation_layer.leaky_backward(this.delta,this.output);
            end
            
            % 卷积的反向传播
            delta_batch = this.delta; input_batch = this.input;
            kernels_ = this.kernels; stride_ = this.stride;pad_ = this.padding;
            for b=1:size(this.delta,4) % batch图像个数
                d = delta_batch(:,:,:,b); % 单幅图像的delta,size=[rows,cols,filters]
                X = input_batch(:,:,:,b); % 单幅图像输入xx,size=[rows,cols,channels]
                [dkernels_,dX_] = single_dkernel_convolve(X,d,kernels_,stride_,pad_);
                dkernels_batch(:,:,:,:,b)=dkernels_;
                dX_batch(:,:,:,b) = dX_;
            end
            this.dkernels = mean(dkernels_batch,5);
            this.kernels = this.kernels - this.dkernels; % 更新权重,梯度下降
            this.update_delta = dX_batch;
            
            obj = this;
        end
        
        
        % 融合batch_norm、激活函数到conv中
        function obj=fuse_conv_batchnorm(this)
            bn = this.batch_norm; % 传入的batch_norm
            obj = this;
            eps = 1e-5;
            if ~isempty(bn)
                % conv层的weights融合,矩阵运算
                bn_w = bn.gamma./sqrt(bn.rolling_var + eps);
                tmp = reshape(this.kernels,[],this.filters).*reshape(bn_w,[],this.filters);
                obj.kernels = reshape(tmp,size(this.kernels));                
                % conv层bias融合
                obj.bias = bn.beta - bn.rolling_mean.*bn.gamma./sqrt(bn.rolling_var + eps);
            end
        end
    end
    

    methods(Static)
        % 单张图像的卷积
        function output = convolve3d(input,padding,stride,groups,kernels,bias) 
            [rows,cols,channels] = size(input);
                
            % 保证截取kernal大小矩阵不溢出
            k_size = size(kernels,1);
            if padding<floor(k_size/2)
                padding = floor(k_size/2);
            end
            % 四周用0填充，填充数目padding
            img_pad = zeros(rows+2*padding,cols+2*padding,channels);
            img_pad(padding+1:padding+rows,padding+1:padding+cols,:) = input;
            
            % 索引meshgrid生成
            [idx_y,idx_x] = meshgrid(1:stride:rows,1:stride:cols);
            y_center = idx_y + padding; x_center = idx_x + padding;
            
            step = floor(k_size/2);
            nums = size(idx_y,1)*size(idx_y,2);
            for idx = 1:nums
                kernel_pixels = img_pad(y_center(idx) - step:y_center(idx) + step,...
                                        x_center(idx) - step:x_center(idx) + step,:);
                                    
                if groups>1 % 存在分组卷积的情况
                    [rows_k,cols_k,c_k] = size(kernel_pixels);
                    M_dot = reshape(kernel_pixels,rows_k,cols_k,c_k/groups,groups).*kernels;
                else
                    M_dot = kernel_pixels.*kernels;
                end
                sum_M_dot = sum(sum(sum(M_dot,1),2),3); 
                % sum_M_dot = sum(M_dot,[1,2,3]); % 前面三个维度之和
                sum_M_dot_ = sum_M_dot(:,:,:);  % 4维矩阵转换成3维矩阵
                
                if stride==1
                    output(idx_y(idx), idx_x(idx),:) = sum_M_dot_;
                else
                    output((idx_y(idx) + stride - 1)/stride,...
                                (idx_x(idx) + stride - 1)/stride,:) = sum_M_dot_;
                end
            end
            
            output = output + bias;
        end
        
        function output = convolve_matlab(input,padding,stride,groups,weight,bias)
            if groups>1 % 分组的情况
                [k_h,k_w,k_c_grouped,k_f] = size(weight); 
                weight = reshape(weight,k_h,k_w,k_c_grouped,k_f/groups,groups);
                bias = reshape(bias,k_f,1);
            end
            
            if canUseGPU % GPU
                dlX = dlarray(gpuArray(input),'SSC');
                weight = dlarray(gpuArray(weight));
                bias = dlarray(gpuArray(bias));
            else % CPU
                dlX = dlarray(input,'SSC');
                weight = dlarray(weight);
            end
            if size(weight,1)==1
                padding = 0;
            end
            out = dlconv(dlX,weight,bias,'Stride',stride,'Padding',padding); % 卷积操作
            output = extractdata(gather(out));
        end
        
        %============================================================================%
        function output = convolve2d(input,kernels,padding)
            % stride默认为1
            if length(size(input))==3 % dkernels
                [rows,cols,channels] = size(input);

                % 四周用0填充，填充数目padding
                img_pad = zeros(rows+2*padding,cols+2*padding,channels,'single');
                img_pad(padding+1:padding+rows,padding+1:padding+cols,:) = input;

                % 索引meshgrid生成
                k_size = size(kernels,1);
                [idx_y,idx_x] = meshgrid(k_size:rows+2*padding,k_size:cols+2*padding);

                nums = size(idx_y,1)*size(idx_y,2);
                for idx = 1:nums
                    kernel_pixels = img_pad(idx_y(idx) - k_size + 1:idx_y(idx), idx_x(idx) - k_size + 1:idx_x(idx),:);
                    M_dot = kernel_pixels.*kernels;
                    sum_M_dot = sum(sum(M_dot,1),2);
                    % sum_M_dot = sum(M_dot,[1,2]); % 前面三个维度之和
                    output(idx_y(idx)-k_size+1, idx_x(idx)-k_size+1,:,:) = sum_M_dot; 
                end 
            end
            
            if length(size(input))==4 % dX
                [rows,cols,channels,filters] = size(input);

                % 四周用0填充，填充数目padding
                img_pad = zeros(rows+2*padding,cols+2*padding,channels,filters,'single');
                img_pad(padding+1:padding+rows,padding+1:padding+cols,:,:) = input;

                % 索引meshgrid生成
                k_size = size(kernels,1);
                [idx_y,idx_x] = meshgrid(k_size:rows+2*padding,k_size:cols+2*padding);

                nums = size(idx_y,1)*size(idx_y,2);
                for idx = 1:nums
                    kernel_pixels = img_pad(idx_y(idx) - k_size + 1:idx_y(idx), idx_x(idx) - k_size + 1:idx_x(idx),:,:);
                    M_dot = kernel_pixels.*kernels;
                    sum_M_dot = sum(M_dot,[1,2,4]); % 前面三个维度之和
                    output(idx_y(idx)-k_size+1, idx_x(idx)-k_size+1,:) = sum_M_dot; 
                end 
            end
        end
        
        % 向矩阵中间添加0元素，row_z,col_z行与行、列与列之间添加几行或几列0元素
        function output=insert_zeros(mat,row_z,col_z)
            [rows,cols,channels,filters] = size(mat);
            new_rows = rows+(rows-1)*row_z; new_cols = cols+(cols-1)*col_z;
            output = zeros(new_rows,new_cols,channels,filters);
            rows_range = 1:row_z+1:new_rows; cols_range = 1:col_z+1:new_cols;
            output(rows_range,cols_range,:,:) = mat;
        end
        
        % 卷积核与输入求导 X:[rows1,cols1,channels], delta:[rows2,cols2,filters]
        function [dkernels,dX] = single_dkernel_convolve(X,delta,kernels,stride,pad)
            import layers.conv_layer.convolve2d
            import layers.conv_layer.insert_zeros
            import layers.conv_layer.calculate_pad
            
            % delta在chnnels的维度上求和，则反向传播需要复制为[rows2,cols2,channels,filters]
            channels = size(X,3); [rows2,cols2,filters]=size(delta);
            copy_channels = repmat(delta,1,channels,1);
            copy_delta = reshape(copy_channels,rows2,cols2,channels,filters);
            
            k_size = size(kernels,1);
             % 保证截取kernal大小矩阵不溢出
            if pad<floor(k_size/2)
                pad = floor(k_size/2);
            end
            if k_size == 1
                dkernels = sum(X.*copy_delta,[1,2]); % 卷积核求导
                dX = sum(copy_delta.*kernels,4); % 输入求导
            else
                % k_size>1则表示全卷积操作,kernels前两维需要旋转180度
                rot_kernels180 = rot90(kernels,2);
                if stride==1
                    pad = calculate_pad(size(X,1),size(copy_delta,1),size(kernels,1));
                    dkernels = convolve2d(X,copy_delta,pad);
                    % dkernels = dlconv(dlarray(single(X),'SSC'),dlarray(single(copy_delta)),0,'Padding',pad);
                    
                    pad = calculate_pad(size(copy_delta,1),size(rot_kernels180,1),size(X,1));
                    dX = convolve2d(copy_delta,rot_kernels180,pad); % 全卷积操作
                    % dX = dlconv(dlarray(single(copy_delta),'SSC'),dlarray(single(rot_kernels180)),0,'Padding',pad);
                else
                    copy_delta = insert_zeros(copy_delta,stride-1,stride-1); % 插入0元素
                    dkernels = convolve2d(X,copy_delta,pad);
                    dX = convolve2d(copy_delta,rot_kernels180,k_size-1);
                    
%                     dkernels = dlconv(dlarray(single(X),'SSC'),dlarray(single(copy_delta)),0,'Padding',pad);
%                     dX = dlconv(dlarray(single(copy_delta),'SSC'),dlarray(single(rot_kernels180)),0,'Padding',k_size-1);
                end
            end
        end
        
        % 计算pad，防止报错
        function pad = calculate_pad(in_size,k_size,out_size)
            pad = 0.5*(out_size - 1 + k_size - in_size);
        end
        
    end
    
    
end


