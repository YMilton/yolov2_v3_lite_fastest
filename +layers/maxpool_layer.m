
% 最大池化层
classdef maxpool_layer
    properties(Access=public)
        name='maxpool';
        input % 输入[rows,cols,c]
        pool_size = 2
        stride = 2
        maxpool_idx %记录最大值位置的矩阵
        output
        
        delta % 反向传播的梯度
        update_delta
    end
    
    methods(Access=public)
        function obj = maxpool_layer(input,pool_size,stride)
            % [rows,cols,channels] = size(input);
            obj.input = input;
            obj.pool_size = pool_size;
            obj.stride = stride;
            % 输出大小计算
            % obj.output = zeros(ceil(rows/2),ceil(cols/2),channels);
        end
        
        % 池化操作
        function obj = forward(this)
            import layers.maxpool_layer.mp_single
            if length(size(this.input))==3 % 单幅图像
                [this.output,this.maxpool_idx] = mp_single(this.input,this.stride,this.pool_size);
            end
            
            if length(size(this.input))==4 % 一个batch的图像
                batch = size(this.input,4);
                input_ = this.input; stride_ = this.stride; pool_size_ = this.pool_size;
                parfor b=1:batch % 并行运算
                    [output_(:,:,:,b),maxpool_idx_(:,:,:,b)] = mp_single(input_(:,:,:,b),stride_,pool_size_);
                end
                this.output = output_;
                this.maxpool_idx = maxpool_idx_;
            end
            
            obj = this;
            
%             [h,w,c,b] = size(obj.output);
%             fprintf('maxpool_layer complete! maxpool_size [w,h,c,b]: [%d,%d,%d,%d]\n', w,h,c,b);
        end
        
        
        % 反向传播
        function obj=backward(this)
            import layers.maxpool_layer.pad_val
            
            [rows,cols,~] = size(this.input);
            update_delta_ = zeros(size(this.input),'single');
            maxpool_idx_ = this.maxpool_idx;
            delta_ = this.delta;
            [idx_y,idx_x] = meshgrid(1:this.stride:rows,1:this.stride:cols);
            
            if this.stride>1
                % 最大池化操作strid>=2
                for i=1:size(idx_y,1)*size(idx_y,2)
                    end_y = min(idx_y(i)+this.pool_size-1,rows);
                    end_x = min(idx_x(i)+this.pool_size-1,cols);
                    % 根据最大值位置填充梯度值
                    in_delta = update_delta_(idx_y(i):end_y,idx_x(i):end_x,:); 
                    max_idx = maxpool_idx_(ceil(end_y/this.stride),ceil(end_x/this.stride),:);
                    out_delta = delta_(ceil(end_y/this.stride),ceil(end_x/this.stride),:);
                    in_delta = pad_val(in_delta,out_delta,max_idx); % 输入梯度的计算
                    update_delta_(idx_y(i):end_y,idx_x(i):end_x,:) = in_delta; 
                end
            else
                % stride=1
                for i=1:size(idx_y,1)*size(idx_y,2)
                    end_y = min(idx_y(i)+this.pool_size-1,rows);
                    end_x = min(idx_x(i)+this.pool_size-1,cols);
                     % 根据最大值位置填充梯度值
                    in_delta = update_delta_(idx_y(i):end_y,idx_x(i):end_x,:); 
                    max_idx = maxpool_idx_(idx_y(i),idx_x(i),:);
                    out_delta = delta_(idx_y(i),idx_x(i),:);
                    in_delta = pad_val(in_delta,out_delta,max_idx);
                    update_delta_(idx_y(i):end_y,idx_x(i):end_x,:) = in_delta; 
                end
            end
            this.update_delta = gather(update_delta_);
            obj = this;
        end
    end
    
    methods(Static)
        % 单幅图像就的最大池化
        function [output,maxpool_idx] = mp_single(input,stride,pool_size)
            % index x,y
            [rows,cols,~] = size(input); 
            [idx_y,idx_x] = meshgrid(1:stride:rows,1:stride:cols);
            
            if stride>1
                % 最大池化操作strid>=2
                for i=1:size(idx_y,1)*size(idx_y,2)
                    end_y = min(idx_y(i)+pool_size-1,rows);
                    end_x = min(idx_x(i)+pool_size-1,cols);
                    pool_pixels = input(idx_y(i):end_y,idx_x(i):end_x,:);
                    % 记录最大值与最大值位置
                    pp = reshape(pool_pixels,[],size(pool_pixels,3));
                    [max_val,max_idx] = max(pp,[],1);
                    output(ceil(end_y/stride),ceil(end_x/stride),:) = max_val;
                    maxpool_idx(ceil(end_y/stride),ceil(end_x/stride),:) = max_idx;
                    % 对1,2维取最大值
                    % output(ceil(end_y/stride),ceil(end_x/stride),:) = max(max(pool_pixels,[],1),[],2);
                    % output(ceil(end_y/stride),ceil(end_x/stride),:) = max(pool_pixels,[],[1,2]);
                end
            else
                % stride=1
                for i=1:size(idx_y,1)*size(idx_y,2)
                    end_y = min(idx_y(i)+pool_size-1,rows);
                    end_x = min(idx_x(i)+pool_size-1,cols);
                    pool_pixels = input(idx_y(i):end_y,idx_x(i):end_x,:);
                     % 记录最大值与最大值位置
                    pp = reshape(pool_pixels,[],size(pool_pixels,3));
                    [max_val,max_idx] = max(pp,[],1);
                    output(idx_y(i),idx_x(i),:) = max_val;
                    maxpool_idx(idx_y(i),idx_x(i),:) = max_idx;
                    % 对1,2维取最大值
                    % output(idx_y(i),idx_x(i),:) = max(max(pool_pixels,[],1),[],2);
                    % output(idx_y(i),idx_x(i),:) = max(pool_pixels,[],[1,2]);
                end
            end
            
        end
        
        % 根据记录的位置填充梯度out_delta:1*1*filters,in_delta:2*2*filters
        function output = pad_val(in_delta,out_delta,max_idx)
            for k=1:size(in_delta,3)
               idx = max_idx(:,:,k);
               pixels4 = in_delta(:,:,k);
               val = out_delta(:,:,k);
               pixels4(idx) = val;
               in_delta(:,:,k) = pixels4;
            end
            output = in_delta;
        end
    end
end