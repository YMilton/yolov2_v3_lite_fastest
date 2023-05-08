% 上采样层
classdef upsample_layer
    properties(Access=public)
        name='upsample'
        stride
        input
        output
    end
    
    methods(Access=public)
        function obj = upsample_layer(stride)
            obj.stride = stride;
        end
        
        function obj = forward(this)
            [~,~,~,batch] = size(this.input);
            import layers.upsample_layer.single_upsample
            input_ = this.input;
            stride_ = this.stride;
            if batch==1
                output_ = single_upsample(input_,stride_);
                this.output = output_;
            else
                parfor b=1:batch
                    output_(:,:,:,b) = single_upsample(input_(:,:,:,b),stride_);
                end
                this.output = output_;
            end
            obj = this;
            
%             [h,w,c] = size(obj.output);
%             fprintf('upsample complete! upsample_size [w,h,c]: [%d,%d,%d]\n', w,h,c);
        end
    end
    
    methods(Static)
        % 单幅图像的上采样
        function output = single_upsample(input,stride)
            [rows,cols,channels] = size(input);
            for c=1:channels
                output(:,:,c) = imresize(input(:,:,c),[rows*stride,cols*stride],'nearest');
            end
        end
    end
end