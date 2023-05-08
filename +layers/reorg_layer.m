
% yolov2中的reorg层
classdef reorg_layer
    properties(Access=public)
        name='reorg'
        input
        output
        
        stride
    end
    
    methods(Access=public)
        function obj=forward(this)
            tmp = permute(this.input,[2,1,3]);
            [rows,cols,channels] = size(tmp);
            tmp = reshape(tmp,rows/this.stride, this.stride, cols/this.stride, this.stride, channels);
            tmp = permute(tmp,[1,3,2,4,5]);
            tmp = reshape(tmp,rows/this.stride, cols/this.stride,this.stride*this.stride*channels);
            
            this.output = tmp;
            obj = this;
            
%             [h,w,c] = size(obj.output);
%             fprintf('reorg_layer complete! reorg_size [w,h,c]: [%d,%d,%d]\n', w,h,c);
        end
    end
end