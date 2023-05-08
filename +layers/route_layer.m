% 路由层
classdef route_layer
    properties(Access=public)
        name='route'
        layers % 路由的层索引,数据类型为数字或数组
        filters % 路由层的输出通道
        
        % 路由存在分组的情况
        groups=1
        group_id=1
        
        input
        output
    end
    
    methods(Access=public)
        % layers是字符串
        function obj=route_layer(layers,varargin)
            obj.layers = str2num(layers);
            if ~isempty(varargin) && nargin>0
                obj.groups = varargin{1};
                obj.group_id = varargin{2};
            end
        end
        
        % mynet整个网络，current_idx当前层
        function obj = forward(this,mynet,current_idx)
            if length(this.layers)==1
                if this.layers<0
                    this.input = mynet{current_idx + this.layers}.output;
                else
                    this.input = mynet{this.layers + 1}.output;
                end
                
                % route存在2分组的情况
                if this.groups==2 && this.group_id==1
                    channels = size(this.input,3);
                    channels_per_group = channels/this.groups;
                    start_idx = (this.group_id - 1)*channels_per_group + 1;
                    end_idx = this.group_id*channels_per_group;
                    this.input = this.input(:,:,start_idx:end_idx);
                end
                
            else % 存在多个路由
                this.input = []; % 重新构建，先置为空
                for k=1:length(this.layers)
                    if this.layers(k)<0
                        this.input = cat(3,this.input,mynet{current_idx + this.layers(k)}.output);
                    else
                        this.input = cat(3,this.input,mynet{this.layers(k)+1}.output); % this.layers(k)+1 matlab从1开始，darknet从0开始
                    end
                end
            end
            this.output = this.input;
            
            obj = this;
            
%             [h,w,c] = size(obj.output);
%             fprintf('route_layer complete! route_size [w,h,c]: [%d,%d,%d]\n', w,h,c);
        end
    end
end