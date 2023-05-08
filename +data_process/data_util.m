% 数据处理工具
classdef data_util
  
    methods(Static)
        % 读取文件内容的每一行，保存为cell
        function context = get_file_context(file)
%             context = [];
%             fid = fopen(file,'r');
%             while 1
%                 tline = fgetl(fid);
%                 if ~ischar(tline)
%                     fclose(fid);
%                     break;
%                 end
%                 
%                 context = [context;{strtrim(tline)}];
%             end
            fid = fopen(file);
            cell_names = textscan(fid,'%s','Delimiter','\n');
            context = cell_names{1};
            fclose(fid);
        end
        
        % 一幅图像的前向传播
        function out_val = forward_net(mynet, img_path, input_size)
            mn = my_network;
            mn.mynet = mynet;
            mn.input = single(imresize(imread(img_path),[input_size,input_size]))/255;
            mn = mn.forward; % 网络的前向传播(单张图像)
            out_val = mn.yolos;
        end
        
        
        % 多个gt.wh与anchors的交并比计算，返回最大iou与索引
        function [best_ious,best_idx] = iou_anchors(whs,anchors)
            import data_process.data_util
            for k=1:size(whs,1)
                ious = data_util.iou_bbox_wh(whs(k,:),anchors);
                [max_val,max_idx] = max(ious,[],1);
                best_ious(k,1) = max_val;
                best_idx(k,1) = max_idx;
            end
        end
        
        % 交并比计算(只有宽高)
        function iou = iou_bbox_wh(wh1,wh2)
            w1 = wh1(:,1); h1 = wh1(:,2);
            w2 = wh2(:,1); h2 = wh2(:,2);
            inter_area = min(w1,w2).*min(h1,h2);
            union_area = w1.*h1 + w2.*h2 - inter_area + 1e-20;
            iou = inter_area./union_area;
        end
        
        
        % 一幅图像人工标注框读取
        function out=get_bbox(img_path)
            img_txt = replace(img_path,'images','labels');
            if endsWith(img_path,'.jpg') % jpg
                img_txt = replace(img_txt,'jpg','txt');
            elseif endsWith(img_path,'.bmp') % bmp
                img_txt = replace(img_txt,'bmp','txt');
            elseif endsWith(img_path,'.png') % png
                img_txt = replace(img_txt,'png','txt');
            end
            
            fid = fopen(img_txt,'r');
            out = cell2mat(textscan(fid,'%f %f %f %f %f','delimiter','\n'));
            fclose(fid);
        end
        
    end
    
end