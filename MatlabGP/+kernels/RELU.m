classdef RELU<kernels.Kernel
    
    properties
        theta
    end

    methods

        function obj = RELU(scale,theta)
            obj.scale = scale;
            obj.scales{1} = scale;
            obj.theta = theta;
            obj.thetas{1} = theta;
            obj.kernels{1} = obj;
            obj.w.map = 'none';
            obj.warping{1} = obj.w;
        end

        function [K] = forward_(obj,x1,x2,theta)

            d = obj.dist(x1./theta,x2./theta);

            K = max(1-abs(d),0);
        end

        % function [K,dK] = forward(obj,x1,x2,theta)
        % 
        %     nD = size(x1,2);
        %     nT = numel(theta);
        % 
        %     d = obj.dist(x1./theta,x2./theta);
        % 
        %     K = max(1-abs(d),0);
        % 
        %     if nargout>1
        %         dK = zeros(size(K,1),size(K,2),nT);
        %         for i = 1:nT
        %             dK(:,:,i) = (2/theta(i))*((x1(:,i) - x2(:,i)').^2).*K;
        %         end
        %         dK(abs(dK(:,:,1:nT)) < 1e-12) = 0;
        %     end
        % end
    end
end