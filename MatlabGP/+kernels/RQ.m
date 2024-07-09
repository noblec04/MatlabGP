classdef RQ<kernels.Kernel
    
    properties
        theta
        alpha
    end

    methods

        function obj = RQ(alpha,scale,theta)
            obj.alpha = alpha;
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
        
            K = (1 + (d.^2)/(2*obj.alpha)).^(-1*obj.alpha);
        end

        % function [K,dK] = forward(obj,x1,x2,theta)
        % 
        %     nD = size(x1,2);
        %     nT = numel(theta);
        % 
        %     d = obj.dist(x1./theta,x2./theta);
        % 
        %     K = (1 + (d.^2)/(2*obj.alpha)).^(-1*obj.alpha);
        % 
        %     if nargout>1
        %         dK = zeros(size(K,1),size(K,2),nT);
        %         for i = 1:nT
        %             dK(:,:,i) = (1/theta(i))*((x1(:,i) - x2(:,i)').^2).*K.^2;
        %         end
        %         dK(abs(dK(:,:,1:nT)) < 1e-12) = 0;
        %     end
        % end
    end
end