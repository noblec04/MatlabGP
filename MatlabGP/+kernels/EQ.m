classdef EQ<kernels.Kernel
    
    properties
        theta
    end

    methods

        function obj = EQ(scale,theta)
            obj.scale = scale;
            obj.scales{1} = scale;
            obj.theta = theta;
            obj.thetas{1} = theta;
            obj.kernels{1} = obj;
            obj.w.map = 'none';
            obj.warping{1} = obj.w;
        end


        function [K,dK] = forward(obj,x1,x2,theta)

            nD = size(x1,2);
            nT = numel(theta);

            d = obj.dist(x1./theta,x2./theta);

            K = exp(-d.^2);

            if nargout>1
                dK = zeros(size(K,1),size(K,2),nT);
                for i = 1:nT
                    dK(:,:,i) = (2/theta(i))*((x1(:,i) - x2(:,i)').^2).*K;
                end
                dK(dK(:,:,1:nT) < 1e-12) = 0;
            end
        end

        function dK = grad(obj,x1,x2,theta)

        end

        function obj = periodic(obj,dim,P)
            obj.w.period = P;
            obj.w.dim = dim;
            obj.warping{1} = obj.w;
            obj.warping{1}.map = 'periodic';
        end
    end
end