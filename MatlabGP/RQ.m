classdef RQ<Kernel
    
    properties
        theta
    end

    methods

        function obj = RQ(scale,theta)
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

            alpha = 1;
        
            K = (1 + (d.^2)/2).^(-1*alpha);

            if nargout>1
                dK = 0*K;
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