classdef GE<kernels.Kernel
    
    properties
        theta
    end

    methods

        function obj = GE(scale,theta)
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

            sig1 = theta(1);
            sig2 = theta(2);

            d = obj.dist(x1./sig2,x2./sig2);

            d1 = x1*x1'/sig1;
            d2 = x2*x2'/sig1;

            K = exp(-abs(d1))*exp(-d.^2)*exp(-abs(d2));            

            if nargout>1
                dK = 0*K;
            end
        end

        function obj = periodic(obj,dim,P)
            obj.w.period = P;
            obj.w.dim = dim;
            obj.warping{1} = obj.w;
            obj.warping{1}.map = 'periodic';
        end
    end
end