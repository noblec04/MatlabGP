classdef WN_Het<kernels.Kernel
    
    properties
        theta
        lb
        ub
    end

    methods

        function obj = WN_Het(scale,theta)
            obj.scale = scale;
            obj.scales{1} = scale;
            obj.theta = theta;
            obj.thetas{1} = theta;
            obj.kernels{1} = obj;
            obj.w.map = 'none';
            obj.warping{1} = obj.w;
            
            obj.lb = eps;
            obj.ub = 10;
        end

        function [K] = forward_(obj,x1,x2,theta)

            n1 = size(x1,1);
            n2 = size(x2,1);

            if n1==n2
                K = diag(theta);
            else
                d = obj.dist(x1./theta,x2./theta);
                K = 0*d;
            end

        end
        
    end
end