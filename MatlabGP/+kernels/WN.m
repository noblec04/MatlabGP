classdef WN<kernels.Kernel
    
    properties
        theta
    end

    methods

        function obj = WN(scale,theta)
            obj.scale = scale;
            obj.scales{1} = scale;
            obj.theta = theta;
            obj.thetas{1} = theta;
            obj.kernels{1} = obj;
            obj.w.map = 'none';
            obj.warping{1} = obj.w;
        end

        function [K] = forward_(~,x1,x2,theta)

            n1 = size(x1,1);
            n2 = size(x2,1);

            if n1==n2
                K = theta*eye(n1,n2);
            else
                a = zeros(n1,n2);
                K = 0*a;
            end

        end
        
    end
end