classdef CEQ<kernels.Kernel
    
    properties
        theta
        alpha
    end

    methods

        function obj = CEQ(alpha,scale,theta)
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

            K = (1 - erf(d*obj.alpha/4)).*exp(-d.^2);

        end

        % function [K,dK] = forward(obj,x1,x2,theta)
        % 
        %     nn = size(x1,1);
        %     nT = numel(theta);
        % 
        %     if nargout==1
        % 
        %         d = obj.dist(x1./theta,x2./theta);
        % 
        %         K = (1 - erf(d*obj.alpha/4)).*exp(-d.^2);
        % 
        %     else
        %         thetaAD = AutoDiff(theta);
        % 
        %         dAD = obj.dist(x1./thetaAD,x2./thetaAD);
        % 
        %         KAD = (1 - erf(dAD*obj.alpha/4)).*exp(-dAD.^2);
        % 
        %         K = getvalue(KAD);
        %         dK = getderivs(KAD);
        %         dK = reshape(full(dK),[nn nn nT]);
        %     end
        % end
    end
end