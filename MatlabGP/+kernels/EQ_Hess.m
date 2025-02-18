classdef EQ_Hess<kernels.Kernel
    
    properties
        theta
    end

    methods

        function obj = EQ_Hess(scale,theta)
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

            K = exp(-d.^2);

            K = K + (d.^2).*K;

        end

        function I = qK(obj,x2,lb,ub,theta)
            
            %Emukit Bayesian Quadrature

            elb = erf((lb - x2)./theta);
            eub = erf((ub - x2)./theta);

            Km = sqrt(pi/2)*theta.*(eub - elb);

            A = ones(size(Km(:,1)));

            for i = 1:size(Km,2)
                A = A.*Km(:,i);
            end

            I = obj.scale*A;%prod(Km,2);
        
        end

        function I = qKm(obj,x2,lb,ub,theta)
            
            %Emukit Bayesian Quadrature

            elb = erf((lb - x2)./theta);
            eub = erf((ub - x2)./theta);

            Km = sqrt(pi)*theta.*(eub - elb)/2;

            A = ones(size(Km(:,1)));

            for i = 1:size(Km,2)
                A = A.*Km(:,i);
            end

            I = obj.scale*A;%prod(Km,2);
        
        end

        function I = qKq(obj,lb,ub,theta)

            %Emukit Bayesian Quadrature

            dul = (ub - lb)./theta;
            expTerm = (exp(-dul.^2) - 1)/sqrt(pi);
            erfTerm = erf(dul).*dul;
            totalTerm = (2*sqrt(pi)*theta.^2).*(expTerm+erfTerm);

            A = ones(size(totalTerm(:,1)));

            for i = 1:size(totalTerm,2)
                A = A.*totalTerm(:,i);
            end

            I = obj.scale*A;%prod(totalTerm,2);

        end
        
    end
end