classdef ASVGF

    %Accelerated Stein Variational Gradient Flow

    properties
        lb
        ub
        logp

        alpha
        tau

        kern

        X
        Y
    end

    methods
        function obj = ASVGF(logp,lb,ub,N,kern,alpha,tau)

            obj.logp = logp;
            obj.lb = lb;
            obj.ub = ub;

            obj.X = lb + (ub-lb).*lhsdesign(N,length(lb));
            obj.Y = 0*obj.X+normrnd(0,0.01,size(obj.X));

            obj.kern = kern;

            obj.alpha = alpha;
            obj.tau = tau;

        end

        function [obj,x,F] = step(obj)

            obj.X = obj.X + obj.tau*obj.Y;

            Kxx = obj.kern.build(obj.X,obj.X);

            %V = Kxx\obj.Y;

            [nx,~] = size(obj.X);

            for i = 1:nx
                [F(i),dF(i,:)] = obj.logp(obj.X(i,:));
            end

            obj.Y = obj.alpha*obj.Y - (obj.tau/nx)*Kxx*dF;
            
            x = obj.X;

        end
    end
end