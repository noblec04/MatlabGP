classdef AdamLS

    properties

        iter = 0;

        lb = [];
        ub = [];
        
        lr
        
        wd

        beta1
        beta2
        
        mt
        vt

        dFt=0;
        
    end

    methods

        function obj = AdamLS(x0,varargin)

            input=inputParser;
            input.KeepUnmatched=true;
            input.PartialMatching=false;
            input.addOptional('lb',[]);
            input.addOptional('ub',[]);
            input.addOptional('beta1',0.9);
            input.addOptional('beta2',0.999);
            input.addOptional('lr',0.1);
            input.addOptional('wd',0);
            input.parse(varargin{:})
            in=input.Results;

            obj.lb = in.lb;
            obj.ub = in.ub;

            obj.lr = in.lr;
            
            obj.mt = 0*x0;
            obj.vt = 0*x0;
            
            obj.beta1 = in.beta1;
            obj.beta2 = in.beta2;
            
            obj.wd = in.wd;
            
            
        end

        function [obj,x] = step(obj,x,F,dF)

            obj.iter = obj.iter + 1;

            dF = dF + obj.wd.*x;
            
            obj.mt = obj.beta1*obj.mt + (1 - obj.beta1)*dF;
            obj.vt = obj.beta2*obj.vt + (1 - obj.beta2)*dF.^2;
            
            mth = obj.mt./(1 - obj.beta1);
            vth = obj.vt./(1 - obj.beta2);

            dG = obj.dFt - dF;

            obj.dFt = dF;

            DFC = 1./(1 + exp(-1*abs(dG)));
            

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%% End Update Algorithm params %%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %update parameters
            lrtest = [0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.15 0.2 0.5 1];

            for i = 1:length(lrtest)
                
                xi = x - lrtest(i)*DFC.*mth./(sqrt(vth) + eps );
                yi(i) = F(xi);

            end

            [~,imin] = min(yi);
            obj.lr = lrtest(imin);

            x = x - obj.lr*DFC.*mth./(sqrt(vth) + eps );

            %reflective upper bound
            if ~isempty(obj.ub)
                for jj = 1:length(x)
                    if x(jj)>obj.ub(jj)
                        x(jj)=obj.ub(jj) - 0.1*abs(obj.lr*mth(jj)./(sqrt(vth(jj)) + eps));
                    end
                end
            end

            %reflective lower bound
            if ~isempty(obj.lb)
                for jj = 1:length(x)
                    if x(jj)<obj.lb(jj)
                        x(jj)=obj.lb(jj) + 0.1*abs(obj.lr*mth(jj)./(sqrt(vth(jj)) + eps));
                    end
                end
            end

        end

    end
end