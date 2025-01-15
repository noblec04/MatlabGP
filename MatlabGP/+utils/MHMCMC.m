classdef MHMCMC

    properties
        lb
        ub
        logp
    end

    methods
        function obj = MHMCMC(logp,lb,ub)

            obj.logp = logp;
            obj.lb = lb;
            obj.ub = ub;

        end

        function [x,logp_prop, N] = step(obj,x,logp_curr)

            accept=false;

            N=0;

            for i = 1:size(x,2)
                pd = makedist('Normal','mu',x(i),'sigma',0.3);
                propd{i} = truncate(pd,obj.lb(i),obj.ub(i));
            end

            while ~accept

                N=N+1;
                
                prop=0*x;
                for i = 1:size(x,2)
                    prop(i) = random(propd{i},1);
                end
                logp_prop = obj.logp(prop);
                A =  logp_prop/ logp_curr;
                if(rand^3 < A)
                    x = prop;
                    accept=true;
                end

            end

        end
    end
end