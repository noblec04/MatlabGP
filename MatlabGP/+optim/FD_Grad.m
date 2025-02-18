classdef FD_Grad

    properties

        iter = 0;

        lb = [];
        ub = [];
        
        wd
        lr

        mk
        sigma

        N

        xmin
        fmin
        
    end

    methods

        function obj = FD_Grad(V,N,varargin)

            input=inputParser;
            input.KeepUnmatched=true;
            input.PartialMatching=false;
            input.addOptional('lb',[]);
            input.addOptional('ub',[]);
            input.addOptional('wd',0);
            input.addOptional('lr',0.01);
            input.parse(varargin{:})
            in=input.Results;


            obj.lb = in.lb;
            obj.ub = in.ub;

            obj.wd = in.wd;
            obj.lr = in.lr;

            obj.mk = V;
            obj.xmin = obj.mk;

            obj.sigma = 0*obj.mk + 0.1;

            obj.N = N;
            
        end

        function [obj,x,f] = step(obj,F)

            obj.iter = obj.iter + 1;

            xi = mvnrnd(obj.mk,diag(obj.sigma),obj.N);

            %reflective upper bound
            if ~isempty(obj.ub)
                for jj = 1:size(xi,1)
                    for kk = 1:size(xi,2)
                        if xi(jj,kk)>obj.ub(kk)
                            xi(jj,kk)=obj.ub(kk);
                        end
                    end
                end
            end

            %reflective lower bound
            if ~isempty(obj.lb)
                for jj = 1:size(xi,1)
                    for kk = 1:size(xi,2)
                        if xi(jj,kk)<obj.lb(kk)
                            xi(jj,kk)=obj.lb(kk);
                        end
                    end
                end
            end

            for i = 1:obj.N
                fi(i) = F(xi(i,:));
            end

            [fi,io] = sort(fi,'ascend');
            xi = xi(io,:);

            if obj.iter==1
                obj.fmin = fi(1);
            end

            fm = mean(fi);
            xm = mean(xi);

            dx = xi - xm;
            df = fi - fm;

            % H = 0;
            % 
            % for i = 1:length(df)
            %     H = H + df(i)'./(dx(i,:)'*dx(i,:));
            % end
            % 
            % H = H/length(df);

            warning off

            dfdx = mean(df'./dx);
            dfdxs = std(df'./dx);

            dx = dfdx;%(H\dfdx')';

            obj.mk = obj.mk - obj.lr*(dx)/norm(dx) - obj.wd*obj.mk;%(dfdx+2*dfdxs);
            obj.sigma = obj.sigma.*exp(-obj.lr*(obj.sigma - dfdxs/norm(dfdxs)));

            if obj.iter>1&&rand>0.85

                if fi(1)>obj.fmin
                    obj.mk = obj.xmin;
                end

            end

            x = xi(1,:);
            f = fi(1);

            if f<obj.fmin
                obj.xmin = x;
                obj.fmin = f;
            end

            x = obj.xmin;
            f = obj.fmin;

        end

    end
end