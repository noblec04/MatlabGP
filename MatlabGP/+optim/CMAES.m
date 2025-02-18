classdef CMAES

    properties

        iter = 0;

        lb = [];
        ub = [];
        
        wd

        mk
        Ck

        sigma
        w
        lambda
        mu
        mueff

        Ps
        Pc

        cc
        cs
        c1
        cmu
        damps

        ENN

        xmin
        fmin
        
    end

    methods

        function obj = CMAES(n,N,varargin)

            input=inputParser;
            input.KeepUnmatched=true;
            input.PartialMatching=false;
            input.addOptional('lb',[]);
            input.addOptional('ub',[]);
            input.addOptional('wd',0);
            input.parse(varargin{:})
            in=input.Results;


            obj.lb = in.lb;
            obj.ub = in.ub;

            obj.wd = in.wd;

            if isempty(obj.lb)
                lb1 = 0;
                ub1 = 1;
            else
                lb1 = obj.lb;
                ub1 = obj.ub;
            end

            obj.mk = lb1 + (ub1 - lb1).*normrnd(0,1,[1 n]);
            obj.Ck = eye(n);

            obj.xmin = obj.mk;

            obj.Pc = zeros(n,1);
            obj.Ps = zeros(n,1);

            obj.sigma = 1;

            obj.lambda = 4 + floor(3*log(N));
            obj.mu = obj.lambda/2;
            obj.w = log(obj.mu + 0.5) - log(1:obj.mu);

            obj.mu = floor(obj.mu);

            obj.w = obj.w/sum(obj.w);

            obj.mueff = sum(obj.w)^2/sum(obj.w.^2);

            obj.cc = (4 + obj.mueff/N)/(N+4+2*obj.mueff/N);
            obj.cs = (obj.mueff+2)/(N+obj.mueff+5);
            obj.c1 = 2/((N+1.3)^2 + obj.mueff);
            obj.cmu = min(1-obj.c1,2*(obj.mueff - 2 + 1/obj.mueff)/((N+2)^2 + obj.mueff));

            obj.damps = 1 + 2*max(0, sqrt((obj.mueff-1)/(N+1))-1) + obj.cs;

            obj.ENN = sqrt(N)*(1 - (1/(4*N)) + (1/(21*N^2)));
            
        end

        function [obj,x,f] = step(obj,F)

            obj.iter = obj.iter + 1;

            xi = mvnrnd(obj.mk,(obj.sigma)*obj.Ck,obj.lambda);

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

            for i = 1:obj.lambda
                fi(i) = F(xi(i,:));
            end

            [fi,io] = sort(fi,'ascend');

            xi = xi(io,:);

            if obj.iter==1
                obj.fmin = fi(1);
            end

            mkp1 = obj.w*xi(1:obj.mu,:);

            dm = (mkp1 - obj.mk)./obj.sigma;

            obj.Ps = (1-obj.cs)*obj.Ps + sqrt(obj.cs*(2-obj.cs))*sqrt(obj.mueff)*(sqrt(obj.Ck)\dm')';

            hsig = norm(obj.Ps)/sqrt(1-(1-obj.cs)^(2*obj.iter/obj.lambda))/obj.ENN < 1.4 + 2/(size(xi,1)+1);

            obj.Pc = (1 - obj.cc)*obj.Pc + hsig*sqrt(1 - (1 - obj.cc)^2)*sqrt(obj.mu)*dm;

            dx = (xi(1:obj.mu,:) - obj.mk)/obj.sigma;

            obj.Ck = (1 - obj.c1 - obj.cmu)*obj.Ck + obj.c1*(obj.Pc*obj.Pc' + (1-hsig)*obj.cc*(2-obj.cc)*obj.Ck)...
                +obj.cmu*dx'*diag(obj.w)*dx;

            % obj.Ck = abs(obj.Ck);
            %obj.Ck = triu(obj.Ck) + triu(obj.Ck,1)';

            obj.sigma = obj.sigma*exp((obj.cs/obj.damps)*(norm(obj.Ps)/obj.ENN - 1));

            obj.mk = mkp1;

            if obj.iter>1&&rand>0.85

                if fi(1)>obj.fmin
                    obj.mk = obj.xmin;
                    obj.sigma = 0.3;
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