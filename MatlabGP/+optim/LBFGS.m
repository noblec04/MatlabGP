classdef LBFGS

    properties

        iter = 0;

        lb = [];
        ub = [];

        lr

        m

        xo
        go

        rho

        y
        s

    end

    methods

        function obj = LBFGS(x0,varargin)

            input=inputParser;
            input.KeepUnmatched=true;
            input.PartialMatching=false;
            input.addOptional('lb',[]);
            input.addOptional('ub',[]);
            input.addOptional('m',5);
            input.addOptional('lr',0.02);
            input.parse(varargin{:})
            in=input.Results;

            obj.lb = in.lb;
            obj.ub = in.ub;

            obj.lr = in.lr;

            obj.m = in.m;

            obj.xo = x0;
            obj.go = 0*x0;

        end

        function [obj,x] = step(obj,x,dF)

            obj.iter = obj.iter + 1;
            k = obj.iter;

            obj.s{mod(k-1, obj.m) + 1} = x - obj.xo;
            obj.y{mod(k-1, obj.m) + 1} = dF - obj.go;

            % Calculate rho and store
            obj.rho(mod(k-1, obj.m) + 1) = 1 / max(abs(obj.y{mod(k-1, obj.m) + 1} * obj.s{mod(k-1, obj.m) + 1}'),0.1);

            obj.xo = x;
            obj.go = dF;

            if k<=obj.m
                x = x - obj.lr *dF/20;
                return
            end

            % Two-loop recursion for computing the update direction (q)

            % First loop (forward pass)
            alpha = zeros(obj.m, 1);
            q = dF;

            for i = obj.m:-1:1
                j = mod(k-i, obj.m) + 1; % Circular buffer indexing
                alpha(i) = obj.rho(j) * obj.s{j} * q';
                q = q - alpha(i) * obj.y{j};
            end

            % Apply initial Hessian approximation
            r = q;  % Or r = q if H0 is the identity

            % Second loop (backward pass)
            for i = 1:obj.m
                j = mod(k-i, obj.m) + 1; % Circular buffer indexing
                beta = obj.rho(j) * obj.y{j} * r';
                r = r + (alpha(i) - beta) * obj.s{j};
            end

            % Compute search direction (p)
            p = -r;

            p = -1*sign(dF).*min(abs(p),abs(dF));

            x = x + obj.lr * p/k^(0.2);

            %reflective upper bound
            if ~isempty(obj.ub)
                for jj = 1:length(x)
                    if x(jj)>obj.ub(jj)
                        x(jj)=obj.ub(jj) - 0.1*abs(obj.lr*p(jj));
                    end
                end
            end

            %reflective lower bound
            if ~isempty(obj.lb)
                for jj = 1:length(x)
                    if x(jj)<obj.lb(jj)
                        x(jj)=obj.lb(jj) + 0.1*abs(obj.lr*p(jj));
                    end
                end
            end

        end

    end
end