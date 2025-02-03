classdef PSO

    properties

        iter = 0;

        lb = [];
        ub = [];

        lr

        x
        v
        f
        df

        xbest
        fbest

        beta1
        beta2

        M

    end

    methods

        function obj = PSO(N,x0,varargin)

            input=inputParser;
            input.KeepUnmatched=true;
            input.PartialMatching=false;
            input.addOptional('lb',[]);
            input.addOptional('ub',[]);
            input.addOptional('beta1',0.1);
            input.addOptional('beta2',0.2);
            input.addOptional('M',1);
            input.parse(varargin{:})
            in=input.Results;

            obj.lb = in.lb;
            obj.ub = in.ub;

            nD = length(x0);

            obj.x = lhsdesign(N,nD);
            obj.v = 2*lhsdesign(N,nD)-1;
            obj.f = zeros(N,1);

            obj.xbest = zeros(N,nD);
            obj.fbest = zeros(N,1);

            obj.beta1 = in.beta1;
            obj.beta2 = in.beta2;

            obj.M = in.M;
        end

        function [obj,xbest,fbest] = step(obj,F)

            obj.iter = obj.iter + 1;

            r1 = rand;
            r2 = rand;

            if obj.iter>1
                [~,ib] = min(obj.fbest);
                personal_coefficient = obj.beta1 * r1 * (obj.xbest - obj.x);
                social_coefficient = obj.beta2 * r2 * (obj.xbest(ib,:) - obj.x);
                vn = obj.M * obj.v + personal_coefficient + social_coefficient;
            else
                vn = obj.M * obj.v;
            end

            obj.v = vn;

            obj.x = obj.x + obj.v;

            %reflective upper bound
            if ~isempty(obj.ub)
                for kk = 1:size(obj.x,1)
                    for jj = 1:size(obj.x,2)
                        if obj.x(kk,jj)>obj.ub(jj)
                            obj.x(kk,jj)=obj.ub(jj) - 0.1*abs(obj.v(kk,jj));
                        end
                    end
                end
            end

            %reflective lower bound
            if ~isempty(obj.lb)
                for kk = 1:size(obj.x,1)
                    for jj = 1:length(obj.x)
                        if obj.x(kk,jj)<obj.lb(jj)
                            obj.x(kk,jj)=obj.lb(jj) + 0.1*abs(obj.v(kk,jj));
                        end
                    end
                end
            end

            for i = 1:size(obj.x,1)
                obj.f(i) = F(obj.x(i,:));
            end

            for i = 1:size(obj.f,1)

                if obj.f(i)<obj.fbest(i)

                    obj.fbest(i) = obj.f(i);
                    obj.xbest(i,:) = obj.x(i,:);

                end

            end

            if obj.iter == 1
                obj.fbest = obj.f;
                obj.xbest = obj.x;
            end

            [~,ib] = min(obj.fbest);

            fbest = obj.fbest(ib);
            xbest = obj.xbest(ib,:);

        end

    end
end