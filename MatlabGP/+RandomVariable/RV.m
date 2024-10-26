classdef RV
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        dist
        pd
        
        res
        
        lb
        ub
        
        mu
        std
        
    end
    
    methods
        function obj = RV(varargin)
           
            input=inputParser;
            input.KeepUnmatched=true;
            input.PartialMatching=false;
            input.addOptional('dist',normrnd(0,1,[100,1]));
            input.addOptional('lb',[]);
            input.addOptional('ub',[]);
            input.addOptional('res',1000);
            input.parse(varargin{:})
            in=input.Results;
            
            
            obj.dist = in.dist;
            
            if~isempty(in.lb)
                obj.lb = in.lb;
            else
                obj.lb = min(in.dist);
            end
            
            if~isempty(in.ub)
                obj.ub = in.ub;
            else
                obj.ub = max(in.dist);
            end
            
            obj.res = in.res;
            
            obj.pd = fitdist(obj.dist(:),'kernel');
            
            obj.mu = obj.pd.mean;
            obj.std = obj.pd.std;
            
        end
        
        function I = expectation(obj,ff)
           
            x = random(obj.pd,[obj.res,1]);
            
            I = mean(ff(x));
            
        end
        
        function x = sample(obj,n)
            
            x = random(obj.pd,[n,1]);
            
        end
        
        function plot(obj,x)
           
            plot(x,obj.pd.pdf(x));
            
        end
        
        function plotCDF(obj,x)
           
            plot(x,obj.pd.cdf(x));
            
        end

        function [x1,x2] = sample_(obj,obj2)

            pd1 = obj.pd;
            pd2 = obj2.pd;
            
            x1 = random(pd1,[max(obj.res,obj2.res),1]);
            x2 = random(pd2,[max(obj.res,obj2.res),1]);

        end

        function yy = build_(obj,obj2,aa)

            pd3 = fitdist(aa,'kernel');
           
            x = random(pd3,[max(obj.res,obj2.res),1]);
            
            yy = RandomVariable.RV('dist',x,'lb',min(aa),'ub',max(aa),'res',max(obj.res,obj2.res));

        end

        function [x1] = selfsample_(obj)

            pd1 = obj.pd;
            x1 = random(pd1,[obj.res,1]);

        end

        function yy = selfbuild_(obj,aa)

            pd3 = fitdist(aa,'kernel');
           
            x = random(pd3,[obj.res,1]);
            
            yy = RandomVariable.RV('dist',x,'lb',min(aa),'ub',max(aa),'res',obj.res);

        end

        function yy = cos(obj)
            
            [x1] = obj.selfsample_();
            
            aa = cos(x1(:));
            
            yy = obj.selfbuild_(aa);

        end

        function yy = sin(obj)
            
            [x1] = obj.selfsample_();
            
            aa = sin(x1(:));
            
            yy = obj.selfbuild_(aa);

        end

        function yy = exp(obj)
            
            [x1] = obj.selfsample_();
            
            aa = exp(x1(:));
            
            yy = obj.selfbuild_(aa);

        end

        function yy = acos(obj)
            
            [x1] = obj.selfsample_();
            
            aa = real(acos(x1(:)));
            
            yy = obj.selfbuild_(aa);

        end

        function yy = atan(obj)
            
            [x1] = obj.selfsample_();
            
            aa = atan(x1(:));
            
            yy = obj.selfbuild_(aa);

        end
        
        function yy = plus(obj,obj2)
                      
            [x1,x2] = obj.sample_(obj2);
            
            aa = x1(:) + x2(:);
            
            yy = obj.build_(obj2,aa);
        end
        
        function yy = minus(obj,obj2)
                      
            [x1,x2] = obj.sample_(obj2);
            
            aa = x1(:) - x2(:);
            
            yy = obj.build_(obj2,aa);
            
        end
        
        function yy = times(obj,obj2)
                      
            [x1,x2] = obj.sample_(obj2);
            
            aa = x1(:).*x2(:);
            
            yy = obj.build_(obj2,aa);
            
        end
        
        function yy = rdivide(obj,obj2)
                      
            [x1,x2] = obj.sample_(obj2);
            
            aa = x1(:)./x2(:);
            aa(isnan(aa)) = [];
            aa(isinf(aa))=[];
            
            yy = obj.build_(obj2,aa);
            
        end
        
        function yy = power(obj,obj2)
                      
            [x1,x2] = obj.sample_(obj2);
            
            aa = real(x1(:).^x2(:));
            aa(isnan(aa)) = [];
            aa(isinf(aa))=[];
            
            yy = obj.build_(obj2,aa);
            
        end
        
        function yy = lt(obj,obj2)
                      
            [x1,x2] = obj.sample_(obj2);
            
            aa = x1(:) < x2(:);
            aa(isnan(aa)) = [];
            aa(isinf(aa))=[];
            
            yy = obj.build_(obj2,aa);
            
        end
    end
end

