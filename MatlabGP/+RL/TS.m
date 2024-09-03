classdef TS
    %Windowed Thompson Sampling - beta dist
    % [1] Trovo, Paladino, Restelli, & Gatti 2020
    properties
        rewards
        window
    end

    methods

        function obj = TS(window,narms)
            obj.rewards = cell(1,narms);
            obj.window = window;
        end

        function obj = addReward(obj,arm,reward)
            
            if length(obj.rewards{1})==obj.window
                for i = 1:numel(obj.rewards)
                    obj.rewards{i}(1)=[];
                end
            end

            for i = 1:numel(obj.rewards)
                obj.rewards{i}(end+1)=0;
            end

            obj.rewards{arm}(end)=log(reward+eps);
        end

        function [arm,nu] = action(obj)
            
            for i = 1:numel(obj.rewards)

                S = sum(obj.rewards{i});
                T = sum(double(obj.rewards{i}~=0));

                sig = 1./(1/100 + T);
                mu = sig*S;

                nu(i) = normrnd(mu,sqrt(sig));
            end

            [~,arm] = max(nu);

        end

        function plotDists(obj)

            X = -10:0.01:10;

            for i = 1:numel(obj.rewards)

                S = sum(obj.rewards{i});
                T = sum(double(obj.rewards{i}~=0));

                sig = 1./(1/100 + T);
                mu = sig*S;

                Y = normpdf(X + mu,mu,sig);

                plot(X+mu,Y);

            end
        end

        function lik = likelihood(obj,R)

            for i = 1:numel(obj.rewards)

                S = sum(obj.rewards{i});
                T = sum(double(obj.rewards{i}~=0));

                sig = 1./(1/100 + T);
                mu = sig*S;

                lik(i) = normpdf(R(i),mu,sig);

            end
        end

    end
end