classdef WTS
    %Windowed Thompson Sampling - beta dist
    % [1] Trovo, Paladino, Restelli, & Gatti 2020
    properties
        rewards
        window
    end

    methods

        function obj = WTS(window,narms)
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

            obj.rewards{arm}(end)=reward;
        end

        function arm = action(obj,sig)

            if nargin<2
                sig = 0*[1:numel(obj.rewards)] + 1;
            end

            
            for i = 1:numel(obj.rewards)

                S = sum(obj.rewards{i});
                T = sum(double(obj.rewards{i}>0));

                nu(i) = betarnd(S+1,T-S+1)*sig(i);
            end

            [~,arm] = max(nu);

        end

        function plotDists(obj)

            X = 0:0.001:1;

            figure
            hold on
            for i = 1:numel(obj.rewards)

                S = sum(obj.rewards{i});
                T = sum(double(obj.rewards{i}>0));

                Y = betapdf(X,S+1,T-S+1);

                plot(X,Y);

            end
            

        end

    end
end