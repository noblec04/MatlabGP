classdef Env

    properties
        state
        reward
    end

    methods
        function obj = Env(S,R)
            
            obj.state = S;
            obj.reward = R;
        end

        function [R,S] = getReward(obj,action)
            
            [R, S] = obj.reward.eval(obj.state,action);

        end

        function obj = setState(obj,S)
            obj.state = S;
        end
    end
end