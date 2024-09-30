classdef Agent
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties
        state
        Q
    end

    methods
        function obj = Agent(Q,S)
            
            obj.Q = Q;
            obj.state = S;

        end

        function a = action(obj,state)
            
            a = obj.Q.eval(state);

        end

        function obj = train(obj,S,R)

            obj.Q = obj.Q.train(S,R);

        end
    end
end