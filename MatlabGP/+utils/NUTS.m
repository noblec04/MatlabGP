classdef NUTS

    % Copyright (c) 2011, Matthew D. Hoffman
    % All rights reserved.
    % 
    % Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    % 
    % Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    % Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    properties
        epsilon
        max_tree_depth
    end

    methods
        function obj = NUTS(epsilon,max_tree_depth)
            obj.epsilon = epsilon;
            obj.max_tree_depth = max_tree_depth;

        end

        function [theta, alpha_ave, logp, grad] = step(obj,f,theta0)

            [logp0, grad0] = f(theta0);
            %logp0 = -1*logp0;
            %grad0 = -1*grad0;

            % Resample momenta.
            r0 = randn(size(theta0));
            % Joint log-probability of theta and momenta r.
            joint = logp0 - 0.5 * (r0 * r0');
            % Resample u ~ uniform([0, exp(joint)]).
            % Equivalent to (log(u) - joint) ~ exponential(1).
            logu = joint - exprnd(1);
            % Initialize tree.
            thetaminus = theta0;
            thetaplus = theta0;
            rminus = r0;
            rplus = r0;
            gradminus = grad0;
            gradplus = grad0;
            % Initial height dir = 0.
            depth = 0;
            % If all else fails, the next sample is the previous sample.
            theta = theta0;
            grad = grad0;
            logp = logp0;
            % Initially the only valid point is the initial point.
            n = 1;

            % Main loop---keep going until the stop criterion is met.
            stop = false;
            while ~stop
                % Choose a direction. -1=backwards, 1=forwards.
                dir = 2 * (rand() < 0.5) - 1;
                % Double the size of the tree.
                if (dir == -1)
                    [thetaminus, rminus, gradminus, ~, ~, ~, thetaprime, gradprime, logpprime, nprime, stopprime, alpha, nalpha] = ...
                        obj.build_tree(thetaminus, rminus, gradminus, logu, dir, depth, f, joint);
                else
                    [~, ~, ~, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, stopprime, alpha, nalpha] = ...
                        obj.build_tree(thetaplus, rplus, gradplus, logu, dir, depth, f, joint);
                end
                % Use Metropolis-Hastings to decide whether or not to move to a
                % point from the half-tree we just generated.
                if (~ stopprime && (rand() < nprime/n))
                    theta = thetaprime;
                    logp = logpprime;
                    grad = gradprime;
                end
                % Update number of valid points we've seen.
                n = n + nprime;
                % Decide if it's time to stop.
                stop = stopprime || obj.stop_criterion(thetaminus, thetaplus, rminus, rplus);
                % Increment depth.
                depth = depth + 1;

                if depth > obj.max_tree_depth
                    %disp('The current NUTS iteration reached the maximum tree depth.')
                    break
                end
            end
            alpha_ave = alpha / nalpha;


        end


        function [thetaprime, rprime, gradprime, logpprime] = leapfrog(~,theta, r, grad, epsilon, f)

            rprime = r + 0.5 * epsilon * grad;
            thetaprime = theta + epsilon * rprime;
            [logpprime, gradprime] = f(thetaprime);

            logpprime = -1*logpprime;
            gradprime = -1*gradprime;

            rprime = rprime + 0.5 * epsilon * gradprime;

        end

        function criterion = stop_criterion(~,thetaminus, thetaplus, rminus, rplus)

            thetavec = thetaplus - thetaminus;
            criterion = (thetavec * rminus' < 0) || (thetavec * rplus' < 0);

        end

        % The main recursion.
        function [thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, stopprime, alphaprime, nalphaprime] = ...
                build_tree(obj,theta, r, grad, logu, dir, depth, f, joint0)

            epsilon = obj.epsilon;
            if (depth == 0)
                % Base case: Take a single leapfrog step in the direction 'dir'.
                [thetaprime, rprime, gradprime, logpprime] = obj.leapfrog(theta, r, grad, dir*epsilon, f);
                joint = logpprime - 0.5 * (rprime * rprime');
                % Is the new point in the slice?
                nprime = logu < joint;
                % Is the simulation wildly inaccurate?
                stopprime = logu - 100 >= joint;
                % Set the return values---minus=plus for all things here, since the
                % "tree" is of depth 0.
                thetaminus = thetaprime;
                thetaplus = thetaprime;
                rminus = rprime;
                rplus = rprime;
                gradminus = gradprime;
                gradplus = gradprime;
                % Compute the acceptance probability.
                alphaprime = exp(logpprime - 0.5 * (rprime * rprime') - joint0);
                if isnan(alphaprime)
                    alphaprime = 0;
                else
                    alphaprime = min(1, alphaprime);
                end
                nalphaprime = 1;
            else
                % Recursion: Implicitly build the height depth-1 left and right subtrees.
                [thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, stopprime, alphaprime, nalphaprime] = ...
                    obj.build_tree(theta, r, grad, logu, dir, depth-1, f, joint0);
                % No need to keep going if the stopping criteria were met in the first
                % subtree.
                if ~stopprime
                    if (dir == -1)
                        [thetaminus, rminus, gradminus, ~, ~, ~, thetaprime2, gradprime2, logpprime2, nprime2, stopprime2, alphaprime2, nalphaprime2] = ...
                            obj.build_tree(thetaminus, rminus, gradminus, logu, dir, depth-1, f, joint0);
                    else
                        [~, ~, ~, thetaplus, rplus, gradplus, thetaprime2, gradprime2, logpprime2, nprime2, stopprime2, alphaprime2, nalphaprime2] = ...
                            obj.build_tree(thetaplus, rplus, gradplus, logu, dir, depth-1, f, joint0);
                    end
                    % Choose which subtree to propagate a sample up from.
                    if (rand() < nprime2 / (nprime + nprime2))
                        thetaprime = thetaprime2;
                        gradprime = gradprime2;
                        logpprime = logpprime2;
                    end
                    % Update the number of valid points.
                    nprime = nprime + nprime2;
                    % Update the stopping criterion.
                    stopprime = stopprime || stopprime2 || obj.stop_criterion(thetaminus, thetaplus, rminus, rplus);
                    % Update the acceptance probability statistics.
                    alphaprime = alphaprime + alphaprime2;
                    nalphaprime = nalphaprime + nalphaprime2;
                end
            end

        end
    end
end