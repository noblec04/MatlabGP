function [t, y] = feuler(odefun, ...   % The ODE function
                        y0, ...        % Initial states
                        ts, ...        % Start time
                        tf, ...        % End time
                        h, ...         % Step time
                        AD ...         % AutoDiff parameter set
                        )
t = ts:h:tf;

if nargin==7
    if isa(y0, 'AutoDiff')
        y{1} = y0;
    else
        y0_AD.values = y0;
        y0_AD.derivatives = 0*getderivs(AD);

        y0 = AutoDiff(y0_AD);

        y{1} = y0;
    end
else
    y{1} = y0;
end

for k = 2 : length(t)
    ydot = odefun(t(k), y{k-1});
    y{k} = y{k-1}+ydot.*h;
end
end