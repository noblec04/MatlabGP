function CheckAutoDiffJacobian(func, x, tol)
% check the function is deterministic

f1=func(x);
f2=func(x);
assert (all(size(f1)==size(f2)));
if ~isempty(f1)
    assert(maxabsdiff(f1, f2) == 0);
end
% check jacobian
[J2, f2] = AutoDiffJacobianFiniteDiff(func, x, 1:numel(x), []);
[J1, f1] = AutoDiffJacobianAutoDiff(func, x);

err_f = maxabsdiff(f1, f2);
if (err_f > tol)
    error('error %g greater then tolerance %g',err_f, tol)
end
assert(err_f < tol);
err_J = maxabsdiff(J1, J2);
if (err_J > tol)
    error('error %g greater then tolerance %g',err_J, tol)
end

function m = maxabsdiff(a, b)
diff = a - b;
m = max(abs(diff(:)));
