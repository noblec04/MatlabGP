% Solve x=A*b with symmetric A(n,n), b(n,m), x(n,m) using conjugate gradients.
% The method is along the lines of PCG but suited for matrix inputs b.
function [x,flag,relres,iter,r] = conjgrad(A,b,tol,maxit)
if nargin<3, tol = 1e-10; end
if nargin<4, maxit = min(size(b,1),20); end
x0 = zeros(size(b)); x = x0;
if isnumeric(A), r = b-A*x; else r = b-A(x); end, r2 = sum(r.*r,1); r2new = r2;
nb = sqrt(sum(b.*b,1)); flag = 0; iter = 1;
relres = sqrt(r2)./nb; todo = relres>=tol; if ~any(todo), flag = 1; return, end
on = ones(size(b,1),1); r = r(:,todo); d = r;
for iter = 2:maxit
  if isnumeric(A), z = A*d; else z = A(d); end
  a = r2(todo)./sum(d.*z,1);
  a = on*a;
  x(:,todo) = x(:,todo) + a.*d;
  r = r - a.*z;
  r2new(todo) = sum(r.*r,1);
  relres = sqrt(r2new)./nb; cnv = relres(todo)<tol; todo = relres>=tol;
  d = d(:,~cnv); r = r(:,~cnv);                           % get rid of converged
  if ~any(todo), flag = 1; return, end
  b = r2new./r2;                                               % Fletcher-Reeves
  d = r + (on*b(todo)).*d;
  r2 = r2new;
end