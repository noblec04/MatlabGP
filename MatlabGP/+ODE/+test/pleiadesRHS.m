function dqdt = pleiadesRHS(t,q)
  x = q(1:7);
  y = q(8:14);
  xDist = (x - x.');
  yDist = (y - y.');
  r = (xDist.^2+yDist.^2).^(3/2);
  m = (1:7)';

  A = xDist.*m./r;
  A(isnan(A))=0;

  B = yDist.*m./r;
  B(isnan(B))=0;

  dqdt = [q(15:28);
          sum(A).';
          sum(B).'];
end