% mode 14 (Lanczos/ARPACK)
function [Q,T] = lanczos_arpack(B,v,d)     % perform Lanczos with at most d MVMs
  mv = ver('matlab');                                   % get the Matlab version
  if numel(mv)==0, error('No ARPACK reverse communication in Octave.'), end
  mv = str2double(mv.Version); old = mv<8;% Matlab 7.14=R2012a has old interface
  [junk,maxArraySize] = computer; m32 = maxArraySize==(2^31-1); % we have 32bit?
  if m32, intstr = 'int32'; else intstr = 'uint64'; end   % according data types
  intconvert = @(x) feval(intstr,x);  % init and allocate depending on # of bits
  n = length(v);
  v = bsxfun(@times,v,1./sqrt(sum(v.*v,1)));               % avoid call to normc
  ido = intconvert(0); nev = intconvert(ceil((d+1)/2)); ncv = intconvert(d+1);
  ldv = intconvert(n); info = intconvert(1);
  lworkl = intconvert(int32(ncv)*(int32(ncv)+8));
  iparam = zeros(11,1,intstr); ipntr = zeros(15,1,intstr);
  if exist('arpackc_reset'), arpackc_reset(); end
  iparam([1,3,7]) = [1,300,1]; tol = 1e-10;
  Q = zeros(n,ncv); workd = zeros(n,3); workl = zeros(lworkl,1); count = 0;
  
  while ido~=99 && count<=d
    count = count+1;
    if old
                   arpackc('dsaupd',ido,'I',intconvert(n),'LM',nev,tol,v,...
                                ncv,Q,ldv,iparam,ipntr,workd,workl,lworkl,info);
    else
      [ido,info] = arpackc('dsaupd',ido,'I',intconvert(n),'LM',nev,tol,v,...
                                ncv,Q,ldv,iparam,ipntr,workd,workl,lworkl,info);
    end
    if info<0
      error(message('ARPACKroutineError',aupdfun,full(double(info))));
    end
    if ido == 1, inds = double(ipntr(1:3));
    else         inds = double(ipntr(1:2)); end
    rows = mod(inds-1,n)+1; cols = (inds-rows)/n+1; % referenced column of ipntr
    if ~all(rows==1), error(message('ipntrMismatchWorkdColumn',n)); end
    switch ido                       % reverse communication interface of ARPACK
      case -1, workd(:,cols(2)) = B(workd(:,cols(1)));
      case  1, workd(:,cols(3)) =   workd(:,cols(1));
               workd(:,cols(2)) = B(workd(:,cols(1)));
      case  2, workd(:,cols(2)) =   workd(:,cols(1));
      case 99                                                        % converged
      otherwise, error(message('UnknownIdo'));
    end
  end
  ncv = int32(ncv);
  Q = Q(:,1:ncv-1);                                            % extract results
  T = diag(workl(ncv+1:2*ncv-1))+diag(workl(2:ncv-1),1)+diag(workl(2:ncv-1),-1);