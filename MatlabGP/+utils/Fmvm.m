function b = Fmvm(a,nj)                % fast Fourier transform for multiple rhs
  Nj = prod(nj); sNj = sqrt(Nj);       % scaling factor to make FFTN orthonormal
  nr = numel(a)/Nj;                        % number of right-hand-side arguments
  b = a; b = reshape(b,[nj(:)',nr]);               % accumarray and target shape
  for i=1:numel(nj), b = ifft(b,[],i); end                       % emulate ifftn
  b = reshape(b,Nj,nr)*sNj;                                  % perform rescaling
