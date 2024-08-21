function a = Fmvmt(b,nj)     % fast Fourier transform transpose for multiple rhs
  Nj = prod(nj); sNj = sqrt(Nj);       % scaling factor to make FFTN orthonormal
  nr = numel(b)/Nj;                        % number of right-hand-side arguments
  b = reshape(b,[nj(:)',nr]);
  for i=1:numel(nj), b = fft(b,[],i); end                         % emulate fftn
  a = reshape(b,Nj,[])/sNj;                                  % perform rescaling