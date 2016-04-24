function d2 = dist2(X1, X2)
% DIST2 -- Compute the squared distance between data matrices.
%
% Usage 1: dist2(X) where X is an NxD matrix. Returns the NxN matrix
% of squared distances.
%
% Usage 2: dist2(X1, X2) where X1 is an NxD matrix and X2 is an MxD
% matrix.  Returns an NxM matrix of squared distances.
%
% Copyright 2012, President and Fellows of Harvard University
% Author: Ryan P. Adams
%

  if nargin == 1
    X1sum = sum(X1.*X1, 2);
    d2 = max(-bsxfun(@minus, bsxfun(@minus, 2*X1*X1', X1sum), X1sum'),0);
  else
    X1sum = sum(X1.*X1, 2);
    X2sum = sum(X2.*X2, 2);
    d2 = max(-bsxfun(@minus, bsxfun(@minus, 2*X1*X2', X1sum), X2sum'),0);
  end
  
end