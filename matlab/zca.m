function [Y W] = zca(X, epsilon)
% ZCA: zero-phase whitening transform
%
% This function computes the linear transformation which results in an
% identify sample covariance matrix, while minimizing distortion.  An
% optional argument can be provided that specifies the regularization
% parameter, i.e., the weight given to the identity when computing the
% covariance matrix.  The function returns the transformed data, as
% well as the weight matrix used to compute it.  It is assumed that
% the data have a sample mean of zero.
% 
% [Y W] = zca(X, epsilon)
%
% X:       Data matrix, size N x D
% epsilon: regularization parameter, optional, defaults to 1e-6
% Y:       Output matrix, size N x D
% W:       Transformation matrix, size D x D
%
% The primary reference for ZCA is:
% A.J. Bell and T.J. Sejnowski. The "independent components" of
% natural scenes are edge filters.  Vision Research. 37(23):3327-38,
% 1997.
%
% This implementation basically follows the treatment given in:
% http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
%
% Author: Ryan P. Adams
% Copyright 2012, President and Fellows of Harvard College.
  
  if nargin == 1
    epsilon = 1e-6;
  end
  
  [N D] = size(X);
  
  % Compute the regularized scatter matrix.
  scatter = (X'*X + epsilon*eye(D));
  
  % The epsilon corresponds to virtual data.
  N = N + epsilon;
  
  % Take the eigendecomposition of the scatter matrix.
  [V D] = eig(scatter);
  
  % This is pretty hacky, but we don't want to divide by tiny
  % eigenvalues, so make sure they're all of reasonable size.
  D = max(diag(D), epsilon);
  
  % Now use the eigenvalues to find the root-inverse of the
  % scatter matrix.
  irD = diag(1./sqrt(D));
  
  % Reassemble into the transformation matrix.
  W = sqrt(N-1) * V * irD * V';
  
  % Apply to the data.
  Y = X*W;
  
  
end