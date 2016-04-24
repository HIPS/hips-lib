function x = vec(X)
% VEC -- unroll a matrix into a column vector.
%
% Given a matrix of size MxN it returns a vector of size MNx1. This
% is just a simple utility function that is essentially the same as
% Numpy's ravel() method.
%
  
x = X(:);
