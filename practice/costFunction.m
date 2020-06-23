function J=costFunction(x,y,theta)

m=size(x,1);
predications=x*theta;
sqrErrors=(predications-y).^2;

J=1/(2*m)*sum(sqrErrors);