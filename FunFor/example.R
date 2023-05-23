library(MASS)
library(caret)
library(refund)
source("R/FunFor.R")
nbx = 100
nbobs = 100
T = seq(0, 1, len = 100)
m = 1
rho = 0.6
mu = matrix(1, nbx, 1)
ar1_cor = function(n, m,rho) {
 exponent = abs(matrix(1:n - 1, nrow = n, ncol = n, byrow = TRUE) - (1:n - 1))
 L = rho^exponent
 diag(L) = m
 L
}

p = nbx
x_sigma = ar1_cor(p, m, rho)
noise_sigma = ar1_cor(length(T), (5 * cos(T) + rnorm(length(T), 0, 1)) / 10, 0.01)
beta_1 = function(t) sin(20 * pi * T) / 3

X = mvrnorm(nbx, mu, x_sigma)
X = as.data.frame(X)
Y = data.frame(matrix(NA, nrow = nbobs, ncol = length(T)))
for(j in 1:nbobs) Y[j, ] = (X[j, 2] * X[j, 3]) * beta_1(T) + mvrnorm(1, rep(0, length(T)), noise_sigma)

formula = paste("data[, 1:length(T)]", "~", paste(names(X), collapse = "+"))
data = cbind(Y, X)
funfor_fit = FunFor(formula, data, mtry = 40, ntree = 10, npc = 3, m_split = 15)