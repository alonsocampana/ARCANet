# FunFor

An R package for the functional random forest (FunFor) algorithm. The FunFor algorithm is able to predict curve responses for new observations and select important variables from a large set of scalar predictors.

The package can be installed by running:

> devtools::install_github("xiaotiand/FunFor")

Then run 

> library(FunFor)
> 
> ?FunFor

to get an example.

Also, see ?optimal_size for how to determine the optimal size of a tree fit. See ?predict.mvRF for how to predict from new observations based on a fitted FunFor model.

The typical running time of a FunFor model on data with sample size n=100, number of scalar predictors p=100, and length of functional curves T=100 is about 276s (4.6min).

> system.time(FunFor(formula, data, mtry = 40, ntree = 10, npc = 3, m_split = 10))

> user  system elapsed 

> 272.636   2.296 275.159

Citations:

Fu, G., Dai, X., & Liang, Y. (2021). Functional random forests for curve response. Scientific Reports, 11(1), 1-14.
