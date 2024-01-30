#!/usr/bin/env Rscript
library(stringr)
args = commandArgs(trailingOnly=TRUE)
code = args[1]
train = read.csv(str_glue("temp/temp_{code}_train.csv"))
test = read.csv(str_glue("temp/temp_{code}_test.csv"))
source("GDSCic50.R")
mod = fitModel(train)
pred = predict(mod, test)
write.csv(data.frame(pred), str_glue("temp/temp_{code}_pred.csv"), row.names=FALSE)