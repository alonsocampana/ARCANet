#!/usr/bin/env Rscript
library(stringr)
library(drc)
args = commandArgs(trailingOnly=TRUE)
code = args[1]
train = read.csv(str_glue("temp/temp_{code}_train.csv"))
test = read.csv(str_glue("temp/temp_{code}_test.csv"))
unique_pairs = train[!duplicated(train[, c(2, 3)]),c(2, 3)]
result_df = data.frame()
for (i in 1:nrow(unique_pairs))
    {
    pair = unique_pairs[i,]
    drug = pair[,1]
    cell = pair[,2]
    tryCatch({
        train_subset = train[train$drug == drug & train$CL == cell,]
        test_subset = test[test$drug == drug & test$CL == cell,]
        model = try(drm(y~x, data=train_subset, fct = LL.2(), logDose = 10, control = drmc(errorm=FALSE)))
        prediction = try(predict(model, test_subset))
        if("try-error" %in% class(prediction))
            {prediction = rep(0, nrow(test_subset))}
        y_pred = cbind(test_subset, prediction)
        model = try(drm(y~x, data=train_subset, fct = LL.3(), logDose=10, control = drmc(errorm=FALSE)))
        prediction = try(predict(model, test_subset))
        if("try-error" %in% class(prediction))
            {prediction = rep(0, nrow(test_subset))}
        y_pred = cbind(y_pred, prediction)
        model = try(drm(y~x, data=train_subset, fct = LL.4(), logDose=10, control = drmc(errorm=FALSE)))
        prediction = try(predict(model, test_subset))
        if("try-error" %in% class(prediction))
            {prediction = rep(0, nrow(test_subset))}
        y_pred = cbind(y_pred, prediction)
        model = try(drm(y~x, data=train_subset, fct = LL.5(), logDose=10, control = drmc(errorm=FALSE)))
        prediction = try(predict(model, test_subset))
        if("try-error" %in% class(prediction))
            {prediction = rep(0, nrow(test_subset))}
        y_pred = cbind(y_pred, prediction)
        result_df = rbind(result_df, y_pred)
        }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    }
write.csv(result_df, str_glue("temp/temp_{code}_pred.csv"), row.names=FALSE)
