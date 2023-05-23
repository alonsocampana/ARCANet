##### Functions

#' Smoothing Function
#'
#' This function is a helper function called by make_split()
#' @param Y Y matrix
#' @param pve Proportion of variance explained
#' @param npc Number of PCs
#' @examples
#' eigsmooth(Y)
fpca2s = function (Y = NULL, ydata = NULL, argvals = NULL, npc = NA, center = TRUE, 
    smooth = TRUE) 
{
    stopifnot(!is.null(Y))
    if (any(is.na(Y))) 
        stop("No missing values in <Y> allowed.")
    if (!is.null(ydata)) {
        stop(paste("<ydata> argument for irregular data is not supported,", 
            "please use fpca.sc instead."))
    }
    X <- Y
    data_dim <- dim(X)
    I <- data_dim[1]
    J <- data_dim[2]
    if (is.na(npc)) {
        npc <- getNPC.DonohoGavish(X)
    }
    irregular <- FALSE
    if (!is.null(argvals)) {
        stopifnot(is.numeric(argvals), length(argvals) == J, 
            all(!is.na(argvals)))
        if (any(diff(argvals)/mean(diff(argvals)) > 1.05 | diff(argvals)/mean(diff(argvals)) < 
            0.95)) {
            warning(paste("non-equidistant <argvals>-grid detected:", 
                "fpca2s will return orthonormal eigenvectors of the function evaluations", 
                "not evaluations of the orthonormal eigenvectors.", 
                "Use fpca.sc() for the latter instead."))
            irregular <- TRUE
        }
    }
    else {
        argvals <- seq(0, 1, length = J)
    }
    meanX <- rep(0, J)
    if (center) {
        meanX <- apply(X, 2, function(x) mean(x, na.rm = TRUE))
        meanX <- smooth.spline(argvals, meanX, all.knots = TRUE, tol=0.000001)$y
        X <- t(t(X) - meanX)
    }
    if (J > I) {
        VV <- X %*% t(X)
        Eigen <- eigen(VV)
        D <- Eigen$values
        sel <- (D > 0)
        V <- Eigen$vectors[, sel == 1]
        D <- D[sel == 1]
        D <- sqrt(D)
        U <- t(X) %*% V %*% diag(1/D)
    }
    if (J <= I) {
        UU <- t(X) %*% X
        Eigen <- eigen(UU)
        D <- Eigen$values
        U <- Eigen$vectors[, D > 0]
        D <- D[D > 0]
        D <- sqrt(D)
    }
    lambda <- D^2/(I - 1)/J
    if (!is.numeric(npc)) 
        stop("Invalid <npc>.")
    if (npc < 1 | npc > min(I, J)) 
        stop("Invalid <npc>.")
    message("Extracted ", npc, " smooth components.")
    if (smooth == TRUE) {
        for (j in 1:npc) {
            temp = smooth.spline(argvals, U[, j], all.knots = TRUE, tol=0.000001)$y
            U[, j] = temp
        }
    }
    if (!irregular) {
        scale <- sqrt(mean(diff(argvals)))
    }
    else {
        scale <- 1
    }
    eigenvectors = U[, 1:npc, drop = FALSE]/scale
    scores = unname(t(lm.fit(x = eigenvectors, y = t(X))$coefficients))
    eigenvalues = diag(var(scores))
    Yhat = t(eigenvectors %*% t(scores) + meanX)
    ret = list(Yhat = Yhat, Y = Y, scores = scores, mu = meanX, 
        efunctions = eigenvectors, evalues = eigenvalues, npc = npc, 
        argvals = argvals)
    class(ret) = "fpca"
    return(ret)
}


eigsmooth = function(Y, pve = 0.9, npc = NULL) {
  if (is.null(npc)) {
    Fpca = suppressMessages(fpca2s(Y))
    varPer = cumsum(Fpca$evalues) / sum(Fpca$evalues)
    npc = min(which(varPer > pve))
  }
  Fpca = suppressMessages(fpca2s(Y, npc = npc))
  Yhat = data.frame(Fpca$Yhat)
  return(Yhat)
}


#' Calculate MISE
#'
#' This function is a helper function called by make_split()
#' @param vec A numeric vector

mise = function(vec) {
  mise = sum(vec^2 * (1 / (length(vec) - 1)))
  return(mise)
}


#' Selecting an optimal tree size
#'
#' This function is to find a optimal tree size
#' @details 
#' This function is used to determine the optimal size of a tree fit.
#' See ?FunFor, the main function, for example.
#' @param formula Formula of model fitted
#' @param data All the data
#' @param npc Number of PCs
#' @param n_folds Number of folds in cross-validation
#' @return An integer for optimal tree size.
#' @examples
#' optimal_size(Y ~ ., data)

optimal_size = function(formula, data, npc = NULL, n_folds = 5) {
  Xform = gsub(".*[~]", "", formula)
  Xs = unlist(strsplit(Xform, "+", fixed = TRUE))
  Xs = gsub("[ ]*", "", Xs)
  train_folds = createFolds(1:dim(data)[1], k = n_folds, list = TRUE, returnTrain = TRUE)
  splits = seq(10, 20, 5)
  cross_error = rep(0, length(splits))
  for (i_split in 1:length(splits)) {
    error = rep(0, length(train_folds))
    for (i in 1:length(train_folds)) {
      train_data = data[train_folds[[i]], ]
      test_data = data[!1:dim(data)[1] %in% train_folds[[i]], ]
      fit = mvTree(formula, train_data, npc = NULL, m_split = splits[i_split], smooth = TRUE)
      test_X = test_data[, names(test_data) %in% Xs]
      test_Y = test_data[, !names(test_data) %in% Xs]
      pred = predict.mvTree(fit, test_X)
      error_list = sapply(1:dim(pred)[1], function(i) mise(as.numeric(pred[i, ] - test_Y[i, ])))
      error[i] = mean(error_list, na.rm = TRUE)
    }
    cross_error[i_split] = mean(error, na.rm = TRUE)
  }
  res = splits[min(which.min(cross_error))]
  return(res)
}


#' Make a tree split
#'
#' This function is a helper function used to make a new split in a FunFor tree
#' @param data All the data
#' @param node Node to split
#' @param Xs Names of X variables
#' @param Ys Names of Y variables
#' @param MVT A matrix
#' @param ancestor Ancestor node
#' @param depth Maximum depth
#' @param npc Number of PCs
#' @param smooth Whether to smooth curves

make_split = function(data, node, Xs, Ys, MVT, ancestor, depth, pred_list, npc, m_split, smooth) {
  n = dim(data)[1]
  p = length(Xs)
  nSplit = length(MVT[, 3][!is.na(MVT[, 3])])
  if (n <= 20 || nSplit > m_split) {
    MVT_temp = data.frame(matrix(NA, nrow = 1, ncol = 5))
    names(MVT_temp) = c("node", "ancestor", "split", "split_val", "terminal")
    MVT_temp[1, 1:5] = c(node, ancestor, NA, NA, 1)
    MVT = rbind(MVT, MVT_temp)
    node = node + 1
    pred = apply(data[, names(data) %in% Ys], 2, mean)
    pred_list = c(pred_list, list(pred))
  } else {
    depth = depth + 1
    Y = data[, names(data) %in% Ys]
    X = data[, names(data) %in% Xs]
    allmax_phi = rep(0, p)
    split = list()
    par_out = foreach (i = 1:p, .combine = "c") %dopar% {
      mise = function(vec) {
      mise = sum(vec^2 * (1 / (length(vec) - 1)))
      return(mise)
        }
      if (is.factor(X[, i])) {
        levels = levels(X[, i])
        nlevel = length(levels)
        index = lapply(1:floor(nlevel / 2), function(x) combn(nlevel, x))
        phis = matrix(nrow = floor(nlevel / 2), ncol = max(unlist(lapply(index, function(x) length(x)))))
        for (j in 1:length(index)) {
          for (k in 1:dim(index[[j]])[2]){
            group = index[[j]][, k]
            l_Y = Y[X[, i] %in% levels[group], ]
            r_Y = Y[!X[, i] %in% levels[group], ]
            if (dim(l_Y)[1] <= 5 || dim(r_Y)[1] <= 5) {
              phis[j, k] = -Inf
            } else {
              if (smooth) {
                sl_Y = eigsmooth(as.matrix(l_Y), npc = npc)
                sr_Y = eigsmooth(as.matrix(r_Y), npc = npc)
              } else {
                sl_Y = as.matrix(l_Y)
                sr_Y = as.matrix(r_Y)
              }
              mu_l = apply(sl_Y, 2, mean)
              mu_r = apply(sr_Y, 2, mean)
              phis[j, k] = mise(as.numeric(mu_l - mu_r))
            }
          }
        }
        max_phi = max(phis, na.rm = TRUE)
        pos = which(phis == max_phi, arr.ind = TRUE)
        split[[i]] = index[[pos[1]]][, pos[2]]
      } else {
        cuts = seq(min(X[, i]), max(X[, i]), (max(X[, i]) - min(X[, i])) / 10)
        cuts = cuts[-c(1, 11)]
        phis = rep(0, 9)
        for (j in 1:9) {
          cut = cuts[j]
          # print(cut)
          # print(i)
          # print(j)
          # print(sum(X[, i] <= cut))
          # print(dim(Y))
          l_Y = Y[X[, i] <= cut, ]
          r_Y = Y[X[, i] > cut, ]
          if (dim(l_Y)[1] <= 5 || dim(r_Y)[1] <= 5) {
            phis[j] = -Inf
          } else {
            if (smooth) {
              sl_Y = eigsmooth(as.matrix(l_Y), npc = npc)
              sr_Y = eigsmooth(as.matrix(r_Y), npc = npc)
            } else {
              sl_Y = as.matrix(l_Y)
              sr_Y = as.matrix(r_Y)
            }
            mu_l = apply(sl_Y, 2, mean)
            mu_r = apply(sr_Y, 2, mean)
            phis[j] = mise(as.numeric(mu_l - mu_r))
          }
        }
        max_phi = max(phis, na.rm = TRUE)
        split[[i]] = min(cuts[which(phis == max_phi)])
      }
      return (list(max_phi, min(cuts[which(phis == max_phi)])))
    }
    allmax_phi = as.vector(unlist(par_out[seq(1,length(par_out)-1,2)]))
    split = as.vector(unlist(par_out[seq(2,length(par_out),2)]))                                                       
    best_split = min(which(allmax_phi == max(allmax_phi)))
    best_split_val = split[[best_split]]
    MVT_temp = data.frame(matrix(NA, nrow = 1, ncol = 5))
    names(MVT_temp) = c("node", "ancestor", "split", "split_val", "terminal")
    MVT_temp[1, 1:5] = c(node, ancestor, best_split, best_split_val, 0)
    MVT = rbind(MVT, MVT_temp)
    ancestor = node
    node = node + 1
    pred = apply(data[, names(data) %in% Ys], 2, mean)
    pred_list = c(pred_list, list(pred))
    if (is.factor(X[, MVT_temp[, 3]])) {
      levels = levels(X[, MVT_temp[, 3]])
      l_data = data[X[, MVT_temp[, 3]] %in% levels[MVT_temp[, 4]], ]
      r_data = data[!X[, MVT_temp[, 3]] %in% levels[MVT_temp[, 4]], ]
    } else {
      l_data = data[X[, MVT_temp[, 3]] <= MVT_temp[, 4], ]
      r_data = data[X[, MVT_temp[, 3]] > MVT_temp[, 4], ]
    }

    res = make_split(l_data, node, Xs, Ys, MVT, ancestor, depth, pred_list, npc, m_split, smooth)
    node = res$node
    MVT = res$MVT
    pred_list = res$pred_list

    res = make_split(r_data, node, Xs, Ys, MVT, ancestor, depth, pred_list, npc, m_split, smooth)
    node = res$node
    MVT = res$MVT
    pred_list = res$pred_list
  }

  res = list(node = node, MVT = MVT, pred_list = pred_list, Ys = Ys, Xs = Xs)
  return(res)
}


#' Fit an FunFor tree
#'
#' This function is used to fit an FunFor tree. It can be called independently or used in the main function FunFor().
#' @param formula Formula of model fitted
#' @param data All the data
#' @param npc Number of PCs
#' @param m_split Optimal tree size
#' @param smooth Whether to smooth curves
#' @examples
#' mvTree(Y ~ ., data, m_split = 20, smooth = TRUE)

mvTree = function(formula, data, npc = NULL, m_split, smooth) {
  Yform = gsub("[~].*", "", formula)
  Xform = gsub(".*[~]", "", formula)
  Xs = unlist(strsplit(Xform, "+", fixed = TRUE))
  Xs = gsub("[ ]*", "", Xs)
  Ys = names(data)[!names(data) %in% Xs]

  node = 1
  ancestor = 0
  depth = 1
  pred_list = list()
  MVT = data.frame(matrix(NA, nrow = 0, ncol = 5))
  names(MVT) = c("node", "ancestor", "split", "split_val", "terminal")
  res = make_split(data, node, Xs, Ys, MVT, ancestor, depth, pred_list,
                   npc = npc, m_split = m_split, smooth = smooth)

  class(res) = "mvTree"
  return(res)
}


#' A helper function for FunFor prediction
#'
#' This function is called by predict.mvTree() when making an prediction.
#' @param MVT A helper matrix
#' @param X X matrix
#' @param start.node Starting node

search_tree = function(MVT, X, start.node) {
  if (MVT[start.node, 5] == 1) {
    return (MVT[start.node, 1])
  } else{
    if (is.factor(X[, MVT[start.node, 3]])) {
      levels = levels(X[, MVT[start.node, 3]])
      if (X[, MVT[start.node, 3]] %in% levels[MVT[start.node, 4]]) {
        search_tree(MVT, X, min(which(MVT[, 2] == start.node)))
      } else {
        search_tree(MVT, X, max(which(MVT[, 2] == start.node)))
      }
    } else{
      if ((X[, MVT[start.node, 3]]) <= MVT[start.node, 4]) {
        search_tree(MVT, X, min(which(MVT[, 2] == start.node)))
      } else {
        search_tree(MVT, X, max(which(MVT[, 2] == start.node)))
      }
    }
  }
}


#' Make predictions based on a fitted tree
#'
#' This function is to make predictions based on a fitted tree.
#' @param fit Saved tree object
#' @param newdata A new X data matrix
#' @examples
#' predict(tree_fit, newX)

predict.mvTree = function(fit, newdata) {
  if (!inherits(fit, "mvTree")) print("Wrong Input!!")
  MVT = fit$MVT
  pred_list = fit$pred_list
  Ys = fit$Ys

  pred_frame = data.frame(matrix(NA, nrow = 0, ncol = length(Ys)))
  for (i in 1:dim(newdata)[1]) {
    X = newdata[i, ]
    start.node = 1
    node = search_tree(MVT, X, start.node)
    predict = pred_list[[node]]
    pred_frame = rbind(pred_frame, predict)
  }

  names(pred_frame) = Ys
  return (pred_frame)
}


#' Fit an FunFor model
#'
#' This is the main function for functional forest.
#' @param formula Formula of model fitted
#' @param data All the data
#' @param mtry Variables tried at each split
#' @param ntree Number of trees
#' @param importance Whether to calculate PVIM
#' @param npc A given number of PCs when smoothing
#' @param m_split Optimal tree size
#' @param smooth Whether to smooth curves
#' @return imp Importance measures for the candidate predictors
#' @return fits A list of the fitted trees. Will be used to make predictions.
#' @details 
#' The FunFor algorithm is able to predict curve responses for new observations and 
#' select important variables from a large set of scalar predictors. see ?optimal_size 
#' for how to determine the optimal size of a tree fit. See ?predict.mvRF for how to 
#' predict from new observations based on a fitted FunFor model.
#' @examples
#' library(MASS)
#' nbx = 100
#' nbobs = 100
#' T = seq(0, 1, len = 100)
#' m = 1
#' rho = 0.6
#' mu = matrix(1, nbx, 1)
#' ar1_cor = function(n, m,rho) {
#'  exponent = abs(matrix(1:n - 1, nrow = n, ncol = n, byrow = TRUE) - (1:n - 1))
#'  L = rho^exponent
#'  diag(L) = m
#'  L
#' }
#'
#' p = nbx
#' x_sigma = ar1_cor(p, m, rho)
#' noise_sigma = ar1_cor(length(T), (5 * cos(T) + rnorm(length(T), 0, 1)) / 10, 0.01)
#' beta_1 = function(t) sin(20 * pi * T) / 3
#'
#' X = mvrnorm(nbx, mu, x_sigma)
#' X = as.data.frame(X)
#' Y = data.frame(matrix(NA, nrow = nbobs, ncol = length(T)))
#' for(j in 1:nbobs) Y[j, ] = (X[j, 2] * X[j, 3]) * beta_1(T) + mvrnorm(1, rep(0, length(T)), noise_sigma)
#'
#' formula = paste("data[, 1:length(T)]", "~", paste(names(X), collapse = "+"))
#' data = cbind(Y, X)
#' o_split = optimal_size(formula, data)
#' funfor_fit = FunFor(formula, data, mtry = 40, ntree = 10, npc = 3, m_split = o_split)

FunFor = function(formula, data, mtry, ntree, importance = TRUE, npc = NULL, m_split = 10, smooth = TRUE,
                  ...) {
  my.cluster <- parallel::makeCluster(32)
  doParallel::registerDoParallel(cl = my.cluster)
  Y = gsub("[~].*", "", formula)
  X = gsub(".*[~]", "", formula)
  Xs = unlist(strsplit(X, "+", fixed = TRUE))
  Xs = gsub("[ ]*", "", Xs)
  Ys = names(data)[!names(data) %in% Xs]
  n = dim(data)[1]
  p = length(Xs)
  imp = matrix(NA, nrow = ntree, ncol = p)
  fits = list()
  for (i in 1:ntree) {
    print(i)
    print(Sys.time())
    bootSam = sort(sample(1:n, 0.25 * n, replace = FALSE))
    bootData = data[bootSam, ]
    bootVarIn = sort(sample(1:p, mtry, replace = FALSE))
    bootVar = Xs[bootVarIn]
    bootX = bootData[, names(bootData) %in% bootVar]
    bootY = bootData[, names(bootData) %in% Ys]
    bootData = cbind(bootY, bootX)
    bootXform = paste(bootVar, collapse = "+")
    sample_f = paste(Y, "~", bootXform, sep = "")
    fits[[i]] = mvTree(sample_f, data = bootData, npc = npc, m_split = m_split, smooth = smooth)
    # pre_X = data[-bootSam, names(data) %in% bootVar]
    # pre_Y = data[-bootSam, names(data) %in% Ys]
    # pre_pred = predict.mvTree(fits[[i]], pre_X)
    # pre_error = sapply(1:dim(pre_pred)[1], function(i) mise(as.numeric(pre_pred[i, ] - pre_Y[i, ])))
    # pre_error = sum(pre_error)
    # for (j in 1:mtry) {
    #   perOrder = sample(1:dim(pre_X)[1], dim(pre_X)[1], replace = FALSE)
    #   aft_X = pre_X
    #   aft_X[, j] = aft_X[, j][perOrder]
    #   aft_pred = predict.mvTree(fits[[i]], aft_X)
    #   aft_error = sapply(1:dim(aft_pred)[1], function(i) mise(as.numeric(aft_pred[i, ] - pre_Y[i, ])))
    #   aft_error = sum(aft_error)
    #   imp[i, bootVarIn[j]] = aft_error - pre_error
    # }
  }
  parallel::stopCluster(cl = my.cluster)
  #importance = apply(imp, 2, function(x) mean(x, na.rm = TRUE))
  #names(importance) = Xs

  result = list(fits = fits)
  class(result) = "mvRF"
  return(result)
}


#' Make predictions based on a FunFor fit
#'
#' This function is to make predictions based on a FunFor fit.
#' @details
#' This function is used to predict from new observations based on a fitted FunFor model.
#' See ?FunFor for examples of how to fit a FunFit model.
#' @param fit Saved FunFor object
#' @param newdata A new X data matrix
#' @return A matrix of predicted Y values.
#' @examples
#' predict(funfor_fit, newX)

predict.mvRF = function(fit, newdata) {
  if (!inherits(fit, "mvRF")) print("Wrong Input!!")
  fits = fit$fits
  pred_list = list()
  for (i in 1:length(fits)) {
    tree = fits[[i]]
    Ys = tree$Ys
    Xs = tree$Xs
    fitdata = newdata[, names(newdata) %in% c(Ys, Xs)]
    pred_list[[i]] = predict.mvTree(tree, fitdata)
  }
  pred_frame = data.frame(matrix(NA, nrow = dim(newdata)[1], ncol = length(Ys)))
  for (i in 1:dim(newdata)[1]) {
    temp = data.frame(matrix(NA, nrow = length(pred_list), ncol = length(Ys)))
    for (j in 1:length(pred_list)) {
      temp[j, ] = pred_list[[j]][i, ]
    }
    pred_frame[i, ] = apply(temp, 2, function(x) mean(x, na.rm = TRUE))
  }

  names(pred_frame) = Ys
  return (pred_frame)
}

