#' Package Title: Robust Prediction Package
#'
#' A brief (one-line) description of what the package does.
#'
#' A more detailed description of the package, its purpose, and the main
#' functionalities it provides. This section can include several lines.
#'
#' @docType package
#' @name RobustPrediction
#' @aliases RobustPrediction-package
#' @details
#' This package provides tools for building robust predictive models.
#' It includes functions for model training, validation, and prediction.
#'
#' @section Dependencies:
#' This package requires the following packages: \code{caret}, \code{randomForest}, \code{glmnet}.
#'
#' @examples
#' # Example usage:
#' data(iris)
#' model <- train_model(iris)
#' predictions <- predict_outcome(model, newdata)
#'
#' @references
#' Doe, J. (2024). \emph{Robust Predictive Modeling Techniques}. Journal of Machine Learning.
"_PACKAGE"



#' Tune and Train RobustTuneC Boosting
#'
#' This function tunes and trains a Boosting classifier with "RobustTuneC" method, using 5-fold cross-validation 
#' and selects the best model based on AUC.
#'
#' @param data Training data as a data frame. The first column should be the response variable.
#' @param dataext External validation data as a data frame. The first column should be the response variable.
#' @param mstop_seq A sequence of boosting iterations to consider. Default is a sequence starting at 5 and increasing by 5 each time, up to 1000.
#' @param nu Learning rate for the boosting algorithm. Default is 0.1.
#' @return A list containing the best number of boosting iterations (`best_mstop`) and the final trained model (`best_model`).
#' @import mboost
#' @import pROC
#' @export
#' 
#' @examples
#' \dontrun{
#' # Example usage:
#' data <- ... # your training data
#' dataext <- ... # your external validation data
#' mstop_seq <- seq(50, 500, by = 50)
#' result <- tuneandtrainRobustTuneCBoost(data, dataext, mstop_seq)
#' result$best_mstop
#' result$best_model
#' }
tuneandtrainRobustTuneCBoost <- function(data, dataext, mstop_seq = seq(5,1000, by = 5), nu = 0.1) {
  
  # library
  library(mboost)
  library(pROC)
  
  
  # 5-fold cross validation
  K <- 5
  
  n <- nrow(data)
  partition <- rep(1:K, length = n)
  partition <- partition[sample(n)]
  
  # initialize CV AUC matrix
  AUC_CV <- matrix(NA, nrow = length(mstop_seq), ncol = K)
  
  # CV
  for (j in 1:K) {
    XTrain <- data[partition != j,]
    XTest <- data[partition == j,]
    
    if (length(levels(as.factor(XTest[,1]))) == 1) {
      AUC_CV[,j] <- NA
    } else {
      # train the model
      fit_Boost_CV <- glmboost(x = as.matrix(XTrain[,2:ncol(XTrain)]), y = as.factor(XTrain[,1]),
                               family = Binomial(), control = boost_control(mstop = max(mstop_seq), nu = nu),
                               center = FALSE)
      
      for (i in 1:length(mstop_seq)) {
        # external validation
        pred_Boost_CV <- predict(fit_Boost_CV[mstop_seq[i]], newdata = XTest[,2:ncol(XTest)], type = "response")
        
        # 1-AUC
        AUC_CV[i,j] <- 1 - auc(response = XTest[,1], predictor = pred_Boost_CV[,1])
      }
    }
  }
  
  # average 1-AUC
  AUC_mean <- rowMeans(AUC_CV, na.rm = TRUE)
  cvmin <- min(AUC_mean, na.rm = TRUE)
  
  # define cseq
  cseq <- c(1, 1.1, 1.3, 1.5, 2)
  
  # choose best mstop
  AUC_Test.c <- numeric(length(cseq))
  
  done <- FALSE
  i <- 1
  
  while ((i <= length(cseq)) & !done) {
    if (cseq[i] * cvmin < 0.4) {
      mstop.c <- min(mstop_seq[which(AUC_mean <= cvmin * cseq[i])], na.rm = TRUE)
    } else {
      if (cvmin < 0.4) {
        mstop.c <- min(mstop_seq[which(AUC_mean <= 0.4)], na.rm = TRUE)
      } else {
        mstop.c <- min(mstop_seq[which(AUC_mean <= cvmin)], na.rm = TRUE)
      }
      done <- TRUE
    }
    
    pred_Boost_Test.c <- predict(fit_Boost_CV[mstop.c], newdata = dataext[,2:ncol(dataext)], type = "response")
    AUC_Test.c[i] <- auc(response = as.factor(dataext[,1]), predictor = pred_Boost_Test.c[,1])[1]
    
    i <- i + 1
  }
  
  nctried <- i - 1
  c <- cseq[max(which(AUC_Test.c[1:(i-1)] == max(AUC_Test.c[1:(i-1)])))]
  
  if (c * cvmin < 0.4) {
    mstop.c <- min(mstop_seq[which(AUC_mean <= cvmin * c)], na.rm = TRUE)
  } else if (cvmin < 0.4) {
    mstop.c <- min(mstop_seq[which(AUC_mean <= 0.4)], na.rm = TRUE)
  } else {
    mstop.c <- min(mstop_seq[which(AUC_mean <= cvmin)], na.rm = TRUE)
  }
  
  # train the final model
  final_model <- glmboost(x = as.matrix(data[,2:ncol(data)]), y = as.factor(data[,1]),
                          family = Binomial(), control = boost_control(mstop = mstop.c, nu = nu),
                          center = FALSE)
  
  # return the result
  res <- list(
    best_mstop = mstop.c,
    best_model = final_model
  )
  
  # class
  class(res) <- "RobustTuneCBoost"
  return (res)
}

#' Tune and Train RobustTuneC Lasso
#'
#' This function tunes and trains a Lasso classifier with "RobustTuneC" method, using 5-fold cross-validation 
#' and selects the best model based on AUC.
#'
#' @param data Training data as a data frame. The first column should be the response variable.
#' @param dataext External validation data as a data frame. The first column should be the response variable.
#' @param maxit Maximum number of iterations. Default is 120000.
#' @param nlambda The number of lambda values to use for cross-validation.
#' @return A list containing the best lambda (`best_lambda`) and the final trained model (`best_model`).
#' @export
tuneandtrainRobustTuneCLasso <- function(data, dataext, maxit = 120000, nlambda = 100) {
  
  # library
  library(glmnet)
  library(pROC)
  
  # Fit Lasso Model on training data
  fit_Lasso <- glmnet(x = as.matrix(data[,2:ncol(data)]), y = as.factor(data[,1]), 
                      family = "binomial", maxit = maxit, nlambda = nlambda, standardize = TRUE)
  # Get lambda sequence to use for CV
  lamseq <- fit_Lasso$lambda
  
  # Split Train in 5 parts
  K <- 5
  n <- nrow(data)
  
  partition <- rep(1:K, length = n)
  partition <- partition[sample(n)]
  
  # Cross Validation
  AUC_CV <- matrix(NA, nrow = length(lamseq), ncol = K)
  
  for (j in 1:K) {
    XTrain <- data[partition != j,]
    XTest <- data[partition == j,]
    
    if (length(levels(as.factor(XTest[,1]))) == 1) {
      AUC_CV[,j] <- NA
    } else {
      # Fit Lasso Model
      fit_Lasso_CV <- glmnet(x = as.matrix(XTrain[,2:ncol(XTrain)]), y = as.factor(XTrain[,1]), 
                             family = "binomial", maxit = maxit, lambda = lamseq, standardize = TRUE)
      
      # external validation
      pred_Lasso_CV <- predict(fit_Lasso_CV, newx = as.matrix(XTest[,2:ncol(XTest)]), s = lamseq, type = "response")
      
      # Determine 1-AUC to choose 'best' model
      for (i in 1:ncol(pred_Lasso_CV)) {
        AUC_CV[i,j] <- 1 - auc(response = XTest[,1], predictor = pred_Lasso_CV[,i])
      }
    }
  }
  
  # Mean of error (1-AUC) for each lambda
  AUC_mean <- rowMeans(AUC_CV, na.rm = TRUE)
  cvmin <- min(AUC_mean, na.rm = TRUE)
  
  # Which error (1-AUC) is minimal
  cseq = c(1, 1.1, 1.3, 1.5, 2)
  AUC_Test.c <- numeric(length(cseq))
  
  done <- FALSE
  i <- 1
  
  while ((i <= length(cseq)) & !done) {
    if (cseq[i] * cvmin < 0.4) {
      lambda.c <- max(lamseq[which(AUC_mean <= cvmin * cseq[i])], na.rm = TRUE)
    } else {
      if (cvmin < 0.4) {
        lambda.c <- max(lamseq[which(AUC_mean <= 0.4)], na.rm = TRUE)
      } else {
        lambda.c <- max(lamseq[which(AUC_mean <= cvmin)], na.rm = TRUE)
      }
      done <- TRUE
    }
    
    pred_Lasso <- predict(fit_Lasso, newx = as.matrix(dataext[,2:ncol(dataext)]), s = lambda.c, type = "response")
    AUC_Test.c[i] <- auc(response = as.factor(dataext[,1]), predictor = pred_Lasso[,1])
    
    i <- i + 1
  }
  
  nctried <- i - 1
  c <- cseq[max(which(AUC_Test.c[1:(i-1)] == max(AUC_Test.c[1:(i-1)])))]
  
  
  if (c * cvmin < 0.4) {
    lambda.c <- max(lamseq[which(AUC_mean <= cvmin * c)], na.rm = TRUE)
  } else if (cvmin < 0.4) {
    lambda.c <- max(lamseq[which(AUC_mean <= 0.4)], na.rm = TRUE)
  } else {
    lambda.c <- max(lamseq[which(AUC_mean <= cvmin)], na.rm = TRUE)
  }
  
  # train the final model
  final_model <- glmnet(x = as.matrix(data[,2:ncol(data)]), y = as.factor(data[,1]), 
                        family = "binomial", lambda = lambda.c, standardize = TRUE)
  
  # return the result
  res <- list(
    best_lambda = lambda.c,
    best_model = final_model
  )
  
  # set class
  class(res) <- "RobustTuneCLasso"
  return(res)
  
}

#' Tune and Train RobustTuneC Random Forest
#'
#' This function tunes and trains a Random Forest classifier with "RobustTuneC" method, using 5-fold cross-validation 
#' and selects the best model based on AUC.
#'
#' @param data Training data as a data frame. The first column should be the response variable.
#' @param dataext External validation data as a data frame. The first column should be the response variable.
#' @param maxit Maximum number of iterations. Default is 120000.
#' @param num.trees Number of trees to grow. Default is 500.
#' @return A list containing the best minimum node size (`best_min_node_size`) and the final trained model (`best_model`).
#' @export
tuneandtrainRobustTuneCRF <- function(data, dataext, num.trees = 500) {
  
  library(mlr)
  library(ranger)
  library(pROC)
  
  # Split Train in 5 parts
  K <- 5
  n <- nrow(data)
  data <- as.data.frame(data)  # Ensure data is a data frame
  dataext <- as.data.frame(dataext)  # Ensure dataext is a data frame
  
  partition <- rep(1:K, length = n)
  partition <- partition[sample(n)]
  
  # Initialize grid for min.node.size
  min.node.size_grid <- unique(round(exp(seq(log(1), log(nrow(data)-1), length=20))))
  
  # Cross Validation
  auc_CV <- matrix(NA, nrow = length(min.node.size_grid), ncol = K)
  
  for (j in 1:K) {
    XTrain <- data[partition != j, ]
    XTest <- data[partition == j, ]
    
    Combined_data <- rbind(XTrain, XTest)
    Combined_data <- as.data.frame(Combined_data)  # Ensure it's a data frame
    Combined_data$y <- as.factor(Combined_data$y)
    
    if (length(levels(as.factor(XTest[, 1]))) == 1) {
      auc_CV[, j] <- NA
    } else {
      
      for (i in 1:length(min.node.size_grid)) {
        # fit RF
        
        task = makeClassifTask(data = Combined_data, target = "y", check.data = FALSE)
        lrn = makeLearner("classif.ranger", predict.type = "prob", num.threads = 1, num.trees = num.trees, min.node.size = min.node.size_grid[i], save.memory = TRUE)
        
        train.set = 1:nrow(XTrain)
        test.set = (nrow(XTrain) + 1):nrow(Combined_data)
        
        model = train(lrn, task, subset = train.set)
        pred = predict(model, task = task, subset = test.set)
        
        auc_CV[i, j] <- 1 - performance(pred, measures = list(mlr::auc))
      }
    }
  }
  
  # Mean of error (1-AUC) for each min.node.size
  AUC_mean <- rowMeans(auc_CV, na.rm = TRUE)
  cvmin <- min(AUC_mean, na.rm = TRUE)
  
  # choose "best" min.node.size
  cseq = c(1, 1.1, 1.3, 1.5, 2)
  AUC_Test.c <- numeric(length(cseq))
  
  done <- FALSE
  i <- 1
  
  while ((i <= length(cseq)) & !done) {
    if (cseq[i] * cvmin < 0.4) {
      min.node.size.c <- min.node.size_grid[max(which(AUC_mean <= cvmin * cseq[i]), na.rm = TRUE)]
    } else {
      if (cvmin < 0.4) {
        min.node.size.c <- min.node.size_grid[max(which(AUC_mean <= 0.4), na.rm = TRUE)]
      } else {
        min.node.size.c <- min.node.size_grid[max(which(AUC_mean <= cvmin), na.rm = TRUE)]
      }
      done <- TRUE
    }
    
    CombinedTrainExtern <- rbind(data, dataext)
    CombinedTrainExtern <- as.data.frame(CombinedTrainExtern)  # Ensure it's a data frame
    CombinedTrainExtern$y <- as.factor(CombinedTrainExtern$y)
    train.set = 1:nrow(data)
    extern.set = (nrow(data) + 1):nrow(CombinedTrainExtern)
    
    task_Test = makeClassifTask(data = CombinedTrainExtern, target = "y", check.data = FALSE)
    lrn_Test.c = makeLearner("classif.ranger", predict.type = "prob", num.threads = 1, num.trees = num.trees, min.node.size = min.node.size.c, save.memory = TRUE)
    
    model_Test.c = train(lrn_Test.c, task_Test, subset = train.set)
    pred_Test.c = predict(model_Test.c, task = task_Test, subset = extern.set)
    
    AUC_Test.c[i] <- performance(pred_Test.c, measures = list(mlr::auc))
    
    i <- i + 1
  }
  
  nctried <- i - 1
  c <- cseq[max(which(AUC_Test.c[1:(i-1)] == max(AUC_Test.c[1:(i-1)])))]
  
  if (c * cvmin < 0.4) {
    min.node.size.c <- min.node.size_grid[max(which(AUC_mean <= cvmin * c), na.rm = TRUE)]
  } else if (cvmin < 0.4) {
    min.node.size.c <- min.node.size_grid[max(which(AUC_mean <= 0.4), na.rm = TRUE)]
  } else {
    min.node.size.c <- min.node.size_grid[max(which(AUC_mean <= cvmin), na.rm = TRUE)]
  }
  
  # train the final model
  data <- as.data.frame(data)  # Ensure data is a data frame
  data$y <- as.factor(data$y)  # Ensure the target variable is a factor
  final_model <- ranger(
    dependent.variable.name = "y", 
    data = data, 
    num.trees = num.trees, 
    min.node.size = min.node.size.c, 
    probability = TRUE
  )
  
  # return the result
  res <- list(
    best_min_node_size = min.node.size.c,
    best_model = final_model
  )
  
  # Set class
  class(res) <- "RobustTuneCRF"
  return(res)
}


#' Tune and Train RobustTuneC Ridge
#'
#' This function tunes and trains a Ridge classifier with "RobustTuneC" method, using 5-fold cross-validation 
#' and selects the best model based on AUC.
#'
#' @param data Training data as a data frame. The first column should be the response variable.
#' @param dataext External validation data as a data frame. The first column should be the response variable.
#' @param maxit Maximum number of iterations. Default is 120000.
#' @param nlambda The number of lambda values to use for cross-validation.
#' @return A list containing the best lambda (`best_lambda`) and the final trained model (`best_model`).
#' @export
tuneandtrainRobustTuneCRidge <- function(data, dataext, maxit = 120000, nlambda = 100) {
  
  # library
  library(glmnet)
  library(pROC)
  
  # Fit Ridge Model on training data
  fit_Ridge <- glmnet(x = as.matrix(data[, 2:ncol(data)]), y = as.factor(data[, 1]), 
                      family = "binomial", maxit = maxit, nlambda = nlambda, alpha = 0, standardize = TRUE)
  # Get lambda sequence to use for CV
  lamseq <- fit_Ridge$lambda
  
  # Split Train in 5 parts
  K <- 5
  n <- nrow(data)
  
  partition <- rep(1:K, length = n)
  partition <- partition[sample(n)]
  
  # Cross Validation
  AUC_CV <- matrix(NA, nrow = length(lamseq), ncol = K)
  
  for (j in 1:K) {
    XTrain <- data[partition != j, ]
    XTest <- data[partition == j, ]
    
    if (length(levels(as.factor(XTest[, 1]))) == 1) {
      AUC_CV[, j] <- NA
    } else {
      # Fit Ridge Model
      fit_Ridge_CV <- glmnet(x = as.matrix(XTrain[, 2:ncol(XTrain)]), y = as.factor(XTrain[, 1]), 
                             family = "binomial", maxit = maxit, lambda = lamseq, alpha = 0, standardize = TRUE)
      
      # Extern Validation
      pred_Ridge_CV <- predict(fit_Ridge_CV, newx = as.matrix(XTest[, 2:ncol(XTest)]), s = lamseq, type = "response")
      
      # Determine AUC to choose 'best' model
      for (i in 1:ncol(pred_Ridge_CV)) {
        AUC_CV[i, j] <- 1 - auc(response = XTest[, 1], predictor = pred_Ridge_CV[, i])
      }
    }
  }
  
  # Mean of error (1-AUC) for each lambda
  AUC_mean <- rowMeans(AUC_CV, na.rm = TRUE)
  # Which error (1-AUC) is minimal
  cvmin <- min(AUC_mean, na.rm = TRUE)
  
  
  cseq = c(1, 1.1, 1.3, 1.5, 2)
  AUC_Test.c <- numeric(length(cseq))
  
  done <- FALSE
  i <- 1
  
  while ((i <= length(cseq)) & !done) {
    if (cseq[i] * cvmin < 0.4) {
      lambda.c <- max(lamseq[which(AUC_mean <= cvmin * cseq[i])], na.rm = TRUE)
    } else {
      if (cvmin < 0.4) {
        lambda.c <- max(lamseq[which(AUC_mean <= 0.4)], na.rm = TRUE)
      } else {
        lambda.c <- max(lamseq[which(AUC_mean <= cvmin)], na.rm = TRUE)
      }
      done <- TRUE
    }
    
    pred_Ridge <- predict(fit_Ridge, newx = as.matrix(dataext[, 2:ncol(dataext)]), s = lambda.c, type = "response")
    AUC_Test.c[i] <- auc(response = as.factor(dataext[, 1]), predictor = pred_Ridge[, 1])
    
    i <- i + 1
  }
  
  nctried <- i - 1
  c <- cseq[max(which(AUC_Test.c[1:(i-1)] == max(AUC_Test.c[1:(i-1)])))]
  
  if (c * cvmin < 0.4) {
    lambda.c <- max(lamseq[which(AUC_mean <= cvmin * c)], na.rm = TRUE)
  } else if (cvmin < 0.4) {
    lambda.c <- max(lamseq[which(AUC_mean <= 0.4)], na.rm = TRUE)
  } else {
    lambda.c <- max(lamseq[which(AUC_mean <= cvmin)], na.rm = TRUE)
  }
  
  # train the final model
  final_model <- glmnet(x = as.matrix(data[, 2:ncol(data)]), y = as.factor(data[, 1]), 
                        family = "binomial", lambda = lambda.c, alpha = 0, standardize = TRUE)
  
  # Return result:
  res <- list(
    best_lambda = lambda.c,
    best_model = final_model
  )
  
  # Set class
  class(res) <- "RobustTuneCRidge"
  return(res)
  
}


#' Tune and Train RobustTuneC Support Vector Machine(SVM)
#'
#' This function tunes and trains a SVM classifier with "RobustTuneC" method, using 5-fold cross-validation 
#' and selects the best model based on AUC.
#'
#' @param data Training data as a data frame. The first column should be the response variable.
#' @param dataext External validation data as a data frame. The first column should be the response variable.
#' @param seed Random seed for reproducibility. Default is 123.
#' @param kernel The kernel type to be used in the algorithm. It can be "linear", "polynomial", "radial", or "sigmoid". Default is "linear".
#' @param cost_seq A sequence of cost values to consider for cross-validation. Default is `2^(-15:15)`.
#' @param scale A logical vector indicating the variables to be scaled. Default is `FALSE`.
#' @return A list containing the best cost (`best_cost`) and the final trained model (`best_model`).
#' @export
tuneandtrainRobustTuneCSVM <- function(data, dataext, seed = 123, kernel = "linear", cost_seq = 2^(-15:15), scale = FALSE) {
  
  # library
  library(mlr)
  library(e1071)
  library(pROC)
  
  # Split Train in 5 parts
  K <- 5
  n <- nrow(data)
  
  partition <- rep(1:K, length = n)
  partition <- partition[sample(n)]
  
  # Cross Validation
  auc_CV <- matrix(NA, nrow = length(cost_seq), ncol = K)
  
  for (j in 1:K) {
    XTrain <- data[partition != j, ]
    XTest <- data[partition == j, ]
    
    Combined_data <- rbind(XTrain, XTest)
    Combined_data <- as.data.frame(Combined_data)  # Ensure it's a data frame
    Combined_data$y <- as.factor(Combined_data$y)
    
    if (length(levels(as.factor(XTest[, 1]))) == 1) {
      auc_CV[, j] <- NA
    } else {
      i <- 1
      for (c in cost_seq) {
        # Fit SVM
        set.seed(seed)
        
        task <- makeClassifTask(data = Combined_data, target = "y", check.data = FALSE)
        lrn <- makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = c, scale = scale)
        
        train.set <- 1:nrow(XTrain)
        test.set <- (nrow(XTrain) + 1):nrow(Combined_data)
        
        model <- train(lrn, task, subset = train.set)
        pred <- predict(model, task = task, subset = test.set)
        
        auc_CV[i, j] <- 1 - performance(pred, measures = list(mlr::auc))
        
        i <- i + 1
      }
    }
  }
  
  # Mean of error (1-AUC) for each cost value
  AUC_mean <- rowMeans(auc_CV, na.rm = TRUE)
  cvmin <- min(AUC_mean, na.rm = TRUE)
  
  # choose c
  cseq = c(1, 1.1, 1.3, 1.5, 2)
  AUC_Test.c <- numeric(length(cseq))
  
  done <- FALSE
  i <- 1
  
  while ((i <= length(cseq)) & !done) {
    if (cseq[i] * cvmin < 0.4) {
      cost.c <- min(cost_seq[which(AUC_mean <= cvmin * cseq[i])], na.rm = TRUE)
    } else {
      if (cvmin < 0.4) {
        cost.c <- min(cost_seq[which(AUC_mean <= 0.4)], na.rm = TRUE)
      } else {
        cost.c <- min(cost_seq[which(AUC_mean <= cvmin)], na.rm = TRUE)
      }
      done <- TRUE
    }
    
    CombinedTrainExtern <- rbind(data, dataext)
    CombinedTrainExtern <- as.data.frame(CombinedTrainExtern)  # Ensure it's a data frame
    CombinedTrainExtern$y <- as.factor(CombinedTrainExtern$y)
    train.set <- 1:nrow(data)
    extern.set <- (nrow(data) + 1):nrow(CombinedTrainExtern)
    
    task_Test <- makeClassifTask(data = CombinedTrainExtern, target = "y", check.data = FALSE)
    lrn_Test.c <- makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = cost.c, scale = scale)
    
    model_Test.c <- train(lrn_Test.c, task_Test, subset = train.set)
    pred_Test.c <- predict(model_Test.c, task = task_Test, subset = extern.set)
    
    AUC_Test.c[i] <- performance(pred_Test.c, measures = list(mlr::auc))
    
    i <- i + 1
  }
  
  nctried <- i - 1
  c <- cseq[max(which(AUC_Test.c[1:(i-1)] == max(AUC_Test.c[1:(i-1)])))]
  
  if (c * cvmin < 0.4) {
    cost.c <- min(cost_seq[which(AUC_mean <= cvmin * c)], na.rm = TRUE)
  } else if (cvmin < 0.4) {
    cost.c <- min(cost_seq[which(AUC_mean <= 0.4)], na.rm = TRUE)
  } else {
    cost.c <- min(cost_seq[which(AUC_mean <= cvmin)], na.rm = TRUE)
  }
  
  # train the final model
  data <- as.data.frame(data)  # Ensure data is a data frame
  data$y <- as.factor(data$y)  # Ensure the target variable is a factor
  
  final_task <- makeClassifTask(data = data, target = "y", check.data = FALSE)
  final_learner <- makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = cost.c, scale = scale)
  final_model <- train(final_learner, final_task)
  
  # return result
  res <- list(
    best_cost = cost.c,
    best_model = final_model
  )
  
  # Set class
  class(res) <- "RobustTuneCSVM"
  return(res)
}

#' Tune and Train by tuning method RobustTuneC
#'
#' This function tunes and trains a specified classifier using the appropriate method and data.
#'
#' @param data Training data as a data frame. The first column should be the response variable.
#' @param dataext External validation data as a data frame. The first column should be the response variable.
#' @param classifier The classifier to use. Must be one of 'boosting', 'rf', 'lasso', 'ridge', 'svm'.
#' @param ... Additional arguments to pass to the specific classifier function.
#' @return A list containing the results from the specific classifier tuning and training function.
#' @export
tuneandtrainRobustTuneC <- function(data, dataext, classifier, ...) {
  
  # arguments
  #args <- list(...)
  
  # run function by classifer
  if (classifier == "boosting") {
    res <- tuneandtrainRobustTuneCBoost(data = data, dataext = dataext, ...)
  } else if (classifier == "rf") {
    res <- tuneandtrainRobustTuneCRF(data = data, dataext = dataext, ...)
  } else if (classifier == "lasso") {
    res <- tuneandtrainRobustTuneCLasso(data = data, dataext = dataext, ...)
  } else if (classifier == "ridge") {
    res <- tuneandtrainRobustTuneCRidge(data = data, dataext = dataext, ...)
  } else if (classifier == "svm") {
    res <- tuneandtrainRobustTuneCSVM(data = data, dataext = dataext, ...)
  } else {
    stop("Unsupported classifier type. Choose from 'boosting', 'rf', 'lasso', 'ridge', 'svm'.")
  }
  
  return(res)
}

#' Tune and Train External Boosting
#'
#' This function tunes and trains a Boosting classifier with using an external validation dataset.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param mstop_seq A numeric vector of boosting iterations to be evaluated. Default is a sequence starting at 5 and increasing by 5 each time, up to 1000.
#' @param nu A numeric value for the learning rate. Default is 0.1.
#'
#' @return A list containing the best number of boosting iterations (`best_mstop`) and the final trained model (`best_model`).
#' @import mboost
#' @import pROC
#' @export
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' data <- ... # your training data
#' dataext <- ... # your external validation data
#' mstop_seq <- seq(50, 500, by = 50)
#' result <- tuneandtrainExt(data, dataext, mstop_seq)
#' result$best_mstop
#' result$best_model
#' }
tuneandtrainExtBoost <- function(data, dataext, mstop_seq = seq(5, 1000, by = 5), nu = 0.1) {
  # Load necessary libraries
  library(mboost)
  library(pROC)
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  # Fit initial boosting model
  fit_Boost <- glmboost(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]),
                        family = Binomial(), control = boost_control(mstop = max(mstop_seq), nu = nu),
                        center = FALSE)
  
  AUC <- numeric(length(mstop_seq))
  
  # External validation
  for (i in seq_along(mstop_seq)) {
    mseq <- mstop_seq[i]
    # Prediction on external data
    pred_Boost <- predict(fit_Boost[mseq], newdata = as.matrix(Extern[, -1]), type = "response")
    # Calculate AUC
    AUC[i] <- auc(response = as.factor(Extern[, 1]), predictor = pred_Boost)
  }
  
  # Choose the best mstop
  chosen_model <- which.max(AUC)
  chosen_mstop <- mstop_seq[chosen_model]
  
  # Train the final model with the chosen mstop
  final_model <- glmboost(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]),
                          family = Binomial(), control = boost_control(mstop = chosen_mstop, nu = nu),
                          center = FALSE)
  
  # Return the result
  res <- list(
    best_mstop = chosen_mstop,
    best_model = final_model
  )
  
  # Set class
  class(res) <- "ExtBoost"
  return(res)
}


#' Tune and Train External Lasso
#'
#' This function tunes and trains a Lasso classifier using an external validation dataset.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param maxit An integer specifying the maximum number of iterations. Default is 120000.
#' @param nlambda An integer specifying the number of lambda values to use in the Lasso model.
#'
#' @return A list containing the best lambda value (`best_lambda`), the final trained model (`best_model`), the AUC on the training data (`AUC_Train`), and the number of active coefficients (`active_set_Train`).
#' @import mboost
#' @import pROC
#' @export
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' data <- ... # your training data
#' dataext <- ... # your external validation data
#' result <- tuneandtrainExtLasso(data, dataext)
#' result$best_lambda
#' result$best_model
#' result$AUC_Train
#' result$active_set_Train
#' }

tuneandtrainExtLasso <- function(data, dataext, maxit = 120000, nlambda = 100) {
  # Load necessary libraries
  library(glmnet)
  library(pROC)
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  # Fit Lasso Model
  fit_Lasso <- glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                      family = "binomial", maxit = maxit, nlambda = nlambda, standardize = TRUE)
  
  # External Validation
  pred_Lasso <- predict(fit_Lasso, newx = as.matrix(Extern[, -1]), s = fit_Lasso$lambda, type = "response")
  
  # Determine AUC to choose 'best' model
  AUC <- numeric(ncol(pred_Lasso))
  
  for (i in seq_along(AUC)) {
    AUC[i] <- auc(response = as.factor(Extern[, 1]), predictor = pred_Lasso[, i])
  }
  
  chosen_model <- which.max(AUC)
  chosen_lambda <- fit_Lasso$lambda[chosen_model]
  coef_active_a <- coef(fit_Lasso, s = chosen_lambda)
  active_set_a <- length(coef_active_a@x)
  
  # Determine AUC of the chosen model on the Training dataset
  pred_Lasso_Train <- predict(fit_Lasso, newx = as.matrix(Train[, -1]), s = chosen_lambda, type = "response")
  AUC_Train <- auc(response = as.factor(Train[, 1]), predictor = as.numeric(pred_Lasso_Train))
  
  # Train the final model with the chosen lambda
  final_model <- glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                        family = "binomial", maxit = maxit, lambda = chosen_lambda, standardize = TRUE)
  
  # Return the result
  res <- list(
    best_lambda = chosen_lambda,
    best_model = final_model,
    AUC_Train = AUC_Train,
    active_set_Train = active_set_a
  )
  
  # Set class
  class(res) <- "ExtLasso"
  return(res)
}


#' Tune and Train External Random Forest
#'
#' This function tunes and trains a Random Forest classifier using an external validation dataset.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param num.trees An integer specifying the number of trees in the Random Forest. Default is 500.
#'
#' @return A list containing the best `min.node.size` value and the final trained model (`best_model`).
#' @import ranger
#' @import mlr
#' @import pROC
#' @export
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' data <- ... # your training data
#' dataext <- ... # your external validation data
#' result <- tuneandtrainExtRF(data, dataext, num.trees = 500)
#' result$best_min.node.size
#' result$best_model
#' }
tuneandtrainExtRF <- function(data, dataext, num.trees = 500) {
  # Load necessary libraries
  library(ranger)
  library(mlr)
  library(pROC)
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  Combined_data <- rbind(Train, Extern)
  Combined_data[, 1] <- as.factor(Combined_data[, 1])
  
  # Initialize AUC vector
  auc_value <- numeric(nrow(Train) - 1)
  
  # Tune min.node.size parameter
  for (i in 1:(nrow(Train) - 1)) {
    # Fit Random Forest model
    task <- makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
    lrn <- makeLearner("classif.ranger", predict.type = "prob", num.threads = 1, num.trees = num.trees, min.node.size = i, save.memory = TRUE)
    
    train.set <- 1:nrow(Train)
    test.set <- (nrow(Train) + 1):nrow(Combined_data)
    
    model <- train(lrn, task, subset = train.set)
    pred <- predict(model, task = task, subset = test.set)
    
    auc_value[i] <- performance(pred, measures = list(mlr::auc))
  }
  
  chosen_min.node.size <- which.max(auc_value)
  
  # Train the final model with the chosen min.node.size
  final_task <- makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
  final_lrn <- makeLearner("classif.ranger", predict.type = "prob", num.trees = num.trees, min.node.size = chosen_min.node.size, save.memory = TRUE)
  
  final_model <- train(final_lrn, final_task, subset = 1:nrow(Train))
  
  # Return the result
  res <- list(
    best_min.node.size = chosen_min.node.size,
    best_model = final_model
  )
  
  # Set class
  class(res) <- "ExtRF"
  return(res)
}


#' Tune and Train External Ridge Regression
#'
#' This function tunes and trains a Ridge Regression classifier using an external validation dataset.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param maxit An integer specifying the maximum number of iterations. Default is 120000.
#' @param nlambda An integer specifying the number of lambda values to use in the Ridge model.
#'
#' @return A list containing the best lambda value (`best_lambda`), the final trained model (`best_model`), the AUC on the training data (`AUC_Train`).
#' @import glmnet
#' @import pROC
#' @export
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' data <- ... # your training data
#' dataext <- ... # your external validation data
#' result <- tuneandtrainExtRidge(data, dataext)
#' result$best_lambda
#' result$best_model
#' result$AUC_Train
#' }

tuneandtrainExtRidge <- function(data, dataext, maxit = 120000, nlambda = 100) {
  # Load necessary libraries
  library(glmnet)
  library(pROC)
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  # Fit Ridge Regression Model
  fit_Ridge <- glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                      family = "binomial", maxit = maxit, nlambda = nlambda, alpha = 0, standardize = TRUE)
  
  # External Validation
  pred_Ridge <- predict(fit_Ridge, newx = as.matrix(Extern[, -1]), type = "response")
  
  # Determine AUC to choose 'best' model
  AUC <- numeric(ncol(pred_Ridge))
  
  for (i in seq_along(AUC)) {
    AUC[i] <- auc(response = as.factor(Extern[, 1]), predictor = as.numeric(pred_Ridge[, i]))
  }
  
  chosen_model <- which.max(AUC)
  chosen_lambda <- fit_Ridge$lambda[chosen_model]
  
  # Determine AUC of the chosen model on the Training dataset
  pred_Ridge_Train <- predict(fit_Ridge, newx = as.matrix(Train[, -1]), s = chosen_lambda, type = "response")
  AUC_Train <- auc(response = as.factor(Train[, 1]), predictor = as.numeric(pred_Ridge_Train))
  
  # Train the final model with the chosen lambda
  final_model <- glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                        family = "binomial", maxit = maxit, lambda = chosen_lambda, alpha = 0, standardize = TRUE)
  
  # Return the result
  res <- list(
    best_lambda = chosen_lambda,
    best_model = final_model,
    AUC_Train = AUC_Train
  )
  
  # Set class
  class(res) <- "ExtRidge"
  return(res)
}

#' Tune and Train External SVM
#'
#' This function tunes and trains an SVM classifier using an external validation dataset.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param kernel A character string specifying the kernel type to be used in the SVM. Default is "linear".
#' @param cost_seq A numeric vector of cost values to be evaluated. Default is `2^(-15:15)`.
#' @param scale A logical indicating whether to scale the predictor variables. Default is FALSE.
#'
#' @return A list containing the best cost value (`best_cost`), the final trained model (`best_model`), and the AUC on the external validation data (`AUC_Extern`).
#' @import e1071
#' @import mlr
#' @import pROC
#' @export
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' data <- ... # your training data
#' dataext <- ... # your external validation data
#' result <- tuneandtrainExtSVM(data, dataext, kernel = "linear", cost_seq = 2^(-15:15), scale = FALSE)
#' result$best_cost
#' result$best_model
#' result$AUC_Extern
#' }
tuneandtrainExtSVM <- function(data, dataext, kernel = "linear", cost_seq = 2^(-15:15), scale = FALSE) {
  # Load necessary libraries
  library(e1071)
  library(mlr)
  library(pROC)
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  Combined_data <- rbind(Train, Extern)
  Combined_data[, 1] <- as.factor(Combined_data[, 1])
  
  # Initialize AUC vector
  auc_value <- numeric(length(cost_seq))
  
  # Define AUC measure from mlr package
  auc_measure <- mlr::auc
  
  # Tune cost parameter
  for (i in seq_along(cost_seq)) {
    cost <- cost_seq[i]
    
    # Fit SVM model
    task <- makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
    lrn <- makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = cost, scale = scale)
    
    train.set <- 1:nrow(Train)
    test.set <- (nrow(Train) + 1):nrow(Combined_data)
    
    model <- train(lrn, task, subset = train.set)
    pred <- predict(model, task = task, subset = test.set)
    
    # Corrected line
    auc_value[i] <- performance(pred, measures = auc_measure)
  }
  
  # Choose the best cost
  chosen_cost <- cost_seq[which.max(auc_value)]
  
  # Train the final model with the chosen cost
  final_task <- makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
  final_lrn <- makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = chosen_cost, scale = scale)
  
  final_model <- train(final_lrn, final_task, subset = 1:nrow(Train))
  
  # Predict on the external validation data
  pred_Extern <- predict(final_model, task = final_task, subset = (nrow(Train) + 1):nrow(Combined_data))
  
  # Corrected line: ensure 'measures' is a list
  AUC_Extern <- performance(pred_Extern, measures = list(mlr::auc))
  
  # Return the result
  res <- list(
    best_cost = chosen_cost,
    best_model = final_model,
    AUC_Extern = AUC_Extern
  )
  
  # Set class
  class(res) <- "ExtSVM"
  return(res)
}



#' Tune and Train by tuning method Ext
#'
#' This function tunes and trains a specified classifier using the appropriate method and data.
#'
#' @param data Training data as a data frame. The first column should be the response variable.
#' @param dataext External validation data as a data frame. The first column should be the response variable.
#' @param classifier The classifier to use. Must be one of 'boosting', 'rf', 'lasso', 'ridge', 'svm'.
#' @param ... Additional arguments to pass to the specific classifier function.
#' @return A list containing the results from the specific classifier tuning and training function.
#' @export
tuneandtrainExt <- function(data, dataext, classifier, ...) {
  
  # arguments
  #args <- list(...)
  
  # run function by classifer
  if (classifier == "boosting") {
    res <- tuneandtrainExtBoost(data = data, dataext = dataext, ...)
  } else if (classifier == "rf") {
    res <- tuneandtrainExtRF(data = data, dataext = dataext, ...)
  } else if (classifier == "lasso") {
    res <- tuneandtrainExtLasso(data = data, dataext = dataext, ...)
  } else if (classifier == "ridge") {
    res <- tuneandtrainExtRidge(data = data, dataext = dataext, ...)
  } else if (classifier == "svm") {
    res <- tuneandtrainExtSVM(data = data, dataext = dataext, ...)
  } else {
    stop("Unsupported classifier type. Choose from 'boosting', 'rf', 'lasso', 'ridge', 'svm'.")
  }
  
  return(res)
}

#' Tune and Train Internal Boosting
#'
#' This function tunes and trains a Boosting classifier using internal cross-validation.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param mstop_seq A numeric vector of boosting iterations to be evaluated.
#' @param nu A numeric value for the learning rate. Default is 0.1.
#'
#' @return A list containing the best number of boosting iterations (`best_mstop`) and the AUC on the test data (`AUC_Test`).
#' @import mboost
#' @import pROC
#' @export
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' data <- ... # your training data
#' mstop_seq <- seq(5, 5000, by = 5)
#' result <- tuneandtrainIntBoost(data, mstop_seq, nu = 0.1)
#' result$best_mstop
#' result$AUC_Test
#' }
tuneandtrainIntBoost <- function(data, mstop_seq = seq(5,1000, by = 5), nu = 0.1) {
  # Load necessary libraries
  library(mboost)
  library(pROC)
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  
  Train <- data
  
  # Fit initial boosting model
  fit_Boost <- glmboost(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]),
                        family = Binomial(), control = boost_control(mstop = max(mstop_seq), nu = nu),
                        center = FALSE)
  
  # Split Train in 5 parts
  K <- 5
  n <- nrow(Train)
  
  partition <- rep(1:K, length = n)
  partition <- partition[sample(n)]
  
  # Cross Validation
  AUC_CV <- matrix(NA, nrow = length(mstop_seq), ncol = K)
  
  for (j in 1:K) {
    XTrain <- Train[partition != j, ]
    XTest <- Train[partition == j, ]
    
    if (length(levels(as.factor(XTest[, 1]))) == 1) {
      AUC_CV[, j] <- NA
    } else {
      # Fit Boosting Model
      fit_Boost_CV <- glmboost(x = as.matrix(XTrain[, -1]), y = as.factor(XTrain[, 1]),
                               family = Binomial(), control = boost_control(mstop = max(mstop_seq), nu = nu),
                               center = FALSE)
      
      for (i in seq_along(mstop_seq)) {
        mseq <- mstop_seq[i]
        # Internal Validation
        pred_Boost_CV <- predict(fit_Boost_CV[mseq], newdata = as.matrix(XTest[, -1]), type = "response")
        
        # Determine AUC to choose 'best' model
        AUC_CV[i, j] <- auc(response = as.factor(XTest[, 1]), predictor = pred_Boost_CV)
      }
    }
  }
  
  AUC <- rowMeans(AUC_CV, na.rm = TRUE)
  chosen_mstop <- mstop_seq[which.max(AUC)]
  
  # Train the final model with the chosen mstop
  final_model <- glmboost(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]),
                          family = Binomial(), control = boost_control(mstop = chosen_mstop, nu = nu),
                          center = FALSE)
  
  # Predict on the whole Train dataset using the optimal mstop value
  pred_Boost_Train <- predict(final_model, newdata = as.matrix(Train[, -1]), type = "response")
  AUC_Train <- auc(response = as.factor(Train[, 1]), predictor = pred_Boost_Train)
  
  # Return the result
  res <- list(
    best_mstop = chosen_mstop,
    AUC_Train = AUC_Train
  )
  
  # Set class
  class(res) <- "IntBoost"
  return(res)
}

#' Tune and Train Internal Lasso
#'
#' This function tunes and trains a Lasso classifier using internal cross-validation.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param maxit An integer specifying the maximum number of iterations. Default is 120000.
#' @param nlambda An integer specifying the number of lambda values to use in the Lasso model. Default is 200.
#' @param nfolds An integer specifying the number of folds for cross-validation. Default is 5.
#'
#' @return A list containing the best lambda value (`best_lambda`), the final trained model (`best_model`), the AUC on the training data (`AUC_Train`), and the number of active coefficients (`active_set_Train`).
#' @import glmnet
#' @import pROC
#' @export
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' data <- ... # your training data
#' result <- tuneandtrainIntLasso(data)
#' result$best_lambda
#' result$best_model
#' result$AUC_Train
#' result$active_set_Train
#' }
tuneandtrainIntLasso <- function(data, maxit = 120000, nlambda = 200, nfolds = 5) {
  # Load necessary libraries
  library(glmnet)
  library(pROC)
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  
  # Split data into predictors and response
  X <- as.matrix(data[, -1])
  y <- as.factor(data[, 1])
  
  # Fit initial Lasso model to obtain lambda sequence
  fit_Lasso <- glmnet(x = X, y = y, family = "binomial", maxit = maxit, nlambda = nlambda, standardize = TRUE)
  lamseq <- fit_Lasso$lambda
  
  # Cross-validation
  partition <- sample(rep(1:nfolds, length.out = nrow(data)))
  AUC_CV <- matrix(NA, nrow = nlambda, ncol = nfolds)
  
  for (j in 1:nfolds) {
    XTrain <- X[partition != j, , drop = FALSE]
    yTrain <- y[partition != j]
    XTest <- X[partition == j, , drop = FALSE]
    yTest <- y[partition == j]
    
    if (length(unique(yTest)) == 1) {
      AUC_CV[, j] <- NA
    } else {
      fit_Lasso_CV <- glmnet(x = XTrain, y = yTrain, family = "binomial", maxit = maxit, lambda = lamseq, standardize = TRUE)
      pred_Lasso_CV <- predict(fit_Lasso_CV, newx = XTest, s = lamseq, type = "response")
      
      for (i in 1:ncol(pred_Lasso_CV)) {
        AUC_CV[i, j] <- auc(response = yTest, predictor = pred_Lasso_CV[, i])
      }
    }
  }
  
  # Determine the best lambda based on the highest average AUC
  mean_AUC <- rowMeans(AUC_CV, na.rm = TRUE)
  best_lambda_idx <- which.max(mean_AUC)
  best_lambda <- lamseq[best_lambda_idx]
  
  # Final model training with the best lambda
  final_model <- glmnet(x = X, y = y, family = "binomial", maxit = maxit, lambda = best_lambda, standardize = TRUE)
  
  # Determine AUC on the full training set with the best lambda
  pred_Lasso_Train <- predict(final_model, newx = X, s = best_lambda, type = "response")
  AUC_Train <- auc(response = y, predictor = as.numeric(pred_Lasso_Train))
  
  # Count the number of active coefficients
  active_set_Train <- length(coef(final_model, s = best_lambda)@x)
  
  # Return results
  res <- list(
    best_lambda = best_lambda,
    best_model = final_model,
    AUC_Train = AUC_Train,
    active_set_Train = active_set_Train
  )
  
  # Set class
  class(res) <- "IntLasso"
  return(res)
}


#' Tune and Train Internal Random Forest
#'
#' This function tunes and trains a Random Forest classifier using internal cross-validation.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param num.trees An integer specifying the number of trees in the Random Forest. Default is 500.
#' @param nfolds An integer specifying the number of folds for cross-validation. Default is 5.
#' @param seed An integer specifying the random seed for reproducibility.
#'
#' @return A list containing the best `min.node.size` value, the final trained model (`best_model`), and the AUC on the training data (`AUC_Train`).
#' @import ranger
#' @import mlr
#' @import pROC
#' @export
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' data <- ... # your training data
#' result <- tuneandtrainIntRF(data, num.trees = 500, nfolds = 5, seed = 123)
#' result$best_min.node.size
#' result$best_model
#' result$AUC_Train
#' }
tuneandtrainIntRF <- function(data, num.trees = 500, nfolds = 5, seed = 123) {
  # Load necessary libraries
  library(ranger)
  library(mlr)
  library(pROC)
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  
  # Set random seed for reproducibility
  set.seed(seed)
  
  # Split data into predictors and response
  X <- data[, -1]
  y <- as.factor(data[, 1])
  
  # Combine data
  Combined_data <- cbind(y, X)
  
  # Cross-validation
  partition <- sample(rep(1:nfolds, length.out = nrow(data)))
  auc_CV <- matrix(NA, nrow = nrow(data) - 1, ncol = nfolds)
  
  for (j in 1:nfolds) {
    XTrain <- data[partition != j, ]
    XTest <- data[partition == j, ]
    
    if (length(unique(XTest[, 1])) == 1) {
      auc_CV[, j] <- NA
    } else {
      Combined_data <- rbind(XTrain, XTest)
      Combined_data[, 1] <- as.factor(Combined_data[, 1])
      
      for (i in 1:(nrow(XTrain) - 1)) {
        # Fit Random Forest Model
        task <- makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
        lrn <- makeLearner("classif.ranger", predict.type = "prob", num.threads = 1, num.trees = num.trees, min.node.size = i, save.memory = TRUE)
        
        train.set <- 1:nrow(XTrain)
        test.set <- (nrow(XTrain) + 1):nrow(Combined_data)
        
        model <- train(lrn, task, subset = train.set)
        pred <- predict(model, task = task, subset = test.set)
        
        auc_CV[i, j] <- performance(pred, measures = mlr::auc)
      }
    }
  }
  
  # Determine the best min.node.size based on the highest average AUC
  mean_AUC <- rowMeans(auc_CV, na.rm = TRUE)
  best_min.node.size <- which.max(mean_AUC)
  
  # Train the final model with the best min.node.size
  final_task <- makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
  final_lrn <- makeLearner("classif.ranger", predict.type = "prob", num.trees = num.trees, min.node.size = best_min.node.size, save.memory = TRUE)
  
  final_model <- train(final_lrn, final_task, subset = 1:nrow(data))
  
  # Predict on the training data using the optimal min.node.size
  pred_Lasso_Train <- predict(final_model, task = final_task, subset = 1:nrow(data))
  
  # Calculate AUC on the training data
  AUC_Train <- performance(pred_Lasso_Train, measures = mlr::auc)
  
  # Return the result
  res <- list(
    best_min.node.size = best_min.node.size,
    best_model = final_model,
    AUC_Train = AUC_Train
  )
  
  # Set class
  class(res) <- "IntRF"
  return(res)
}


#' Tune and Train Internal Ridge Regression
#'
#' This function tunes and trains a Ridge Regression classifier using internal cross-validation.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param maxit An integer specifying the maximum number of iterations. Default is 120000.
#' @param nlambda An integer specifying the number of lambda values to use in the Ridge model.
#' @param nfolds An integer specifying the number of folds for cross-validation. Default is 5.
#' @param seed An integer specifying the random seed for reproducibility. Default is 123.
#'
#' @return A list containing the best lambda value (`best_lambda`), the final trained model (`best_model`), the AUC on the training data (`AUC_Train`).
#' @import glmnet
#' @import pROC
#' @export
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' data <- ... # your training data
#' result <- tuneandtrainIntRidge(data, maxit = 120000, nlambda = 200, nfolds = 5, seed = 123)
#' result$best_lambda
#' result$best_model
#' result$AUC_Train
#' }
tuneandtrainIntRidge <- function(data, maxit = 120000, nlambda = 200, nfolds = 5, seed = 123) {
  # Load necessary libraries
  library(glmnet)
  library(pROC)
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  
  # Set random seed for reproducibility
  set.seed(seed)
  
  # Split data into predictors and response
  X <- as.matrix(data[, -1])
  y <- as.factor(data[, 1])
  
  # Fit initial Ridge Regression model to obtain lambda sequence
  fit_Ridge <- glmnet(x = X, y = y, family = "binomial", maxit = maxit, nlambda = nlambda, alpha = 0, standardize = TRUE)
  lamseq <- fit_Ridge$lambda
  
  # Cross-validation
  partition <- sample(rep(1:nfolds, length.out = nrow(data)))
  AUC_CV <- matrix(NA, nrow = length(lamseq), ncol = nfolds)
  
  for (j in 1:nfolds) {
    XTrain <- X[partition != j, , drop = FALSE]
    yTrain <- y[partition != j]
    XTest <- X[partition == j, , drop = FALSE]
    yTest <- y[partition == j]
    
    if (length(unique(yTest)) == 1) {
      AUC_CV[, j] <- NA
    } else {
      fit_Ridge_CV <- glmnet(x = XTrain, y = yTrain, family = "binomial", maxit = maxit, lambda = lamseq, alpha = 0, standardize = TRUE)
      pred_Ridge_CV <- predict(fit_Ridge_CV, newx = XTest, s = lamseq, type = "response")
      
      for (i in 1:ncol(pred_Ridge_CV)) {
        AUC_CV[i, j] <- auc(response = yTest, predictor = as.numeric(pred_Ridge_CV[, i]))
      }
    }
  }
  
  # Determine the best lambda based on the highest average AUC
  mean_AUC <- rowMeans(AUC_CV, na.rm = TRUE)
  best_lambda_idx <- which.max(mean_AUC)
  best_lambda <- lamseq[best_lambda_idx]
  
  # Final model training with the best lambda
  final_model <- glmnet(x = X, y = y, family = "binomial", maxit = maxit, lambda = best_lambda, alpha = 0, standardize = TRUE)
  
  # Predict on the training data using the optimal lambda value
  pred_Ridge_Train <- predict(final_model, newx = X, s = best_lambda, type = "response")
  
  # Calculate AUC on the training data
  AUC_Train <- auc(response = y, predictor = as.numeric(pred_Ridge_Train))
  
  # Return the result
  res <- list(
    best_lambda = best_lambda,
    best_model = final_model,
    AUC_Train = AUC_Train
  )
  
  # Set class
  class(res) <- "IntRidge"
  return(res)
}



#' Tune and Train Internal SVM
#'
#' This function tunes and trains an SVM classifier using internal cross-validation.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param kernel A character string specifying the kernel type to be used in the SVM. Default is "linear".
#' @param cost_seq A numeric vector of cost values to be evaluated. Default is `2^(-15:15)`.
#' @param scale A logical indicating whether to scale the predictor variables. Default is FALSE.
#' @param nfolds An integer specifying the number of folds for cross-validation. Default is 5.
#' @param seed An integer specifying the random seed for reproducibility. Default is 123.
#'
#' @return A list containing the best cost value (`best_cost`), the final trained model (`best_model`), and the AUC on the training data (`AUC_Train`).
#' @import e1071
#' @import mlr
#' @import pROC
#' @export
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' data <- ... # your training data
#' result <- tuneandtrainIntSVM(data, kernel = "linear", cost_seq = 2^(-15:15), scale = FALSE, nfolds = 5, seed = 123)
#' result$best_cost
#' result$best_model
#' result$AUC_Train
#' }
tuneandtrainIntSVM <- function(data, kernel = "linear", cost_seq = 2^(-15:15), scale = FALSE, nfolds = 5, seed = 123) {
  # Load necessary libraries
  library(e1071)
  library(mlr)
  library(pROC)
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  
  # Set random seed for reproducibility
  set.seed(seed)
  
  # Split data into predictors and response
  X <- as.matrix(data[, -1])
  y <- as.factor(data[, 1])
  
  # Combine data
  Combined_data <- data.frame(y, X)
  
  # Cross-validation
  partition <- sample(rep(1:nfolds, length.out = nrow(data)))
  auc_CV <- matrix(NA, nrow = length(cost_seq), ncol = nfolds)
  
  for (j in 1:nfolds) {
    XTrain <- Combined_data[partition != j, ]
    XTest <- Combined_data[partition == j, ]
    
    if (length(unique(XTest[, 1])) == 1) {
      auc_CV[, j] <- NA
    } else {
      for (i in seq_along(cost_seq)) {
        cost <- cost_seq[i]
        
        # Fit SVM model
        task <- makeClassifTask(data = rbind(XTrain, XTest), target = names(Combined_data)[1], check.data = FALSE)
        lrn <- makeLearner("classif.svm", predict.type = "prob", kernel = kernel, par.vals = list(cost = cost), scale = scale)
        
        train.set <- 1:nrow(XTrain)
        test.set <- (nrow(XTrain) + 1):nrow(rbind(XTrain, XTest))
        
        model <- train(lrn, task, subset = train.set)
        pred <- predict(model, task = task, subset = test.set)
        
        auc_CV[i, j] <- performance(pred, measures = list(mlr::auc))
      }
    }
  }
  
  # Determine the best cost based on the highest average AUC
  mean_AUC <- rowMeans(auc_CV, na.rm = TRUE)
  best_cost <- cost_seq[which.max(mean_AUC)]
  
  # Train the final model with the best cost
  final_task <- makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
  final_lrn <- makeLearner("classif.svm", predict.type = "prob", kernel = kernel, par.vals = list(cost = best_cost), scale = scale)
  
  final_model <- train(final_lrn, final_task, subset = 1:nrow(data))
  
  # Predict on the training data using the optimal cost value
  pred_SVM_Train <- predict(final_model, task = final_task, subset = 1:nrow(data))
  
  # Calculate AUC on the training data
  AUC_Train <- performance(pred_SVM_Train, measures = list(mlr::auc))
  
  # Return the result
  res <- list(
    best_cost = best_cost,
    best_model = final_model,
    AUC_Train = AUC_Train
  )
  
  # Set class
  class(res) <- "IntSVM"
  return(res)
}


#' Tune and Train by tuning method Int
#'
#' This function tunes and trains a specified classifier using the appropriate method and data.
#'
#' @param data Training data as a data frame. The first column should be the response variable.
#' @param classifier The classifier to use. Must be one of 'boosting', 'rf', 'lasso', 'ridge', 'svm'.
#' @param ... Additional arguments to pass to the specific classifier function.
#' @return A list containing the results from the specific classifier tuning and training function.
#' @export
tuneandtrainInt <- function(data, classifier, ...) {
  
  # arguments
  #args <- list(...)
  
  # run function by classifer
  if (classifier == "boosting") {
    res <- tuneandtrainIntBoost(data = data,...)
  } else if (classifier == "rf") {
    res <- tuneandtrainIntRF(data = data, ...)
  } else if (classifier == "lasso") {
    res <- tuneandtrainIntLasso(data = data, ...)
  } else if (classifier == "ridge") {
    res <- tuneandtrainIntRidge(data = data, ...)
  } else if (classifier == "svm") {
    res <- tuneandtrainIntSVM(data = data, ...)
  } else {
    stop("Unsupported classifier type. Choose from 'boosting', 'rf', 'lasso', 'ridge', 'svm'.")
  }
  
  return(res)
}

#' Tune and Train Classifier
#'
#' This function tunes and trains a classifier using a specified tuning method.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor) and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data (only needed for specific tuning methods like "ext").
#' @param tuningmethod A character string specifying which tuning approach to use. Options are "robusttunec", "ext", "int", and so on.
#' @param classifier A character string specifying which classifier to use. Options are "boosting", "rf", and so on.
#' @param ... Additional parameters to be passed to the specific tuning and training functions.
#'
#' @return A list containing the results of the tuning and training process, specific to the chosen tuning method and classifier.
#' @export
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' data <- data.frame(y = factor(c(1, 0, 1, 0)), x1 = rnorm(4), x2 = rnorm(4))
#' dataext <- data.frame(y = factor(c(1, 0)), x1 = rnorm(2), x2 = rnorm(2))
#' result <- tuneandtrain(data, dataext, tuningmethod = "robusttunec", classifier = "boosting")
#' }
tuneandtrain <- function(data, dataext = NULL, tuningmethod, classifier, ...) {
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  
  # Initialize result
  res <- NULL
  
  # Choose the tuning method and call the respective function
  if (tuningmethod == "robusttunec") {
    if (is.null(dataext)) stop("dataext is required for the 'robusttunec' method.")
    dataext <- as.data.frame(dataext)
    res <- tuneandtrainRobustTuneC(data, dataext, classifier, ...)
  } else if (tuningmethod == "ext") {
    if (is.null(dataext)) stop("dataext is required for the 'ext' method.")
    dataext <- as.data.frame(dataext)
    res <- tuneandtrainExt(data, dataext, classifier, ...) 
  } else if (tuningmethod == "int") {
    res <- tuneandtrainInt(data, classifier, ...)
  } else {
    stop("Unknown tuning method specified.")
  }
  
  return(res)
}