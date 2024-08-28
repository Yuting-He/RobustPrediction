#' Tune and Train RobustTuneC Random Forest
#'
#' This function tunes and trains a Random Forest classifier using the "RobustTuneC" method. The function 
#' uses K-fold cross-validation (with K specified by the user) to select the best model based on AUC (Area Under the Curve).
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param K Number of folds to use in cross-validation. Default is 5.
#' @param num.trees An integer specifying the number of trees to grow in the Random Forest. Default is 500.
#'
#' @return A list containing the best minimum node size (`best_min_node_size`), the final trained model (`best_model`), and the AUC of the final model (`final_auc`).
#' @export
#'
#' @import mlr
#' @import ranger
#' @import pROC
#'
#' @examples
#' \dontrun{
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage
#' result <- tuneandtrainRobustTuneCRF(sample_data_train, sample_data_extern, K = 5, num.trees = 500)
#' result$best_min_node_size
#' result$best_model
#' result$final_auc
#' }
tuneandtrainRobustTuneCRF <- function(data, dataext, K = 5, num.trees = 500) {
  
  library(mlr)
  library(ranger)
  library(pROC)
  
  # Split Train in K parts
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
  
  # Calculate AUC on the external validation set
  final_predictions <- predict(final_model, data = dataext, type = "response")$predictions[,2]
  final_auc <- auc(response = as.factor(dataext[,1]), predictor = final_predictions)
  
  # return the result
  res <- list(
    best_min_node_size = min.node.size.c,
    best_model = final_model,
    final_auc = final_auc
  )
  
  # Set class
  class(res) <- "RobustTuneCRF"
  return(res)
}