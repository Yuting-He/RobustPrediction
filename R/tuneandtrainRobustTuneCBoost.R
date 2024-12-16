#' Tune and Train RobustTuneC Boosting
#'
#' This function tunes and trains a Boosting classifier using the \code{mboost::glmboost} function 
#' and the "RobustTuneC" method. The function performs K-fold cross-validation on the training dataset 
#' and evaluates a sequence of boosting iterations (\code{mstop}) based on the Area Under the Curve (AUC).
#'
#' After cross-validation, the best \code{mstop} value is selected based on the AUC, and the final Boosting 
#' model is trained using this optimal \code{mstop}. The external validation dataset is then used to calculate 
#' the final AUC and assess the model performance.
#' 
#' @param data Training data as a data frame. The first column should be the response variable.
#' @param dataext External validation data as a data frame. The first column should be the response variable.
#' @param K Number of folds to use in cross-validation. Default is 5.
#' @param mstop_seq A sequence of boosting iterations to consider. Default is a sequence starting at 5 and 
#'   increasing by 5 each time, up to 1000.
#' @param nu Learning rate for the boosting algorithm. Default is 0.1.
#' @return A list containing the best number of boosting iterations (`best_mstop`), 
#'   the final trained model (`best_model`), and the chosen c value(`best_c`).
#'   
#' @export
#' 
#' @examples
#' # Load the sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage with the sample data
#' mstop_seq <- seq(50, 500, by = 50)
#' result <- tuneandtrainRobustTuneCBoost(sample_data_train, sample_data_extern, mstop_seq = mstop_seq)
#' result$best_mstop
#' result$best_model
#' result$best_c
tuneandtrainRobustTuneCBoost <- function(data, dataext, K = 5, mstop_seq = seq(5, 1000, by = 5), nu = 0.1) {
  
  # Ensure data and dataext are matrices
  x_train <- as.matrix(data[, -1])  # Exclude the response variable (first column)
  y_train <- as.factor(data[, 1])   # Response variable
  x_test <- as.matrix(dataext[, -1])  # Exclude the response variable in external data
  y_test <- as.factor(dataext[, 1])   # Response variable in external data
  
  # K-fold cross validation
  n <- nrow(x_train)
  partition <- rep(1:K, length = n)
  partition <- partition[sample(n)]
  
  # Initialize CV AUC matrix
  AUC_CV <- matrix(NA, nrow = length(mstop_seq), ncol = K)
  
  # Cross-validation
  for (j in 1:K) {
    XTrain <- x_train[partition != j, , drop = FALSE]
    XTest <- x_train[partition == j, , drop = FALSE]
    yTrain <- y_train[partition != j]
    yTest <- y_train[partition == j]
    
    if (length(unique(yTest)) == 1) {
      AUC_CV[, j] <- NA
    } else {
      # Train the model
      fit_Boost_CV <- mboost::glmboost(x = XTrain, y = yTrain,
                                       family = mboost::Binomial(), 
                                       control = mboost::boost_control(mstop = max(mstop_seq), nu = nu),
                                       center = FALSE)
      
      for (i in seq_along(mstop_seq)) {
        # External validation
        pred_Boost_CV <- stats::predict(fit_Boost_CV[mstop_seq[i]], newdata = XTest, type = "response")
        
        # 1-AUC
        AUC_CV[i, j] <- 1 - pROC::auc(response = yTest, predictor = pred_Boost_CV[, 1], quiet = TRUE)
      }
    }
  }
  
  # Calculate mean AUC and choose best mstop
  AUC_mean <- rowMeans(AUC_CV, na.rm = TRUE)
  cvmin <- min(AUC_mean, na.rm = TRUE)
  cseq <- c(1, 1.1, 1.3, 1.5, 2)
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
    
    pred_Boost_Test.c <- stats::predict(fit_Boost_CV[mstop.c], newdata = x_test, type = "response")
    AUC_Test.c[i] <- pROC::auc(response = y_test, predictor = pred_Boost_Test.c[, 1], quiet = TRUE)[1]
    
    i <- i + 1
  }
  
  # Final choice of mstop
  nctried <- i - 1
  c_val <- cseq[max(which(AUC_Test.c[1:nctried] == max(AUC_Test.c[1:nctried])))]
  
  if (c_val * cvmin < 0.4) {
    mstop.c <- min(mstop_seq[which(AUC_mean <= cvmin * c_val)], na.rm = TRUE)
  } else if (cvmin < 0.4) {
    mstop.c <- min(mstop_seq[which(AUC_mean <= 0.4)], na.rm = TRUE)
  } else {
    mstop.c <- min(mstop_seq[which(AUC_mean <= cvmin)], na.rm = TRUE)
  }
  
  # Train the final model with the chosen mstop
  final_model <- mboost::glmboost(x = x_train, y = y_train,
                                  family = mboost::Binomial(), 
                                  control = mboost::boost_control(mstop = mstop.c, nu = nu),
                                  center = FALSE)
  
  
  # Return the result
  res <- list(
    best_mstop = mstop.c,
    best_model = final_model,
    best_c = c_val
  )
  
  # Set class for the result
  class(res) <- "RobustTuneCBoost"
  return(res)
}