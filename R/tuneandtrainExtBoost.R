#' Tune and Train External Boosting
#'
#' This function tunes and trains a Boosting classifier using an external validation dataset. The function 
#' evaluates different numbers of boosting iterations and selects the best model based on AUC (Area Under the Curve).
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param mstop_seq A numeric vector of boosting iterations to be evaluated. Default is a sequence starting at 5 and increasing by 5 each time, up to 1000.
#' @param nu A numeric value for the learning rate. Default is 0.1.
#'
#' @return A list containing the best number of boosting iterations (`best_mstop`), the final trained model (`best_model`), and the AUC of the final model (`final_auc`).
#' @import mboost
#' @import pROC
#' @export
#'
#' @examples
#' \dontrun{
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage
#' mstop_seq <- seq(50, 500, by = 50)
#' result <- tuneandtrainExtBoost(sample_data_train, sample_data_extern, mstop_seq = mstop_seq, nu = 0.1)
#' result$best_mstop
#' result$best_model
#' result$final_auc
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
    # Convert predictions to numeric vector (if not already)
    pred_Boost_numeric <- as.numeric(pred_Boost)
    # Calculate AUC
    AUC[i] <- pROC::auc(response = as.factor(Extern[, 1]), predictor = pred_Boost_numeric)
  }
  
  # Choose the best mstop
  chosen_model <- which.max(AUC)
  chosen_mstop <- mstop_seq[chosen_model]
  final_auc <- AUC[chosen_model]
  
  # Train the final model with the chosen mstop
  final_model <- glmboost(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]),
                          family = Binomial(), control = boost_control(mstop = chosen_mstop, nu = nu),
                          center = FALSE)
  
  # Return the result
  res <- list(
    best_mstop = chosen_mstop,
    best_model = final_model,
    final_auc = final_auc
  )
  
  # Set class
  class(res) <- "ExtBoost"
  return(res)
}