#' Tune and Train External Boosting
#'
#' This function tunes and trains a Boosting classifier using the \code{mboost::glmboost} function.
#' It provides two strategies for tuning the number of boosting iterations (\code{mstop}) based on 
#' the \code{estperf} argument:
#' \itemize{
#'   \item When \code{estperf = FALSE} (default): Hyperparameters are tuned using the external validation dataset. 
#'         The \code{mstop} value that gives the highest AUC on the external dataset is selected as the best model.
#'         However, no AUC value is returned in this case, as per best practices.
#'   \item When \code{estperf = TRUE}: Hyperparameters are tuned internally using the training dataset. 
#'         The model is then validated on the external dataset to provide a conservative (slightly pessimistic) AUC estimate.
#' }
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable 
#'   (factor), and the remaining columns should be the predictor variables.
#' @param estperf A logical value indicating whether to use internal tuning with external validation (\code{TRUE}) 
#'   or external tuning (\code{FALSE}). Default is \code{FALSE}.
#' @param mstop_seq A numeric vector specifying the sequence of boosting iterations to evaluate. 
#'   Default is \code{seq(5, 1000, by = 5)}.
#' @param nu A numeric value specifying the learning rate for boosting. Default is \code{0.1}.
#'
#' @return A list containing the following components:
#'   \itemize{
#'     \item \code{best_mstop}: The optimal number of boosting iterations determined during the tuning process.
#'     \item \code{best_model}: The trained Boosting model using the selected \code{mstop}.
#'     \item \code{est_auc}: The AUC value evaluated on the external dataset. This is only returned when \code{estperf = TRUE}, 
#'       providing a conservative (slightly pessimistic) estimate of the model's performance.
#'   }
#' 
#' @export
#'
#' @examples
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage with external tuning (default)
#' mstop_seq <- seq(50, 500, by = 50)
#' result <- tuneandtrainExtBoost(sample_data_train, sample_data_extern, 
#'   mstop_seq = mstop_seq, nu = 0.1)
#' print(result$best_mstop)         # Optimal mstop
#' print(result$best_model)         # Trained Boosting model
#' # Note: est_auc is not returned when estperf = FALSE
#'
#' # Example usage with internal tuning and external validation
#' result_internal <- tuneandtrainExtBoost(sample_data_train, sample_data_extern, 
#'   estperf = TRUE, mstop_seq = mstop_seq, nu = 0.1)
#' print(result_internal$best_mstop) # Optimal mstop
#' print(result_internal$best_model) # Trained Boosting model
#' print(result_internal$est_auc)    # AUC on external validation dataset

tuneandtrainExtBoost <- function(data, dataext, estperf = FALSE, mstop_seq = seq(5, 1000, by = 5), nu = 0.1) {
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  if (!estperf) {
    # External tuning: use external validation to select best mstop
    fit_Boost <- mboost::glmboost(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]),
                                  family = mboost::Binomial(), 
                                  control = mboost::boost_control(mstop = max(mstop_seq), nu = nu),
                                  center = FALSE)
    AUC <- numeric(length(mstop_seq))
    
    for (i in seq_along(mstop_seq)) {
      mseq <- mstop_seq[i]
      pred_Boost <- stats::predict(fit_Boost[mseq], newdata = as.matrix(Extern[, -1]), type = "response")
      AUC[i] <- pROC::auc(response = as.factor(Extern[, 1]), predictor = as.numeric(pred_Boost), quiet = TRUE)
    }
    
    chosen_model <- which.max(AUC)
    chosen_mstop <- mstop_seq[chosen_model]
    
    # Train final model with chosen mstop
    final_model <- mboost::glmboost(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]),
                                    family = mboost::Binomial(), 
                                    control = mboost::boost_control(mstop = chosen_mstop, nu = nu),
                                    center = FALSE)
    
    # Return result without AUC
    res <- list(
      best_mstop = chosen_mstop,
      best_model = final_model,
      est_auc = NULL  # No AUC returned for estperf = FALSE
    )
  } else {
    # Internal tuning: use training data to select best mstop
    fit_Boost <- mboost::glmboost(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]),
                                  family = mboost::Binomial(), 
                                  control = mboost::boost_control(mstop = max(mstop_seq), nu = nu),
                                  center = FALSE)
    AUC <- numeric(length(mstop_seq))
    
    for (i in seq_along(mstop_seq)) {
      mseq <- mstop_seq[i]
      pred_Boost <- stats::predict(fit_Boost[mseq], newdata = as.matrix(Train[, -1]), type = "response")
      AUC[i] <- pROC::auc(response = as.factor(Train[, 1]), predictor = as.numeric(pred_Boost), quiet = TRUE)
    }
    
    chosen_model <- which.max(AUC)
    chosen_mstop <- mstop_seq[chosen_model]
    
    # Validate internally tuned model on external data
    final_model <- mboost::glmboost(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]),
                                    family = mboost::Binomial(), 
                                    control = mboost::boost_control(mstop = chosen_mstop, nu = nu),
                                    center = FALSE)
    pred_Extern <- stats::predict(final_model, newdata = as.matrix(Extern[, -1]), type = "response")
    est_auc <- pROC::auc(response = as.factor(Extern[, 1]), predictor = as.numeric(pred_Extern), quiet = TRUE)
    
    # Return result with AUC
    res <- list(
      best_mstop = chosen_mstop,
      best_model = final_model,
      est_auc = est_auc  # Conservative AUC estimate
    )
  }
  
  # Set class
  class(res) <- "ExtBoost"
  return(res)
}