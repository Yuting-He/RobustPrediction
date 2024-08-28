#' Package Title: Robust Prediction and Tuning Package
#'
#' This package provides robust parameter tuning and predictive modeling techniques.
#'
#' The RobustPrediction package allows users to build and tune classifiers using various methods, 
#' including RobustTuneC, internal, and external tuning approaches. The package supports a range of 
#' classifiers, such as boosting, lasso, ridge, random forest, and SVM. It is designed for users 
#' who need reliable and accurate models, particularly in scenarios where robust parameter tuning is essential.
#'
#' @docType package
#' @name RobustPrediction
#' @aliases RobustPrediction-package
#' @details
#' This package provides comprehensive tools for robust parameter tuning and predictive modeling.
#' It includes functions for tuning (via RobustTuneC, internal, and external methods), training, 
#' and predicting outcomes with various classifiers.
#'
#' @section Dependencies:
#' This package requires the following packages: \code{caret}, \code{randomForest}, \code{glmnet}, \code{e1071}.
#'
#' @examples
#' # Example usage:
#' data(sample_data_train)
#' data(sample_data_extern)
#' res <- tuneandtrain(sample_data_train, sample_data_extern, tuningmethod = "robusttunec", classifier = "boosting")
#'
#' @references
#' Hubert, M., Rousseeuw, P.J., & Verdonck, T. (2020). Robust PCA for skewed data 
#' and its outlier map. \emph{Journal of Classification}, \emph{37}(2), 189-218. 
#' doi:10.1007/s00357-020-09368-z.
"_PACKAGE"

#' Sample Training Data
#'
#' A dataset containing the response variable `y` and 50 selected predictor variables from `data1`.
#'
#' @format A data frame with 46 rows and 51 columns (1 response variable and 50 predictors):
#' \describe{
#'   \item{y}{Response variable (factor)}
#'   \item{x1}{First predictor variable (numeric)}
#'   \item{x2}{Second predictor variable (numeric)}
#'   ...
#'   \item{x50}{Fiftieth predictor variable (numeric)}
#' }
#' @source Generated from `data1` in the package development process.
"sample_data_train"

#' Sample External Data
#'
#' A dataset containing the response variable `y` and 50 selected predictor variables from `data10`.
#'
#' @format A data frame with 49 rows and 51 columns (1 response variable and 50 predictors):
#' \describe{
#'   \item{y}{Response variable (factor)}
#'   \item{x1}{First predictor variable (numeric)}
#'   \item{x2}{Second predictor variable (numeric)}
#'   ...
#'   \item{x50}{Fiftieth predictor variable (numeric)}
#' }
#' @source Generated from `data10` in the package development process.
"sample_data_extern"
