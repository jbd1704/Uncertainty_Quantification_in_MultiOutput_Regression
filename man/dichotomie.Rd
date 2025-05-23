% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/uncertainty_utils.R
\name{dichotomie}
\alias{dichotomie}
\title{Dichotomy Function to Find the Optimal Beta}
\usage{
dichotomie(target, xmin, xmax, n_iter, Eval_simultaneous_coverage_beta)
}
\arguments{
\item{target}{A numeric value representing the desired target for the empirical simultaneous coverage.}

\item{xmin}{The minimum value for Beta (the lower bound of the search interval).}

\item{xmax}{The maximum value for Beta (the upper bound of the search interval).}

\item{n_iter}{An integer indicating the number of iterations to perform for the binary search.}

\item{Eval_simultaneous_coverage_beta}{A function that takes a Beta value as input and returns
the empirical simultaneous coverage for that Beta.}
}
\value{
A numeric value representing the Beta that minimizes the difference between
the empirical simultaneous coverage and the target value.
}
\description{
This function implements the dichotomy method to find the optimal Beta value that
minimizes the difference between the empirical simultaneous coverage and a target value.
It performs binary search over a specified range of Beta values and iterates for a given
number of iterations to find the Beta that best satisfies the target coverage.
}
\examples{
# Example usage:
target_coverage <- 0.90

# Simulated Eval_simultaneous_coverage_beta function (for the example only)
Eval_simultaneous_coverage_beta <- function(Beta) {
  return(0.9)  # Simulated coverage, replace with actual logic
}

# Running dichotomy to find the optimal Beta
Beta_optimal <- dichotomie(target = target_coverage, xmin = 0, xmax = 1, n_iter = 10, Eval_simultaneous_coverage_beta = Eval_simultaneous_coverage_beta)
print(Beta_optimal)

}
