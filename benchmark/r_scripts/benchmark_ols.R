#!/usr/bin/env Rscript
# Benchmark: Linear Regression (OLS)
# Input: CSV file path via command line argument
# Output: JSON with execution time

args <- commandArgs(trailingOnly = TRUE)
data_file <- args[1]

# Read data
data <- read.csv(data_file)
y <- data$y
X <- as.matrix(data[, !names(data) %in% "y"])

# Benchmark
start_time <- Sys.time()
model <- lm(y ~ X)
end_time <- Sys.time()

elapsed <- as.numeric(end_time - start_time, units = "secs")

# Output JSON
cat(sprintf('{"method": "OLS (lm)", "time": %.6f, "n": %d, "p": %d}\n', 
            elapsed, nrow(X), ncol(X)))
