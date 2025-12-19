#!/usr/bin/env Rscript
# Benchmark: Propensity Score Matching
# Input: CSV file path via command line argument
# Output: JSON with execution time

suppressPackageStartupMessages(library(MatchIt))

args <- commandArgs(trailingOnly = TRUE)
data_file <- args[1]

# Read data
data <- read.csv(data_file)

# Build formula: treatment ~ all x columns
x_cols <- grep("^x", names(data), value = TRUE)
formula_str <- paste("treatment ~", paste(x_cols, collapse = " + "))

# Benchmark - nearest neighbor matching
start_time <- Sys.time()
match_out <- matchit(as.formula(formula_str), data = data, method = "nearest")
end_time <- Sys.time()

elapsed <- as.numeric(end_time - start_time, units = "secs")

# Output JSON
cat(sprintf('{"method": "PSM (MatchIt)", "time": %.6f, "n": %d, "matched": %d}\n', 
            elapsed, nrow(data), sum(match_out$weights > 0)))
