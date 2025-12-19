#!/usr/bin/env Rscript
# Benchmark: GMM Estimation
# Input: CSV file path via command line argument  
# Output: JSON with execution time

suppressPackageStartupMessages(library(plm))

args <- commandArgs(trailingOnly = TRUE)
data_file <- args[1]

# Read data
data <- read.csv(data_file)

# Convert to pdata.frame
pdata <- pdata.frame(data, index = c("id", "time"))

# Build formula: y ~ lag(y, 1) + x columns | lag(y, 2:99)
x_cols <- grep("^x", names(data), value = TRUE)
formula_str <- paste("y ~ lag(y, 1) +", paste(x_cols, collapse = " + "), "| lag(y, 2:99)")

# Benchmark - Arellano-Bond GMM
start_time <- Sys.time()
tryCatch({
  model <- pgmm(as.formula(formula_str), data = pdata, 
                effect = "individual", model = "twosteps")
  elapsed <- as.numeric(Sys.time() - start_time, units = "secs")
  cat(sprintf('{"method": "GMM (pgmm)", "time": %.6f, "n": %d}\n', 
              elapsed, nrow(data)))
}, error = function(e) {
  elapsed <- as.numeric(Sys.time() - start_time, units = "secs")
  cat(sprintf('{"method": "GMM (pgmm)", "time": %.6f, "n": %d, "error": "%s"}\n', 
              elapsed, nrow(data), gsub('"', '\\"', e$message)))
})
