#!/usr/bin/env Rscript
# Benchmark: Panel Fixed Effects
# Input: CSV file path via command line argument
# Output: JSON with execution time

suppressPackageStartupMessages(library(plm))

args <- commandArgs(trailingOnly = TRUE)
data_file <- args[1]

# Read data
data <- read.csv(data_file)

# Convert to pdata.frame
pdata <- pdata.frame(data, index = c("id", "time"))

# Build formula dynamically (all x columns)
x_cols <- grep("^x", names(data), value = TRUE)
formula_str <- paste("y ~", paste(x_cols, collapse = " + "))

# Benchmark
start_time <- Sys.time()
model <- plm(as.formula(formula_str), data = pdata, model = "within")
end_time <- Sys.time()

elapsed <- as.numeric(end_time - start_time, units = "secs")

# Output JSON
cat(sprintf('{"method": "Panel FE (plm)", "time": %.6f, "n": %d}\n', 
            elapsed, nrow(data)))
