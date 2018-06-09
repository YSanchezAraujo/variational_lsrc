#!/usr/bin Rscript

library('h5')
library('varbvs')

args = commandArgs(trailingOnly=T)
data_path = args[1]
bf_path = args[2]
save_path = args[3]

df <- read.csv(bf_path)
N <- dim(df)[1]

bf <- rep(0., N)
bflog10 <- rep(0., N)

for (idx in 1:N) {
    fp <- as.character(df[idx, 1])
    fpdf <- read.csv(fp)
    bf[idx] = fpdf[1,2]
    bflog10[idx] = fpdf[2,2]
}

file <- h5file(data_path, 'r')
X <- file["I"][,]
Z <- file["Z"][,]
y <- file["y"][,]

fit <- varbvs(X, Z, y, family="binomial", logodds=matrix(bf, 1))

dir.create("fxvb_out")
setwd(file.path(getwd(), "fxvb_out"))

# things to save together
# logw, w, sa, logoods, pip, beta
## eta, s, mu, alpha
