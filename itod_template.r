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
    fpdf <- read.csv(as.character(df[idx, 1]))
    bf[idx] = fpdf[1,2]
    bflog10[idx] = fpdf[2,2]
}

file <- h5file(data_path, 'r')
X <- file["I"][,]
Z <- file["Z"][,]
y <- file["y"][,]

fit <- varbvs(X, Z, y, family="binomial", logodds=matrix(bflog10))

dir.create("fxvb_out")
setwd(file.path(getwd(), "fxvb_out"))

n170 <- as.data.frame(cbind(fit$logw, fit$w, fit$sa, fit$logodds, fit$pip, fit$beta))
colnames(n170) <- c("logw", "w", "sa", "logodds", "pip", "beta")
write.csv(n170, "logw_w_sa_logodds_pip_beta.csv", row.names=F)
write.csv(as.data.frame(fit$eta), "eta.csv", row.names=F)
write.csv(as.data.frame(fit$s), "s.csv", row.names=F)
write.csv(as.data.frame(fit$mu), "mu.csv", row.names=F)
write.csv(as.data.frame(fit$alpha), "alpha.csv", row.names=F)
