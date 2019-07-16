#!/usr/bin Rscript

library('h5')
library('varbvs')

args = commandArgs(trailingOnly=T)
data_path = args[1]
bf_path = args[2]
save_path = args[3]
cv_path = args[4]
cv_idx = args[5]

df <- read.csv(bf_path)
#test_idx <- read.csv(cv_path)[, paste("test", cv_idx, sep="")]
train_idx <- read.csv(cv_path)[, paste("train", cv_idx, sep="")]
N <- dim(df)[1]
bf <- rep(0., N)
bflog10 <- rep(0., N)

for (idx in 1:N) {
    fpdf <- read.csv(as.character(df[idx, 1]))
    bf[idx] = fpdf[1,2]
    bflog10[idx] = fpdf[2,2]
}

file <- h5file(data_path, 'r')
X <- file["I"][train_idx, ]
Z <- file["Z"][train_idx, ]
y <- file["y"][train_idx]

fit <- varbvs(X, Z, y, family="binomial", logodds=matrix(bflog10))

dir.create(paste("fxvb_out_cv_", cv_idx, sep=""))
setwd(file.path(getwd(), paste("fxvb_out_cv_", cv_idx, sep="")))

n170 <- as.data.frame(cbind(fit$logw, fit$w, fit$sa, fit$logodds, fit$pip, fit$beta))
colnames(n170) <- c("logw", "w", "sa", "logodds", "pip", "beta")
write.csv(n170, "logw_w_sa_logodds_pip_beta.csv", row.names=F)
write.csv(as.data.frame(fit$eta), "eta.csv", row.names=F)
write.csv(as.data.frame(fit$s), "s.csv", row.names=F)
write.csv(as.data.frame(fit$mu), "mu.csv", row.names=F)
write.csv(as.data.frame(fit$alpha), "alpha.csv", row.names=F)
