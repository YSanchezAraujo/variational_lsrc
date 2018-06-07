#!/usr/bin Rscript

library('h5')
library('varbvs')

args = commandArgs(trailingOnly=T)
data_path = args[1]
save_path = args[2]
col_idx = as.numeric(args[3])
cname = args[4]

file <- h5file(data_path, 'r')
icol_name = file[cname][,][col_idx]
Iy <- file["I"][ ,col_idx]
Z <- file["Z"][,]
G <- file["G"][,]
logodds <- seq(-5,-3,0.25)
fit <- varbvs(G, Z, Iy, "gaussian", logodds=logodds)

# null model
X <- matrix(1., dim(G)[1], 1)
fit_null <- varbvs(X, Z, Iy, "gaussian")
bf <- bayesfactor(fit_null$logw, fit$logw)
bfdf <- as.data.frame(c(bf, log10(bf)))
rownames(bfdf) <- c("bf", "ln10bf")
colnames(bfdf) <- "comp"

save_results <- function(mf, bfdf) {
    write.csv(bfdf, "bayesfactor.txt")
    write.csv(as.data.frame(mf$mu.cov), "mu.cov.csv", row.names=F)
    logwdf <- as.data.frame(mf$logw)
    colnames(logwdf) <- "logw"
    write.csv(logwdf, "logw.csv", row.names=F)
    idf <- as.data.frame(cbind(mf$logw, mf$w, mf$sigma, mf$sa, mf$logodds))
    colnames(idf) <- c("logw", "w", "sigma", "sa", "logodds")
    write.csv(idf, "logw_w_sigma_sa_logodds.csv", row.names=F)
    mu_s <- as.data.frame(cbind(mf$pip, mf$beta))
    colnames(mu_s) <- c("pip", "beta")
    write.csv(mu_s, "pip_beta.csv", row.names=F)
    write.csv(as.data.frame(mf$alpha), "alpha.csv", row.names=F)
    write.csv(as.data.frame(mf$mu), "mu.csv", row.names=F)
    write.csv(as.data.frame(mf$s), "s.csv", row.names=F)
    write.csv(as.data.frame(mf$pve), "pve.csv", row.names=F)
}

# writing results to file
write_name = paste("g2i_result_col_idx", col_idx, sep="")

if (!dir.exists(file.path(getwd(), write_name))) {
    dir.create(write_name)
    setwd(write_name)
    save_results(fit, bf)
} else {
    print("directory exists, nothing done")
}

setwd(save_path)
write.csv(bfdf, paste("bf_col_idx-", col_idx, "-", icol_name, sep=""))
