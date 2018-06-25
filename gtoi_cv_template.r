#!/usr/bin Rscript

library('h5')
library('varbvs')

cv_split <- function(indices, split_percent) {
    num_rows <- floor(length(indices)*split_percent)
    use_idx <- sample(indices, size=num_rows, replace=FALSE)
    if (sum(table(use_idx)) != num_rows) {
        print("ERROR IN USE_IDX:CV_SPLIT")
        return(FALSE)
    }
    diff_use_idx <- setdiff(indices, use_idx)
    if (length(intersect(use_idx, diff_use_idx)) != 0) {
        print("ERROR IN SETDIFF:CV_SPIT")
        return(FALSE)
    }
    return(list(train=use_idx, test=diff_use_idx))
}

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

args = commandArgs(trailingOnly=T)
data_path = args[1]
save_path = args[2]
col_idx = as.numeric(args[3])
cname = args[4]
cvsavep = args[5]

file <- h5file(data_path, 'r')
icol_name = file[cname][,][col_idx]
Iy <- file["I"][ ,col_idx]
Z <- file["Z"][,]
G <- file["G"][,]
nsplits=5

write_splits <- function(nsplits, indices, write_path){
    return_paths <- c()
    for (ns in 1:nsplits) {
        df_i <- as.data.frame(cv_split(indices, .5))
        wrp <- file.path(write_path, paste("cv_indicies_", ns, ".csv",sep=""))
        print(wrp)
        write.csv(df_i, wrp, row.names=FALSE)
        return_paths[ns] = wrp
    }
    return(return_paths)
}

cv_read_paths <- write_splits(nsplits, seq(dim(Iy)[1]), cvsavep)

for (ns in seq(nsplits)){
    # need to read in cvpaths
    split <- read.csv(cv_read_paths[ns])

    # alternative model
    fit <- varbvs(
        G[split$train, ],
        Z[split$train, ],
        matrix(Iy[split$train, ]),
        "gaussian",
        logodds=seq(-5,-3,0.25)
    )

    # null model
    X <- matrix(1., length(split$train), 1)
    fit_null <- varbvs(X, Z[split$train, ], matrix(Iy[split$train, ]), "gaussian")
    bf <- bayesfactor(fit_null$logw, fit$logw)
    bfdf <- as.data.frame(c(bf, log10(bf)))
    rownames(bfdf) <- c("bf", "ln10bf")
    colnames(bfdf) <- "comp"

    # writing results to file
    write_name = paste("g2i_result_col_idx", col_idx, "_cv_", ns, sep="")

    if (!dir.exists(file.path(getwd(), write_name))) {
        dir.create(write_name)
    } else {
        print("directory exists, did not create it")
    }

    setwd(write_name)
    save_results(fit, bf)

    setwd(save_path)
    write.csv(bfdf, paste("bf_col_idx-", col_idx, "-", "_cv_", ns, "_",icol_name, sep=""))

}
