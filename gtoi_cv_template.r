#!/usr/bin Rscript

library('h5')
library('varbvs')

args = commandArgs(trailingOnly=T)
data_path = args[1]
save_path = args[2]
col_idx = as.numeric(args[3])
cname = args[4]
cvsavep = args[5]
g2iresdir = args[6]

setwd("/storage/gablab001/data/genus/GIT/genus/bayes/data_sets/files_for_edward/dev_for_container/")
source('utils.r')

file <- h5file(data_path, 'r')
icol_name = file[cname][,][col_idx]
Iy <- file["I"][ ,col_idx]
Z <- file["Z"][,]
G <- file["G"][,]
cv_df <- read.csv(cvsavep)
nsplits <- as.integer(dim(cv_df)[2]/2)

for (ns in seq(nsplits)){
    train_col <- paste("train", ns, sep="")
    #test_col <- paste("test", ns, sep="")

    # alternative model
    fit <- varbvs(
        G[cv_df[, train_col], ],
        Z[cv_df[, train_col], ],
        matrix(Iy[cv_df[, train_col]]),
        "gaussian",
        logodds=seq(-5,-3,0.25)
    )

    # null model
    X <- matrix(1., length(cv_df[, train_col]), 1)
    fit_null <- varbvs(
        X,
        Z[cv_df[, train_col], ],
        matrix(Iy[cv_df[, train_col], ]),
        "gaussian"
    )
    bf <- bayesfactor(fit_null$logw, fit$logw)
    bfdf <- as.data.frame(c(bf, log10(bf)))
    rownames(bfdf) <- c("bf", "ln10bf")
    colnames(bfdf) <- "comp"

    # writing results to file
    write_name <- paste("g2i_result_col_idx", col_idx, "_cv_", ns, sep="")
    gdir <- file.path(g2iresdir, write_name)

    if (!dir.exists(gdir)) {
        dir.create(gdir)
    } else {
        print("directory exists, did not create it")
    }

    setwd(gdir)
    save_results(fit, bf)

    setwd(save_path)
    write.csv(bfdf, paste("bf_col_idx-", col_idx, "-", "_cv_", ns, "_",icol_name, sep=""))

}
