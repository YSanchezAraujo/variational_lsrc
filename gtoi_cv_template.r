#!/usr/bin Rscript

library('h5')
library('varbvs')
source('utils.r')

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
nsplits = 5

cv_df <- make_combined_splits(seq(dim(Iy)[1]), .5, nsplits)

write.csv(
    cv_df,
    file.path(cvsavep, "cv_splits_df.csv"),
    row.names=FALSE
)

for (ns in seq(nsplits)){

    train_col <- paste("train", ns, sep="")
    test_col <- paste("test", ns, sep="")

    # alternative model
    fit <- varbvs(
        G[cv_df[, train_col], ],
        Z[cv_df[, train_col], ],
        matrix(Iy[cv_df[, train_col]])
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
