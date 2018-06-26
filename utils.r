# use for both combined and not combined
cv_split <- function(indices, split_percent) {
    num_rows <- floor(length(indices)*split_percent)
    indices <- sample(indices, size=length(indices), replace=FALSE)
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

make_combined_splits <- function(indices, split_percent, nsplits){
    num_rows <- floor(length(indices)*split_percent)
    sdf <- data.frame(matrix(ncol=1, nrow=num_rows))
    colnames(sdf) <- "todrop"
    sdf_cols <- c()
    for (ns in seq(nsplits)) {
        sdf <- cbind(sdf, as.data.frame(cv_split(indices, split_percent)))
        sdf_cols <- c(sdf_cols, paste("train", ns, sep=""), paste("test", ns, sep=""))
    }
    sdf <- sdf[, !(names(sdf) %in% c("todrop"))]
    colnames(sdf) <- sdf_cols
    return(sdf)
}

# non combined use
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
