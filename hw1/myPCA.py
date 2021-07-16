def myPCA(path,dim):
    # read file
    vl=pd.read_csv(path)
    # use 7 specific features to do PCA
    target = vl.iloc[:][['views','likes','dislikes','comment_count','comments_disabled','ratings_disabled','ratings_disabled','video_error_or_removed']]
    # drop NaN
    target.dropna()
    #
    target['comments_disabled'] =  target['comments_disabled'].astype(int)
    target['ratings_disabled'] = target['ratings_disabled'].astype(int)
    target['ratings_disabled'] = target['ratings_disabled'].astype(int)
    target['video_error_or_removed'] = target['video_error_or_removed'].astype(int)
    # standardization
    target = (target - np.nanmean(target)) / np.nanstd(target)
    # build covariance matrix of input matrix
    cov = target.cov()
    # get eigenvector of cov
    e_val, ev = np.linalg.eig(cov)
    # do PCA
    target_pca = np.dot(target,ev[:,0:dim])
    target_pca = (target_pca - np.nanmean(target_pca)) / np.nanstd(target_pca)
    return target_pca