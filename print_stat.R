printStat <- function(predictedLabels, gtLabels) {
    # Prints accuracy, recall, precision, specificity, F-measure, FDR and ROC for each class
    #
    # Args:
    #   predictedLabels: specifies predicted labels
    #   gtLabels: specifies ground truth labels

    library('AUC')

    accuracy <- sum(predictedLabels == gtLabels) / length(gtLabels)
    print(sprintf('  accuracy:       %.5f', accuracy), quote=FALSE)

    for (label in sort(unique(gtLabels))) {
        classPredictedLabels <- predictedLabels == label
        classGtLabels <- gtLabels == label

        TP <- sum(classPredictedLabels == 1 & classGtLabels == 1)
        FP <- sum(classPredictedLabels == 1 & classGtLabels == 0)
        TN <- sum(classPredictedLabels == 0 & classGtLabels == 0)
        FN <- sum(classPredictedLabels == 0 & classGtLabels == 1)

        precision <- TP / (TP + FP)
        print(sprintf('  %d  precision:   %.5f', label, precision), quote=FALSE)

        recall <- TP / (TP + FN)
        print(sprintf('  %d  recall:      %.5f', label, recall), quote=FALSE)

        specificity <- TN / (FP + TN)
        print(sprintf('  %d  specificity: %.5f', label, specificity), quote=FALSE)

        f1score <- 2 * TP / (2 * TP + FP + FN)
        print(sprintf('  %d  F1-score:    %.5f', label, f1score), quote=FALSE)

        auc <- auc(roc(classPredictedLabels, factor(classGtLabels)))
        print(sprintf('  %d  AUC:         %.5f', label, auc), quote=FALSE)
    }
}
