softmaxByRow <- function(mat) {
    mat <- exp(mat)
    return(mat / apply(mat, 1, sum))
}

meanCrossEntropy <- function(batchPredict, batchLabels) {
    cost <- 0.0
    for (i in 1:length(batchLabels)) {
        cost <- cost - log(batchPredict[i, batchLabels[i] + 1])
    }
    return(cost / length(batchLabels))
}

derivCrossEntropy <- function(batchPredict, batchLabels) {
    for (i in 1:length(batchLabels)) {
        batchPredict[i, batchLabels[i] + 1] <- batchPredict[i, batchLabels[i] + 1] - 1.0
    }
    return(batchPredict / length(batchLabels))
}

learnModel <- function(trainData, trainLabels,
    nIterations=500, learningRate=0.00001, batchSize=500, weightDecay=0.001, momentum=0.5)
{
    # Trains softmax regression
    #
    # Args:
    #   trainData: S-by-F matrix with training samples
    #   trainLabels: S-by-1 matrix with corresponding labels (contains C different labels)
    #   nIterations: number of iterations for SGD optimization process
    #   learningRate: learning rate for SGD optimization process
    #   batchSize: batch size for SGD optimization process
    #   weightDecay: weight decay for SGD optimization process
    #   momentum: momentum for SGD optimization process
    #
    # Returns:
    #   (F+1)-by-C matrix with weights for fitted model
    #
    # See also:
    #   testModel

    trainData <- cbind(trainData, matrix(1, nrow(trainData), 1)) # add biases

    nSamples <- nrow(trainData)
    nFeatures <- ncol(trainData)
    nClasses <- length(unique(trainLabels))

    model <- matrix(runif(nFeatures * nClasses, -1e-4, 1e-4), nrow=nFeatures, ncol=nClasses) # initialize weights
    grad <- matrix(0, nrow=nFeatures, ncol=nClasses) # initialize gradients

    for (iter in 1:nIterations) {
        if (iter %% 100 == 0) {
            learningRate <- learningRate / 10.0 # decrease learning rate during optimization
        }

        # put some random train samples to batch
        idx <- round(runif(batchSize, 1, nSamples))
        batchData <- trainData[idx,]
        batchLabels <- matrix(trainLabels[idx,], nrow=batchSize, ncol=1)

        if (iter %% 10 == 0) {
            cost <- meanCrossEntropy(softmaxByRow(batchData %*% model), batchLabels) # cost on batch
            cost <- cost + weightDecay / 2.0 * sum(model ^ 2) # L2 regularization

            print(sprintf('Iter: %d; cost: %.5f', iter, cost), quote=FALSE)
        }

        deriv <- t(batchData) %*% derivCrossEntropy(softmaxByRow(batchData %*% model), batchLabels) # gradient on batch
        deriv <- deriv + weightDecay * model # L2 regularization

        grad <- momentum * grad - learningRate * deriv # compute gradient using momentum

        model <- model + grad # update weights
    }

    return(model) # return weights
}

testModel <- function(model, testData) {
    # Applies softmax regression model
    #
    # Args:
    #   model: (F+1)-by-C matrix with weights for fitted model
    #   testData: S-by-F matrix with test samples
    #
    # Returns:
    #   S-by-1 matrix with predicted labels (contains less or equal than C different labels)
    #
    # See also:
    #   learnModel

    testData <- cbind(testData, matrix(1, nrow(testData), 1)) # add biases
    return(apply(softmaxByRow(testData %*% model), 1, which.max) - 1) # apply model
}
