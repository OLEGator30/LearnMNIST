source('load_data.R')
source('nn_model.R')
source('print_stat.R')

# load training data
data <- loadMNISTData('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
trainData <- data$data # 60000x784
trainLabels <- data$labels # 60000x1

classifier <- learnModel(trainData, trainLabels) # train 1-layer nn model (softmax regression)

print("On train:", quote=FALSE)
printStat(testModel(classifier, trainData), trainLabels) # print statistics on train data

# load testing data from files
data <- loadMNISTData('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
testData <- data$data  # 10000x784
testLabels <- data$labels # 10000x1

print("On test:", quote=FALSE)
printStat(testModel(classifier, testData), testLabels) # print statistics on test data
