testData = readmatrix('mnist_test.csv');
trainData = readmatrix('mnist_train.csv');

combinedData = [trainData; testData];
save('test_and_train.mat', 'combinedData');
