clear all;
test_data = readmatrix('mnist_test.csv');
train_data = readmatrix('mnist_train.csv');
X_train = train_data(:, 2:end);
Y_train = train_data(:, 1);
X_test = test_data(:,2:end);
Y_test = test_data(:, 1);

gamma=0.1;
timp=tic;
template = templateSVM(...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);

linearSVM = fitcecoc(...
    X_train, ...
    Y_train, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', [0; 1; 2; 3; 4; 5; 6; 7; 8; 9]);

w = linearSVM.BinaryLearners{1}.Beta;
margin = 2 / norm(w);

if margin < gamma
    disp(['Marginea e prea mica:', num2str(margin)]);
else
    disp(['Marginea este acceptata: ', num2str(margin)]);
end

yPredLinear = predict(linearSVM, X_test );
eroare_lineara =loss( linearSVM, X_test , Y_test);
toc;
disp(timp);
disp('Eroare:')
disp(eroare_lineara);

accuracy = sum(yPredLinear == Y_test) / numel(Y_test);
disp(['Acuratetea: ', num2str(accuracy)]);

C = confusionmat(Y_test, yPredLinear);

numar_clase = numel(unique(Y_test));
precizia = zeros(numar_clase, 1);
sensibilitatea = zeros(numar_clase, 1);
f1Score = zeros(numar_clase, 1);

for i = 1:numar_clase
    truePositives = C(i, i);
    falsePositives = sum(C(:, i)) - truePositives;
    falseNegatives = sum(C(i, :)) - truePositives;
    
    precizia(i) = truePositives / (truePositives + falsePositives);
    sensibilitatea(i) = truePositives / (truePositives + falseNegatives);
    f1Score(i) = 2 * (precizia(i) * sensibilitatea(i)) / (precizia(i) + sensibilitatea(i));
end

disp('Precizie:');
disp(precizia);
disp('Sensibilitatea:');
disp(sensibilitatea);
disp('F1-Score:');
disp(f1Score);

figure;
confusionchart(C);
title('Confusion Matrix');
saveas(gcf, 'ConfusionMatrix.png');

figure;
bar([precizia sensibilitatea f1Score]);
title('Precizie,Sensibilitatea, si F1-Score pentru fiecare cifra');
xlabel('Cifra');
ylabel('Scor');
legend('Precizie', 'Sensibilitate', 'F1-Score');
saveas(gcf, 'PSF1Score.png');
