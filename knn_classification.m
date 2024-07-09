load("test_and_train.mat");
timp=tic;
X_antrenare = X_train; 
y_antrenare = Y_train;

X_testare = X_test(1:5000,:);
y_testare = Y_test(1:5000,:);

k = 5;

%realizare de predictii pe setul meu de testare
num_test_samples = size(X_testare, 1);
predictii = zeros(num_test_samples, 1);

for i = 1:num_test_samples
    predictii(i) = predictLabel(X_antrenare, y_antrenare, X_testare(i, :), k);
end
toc;
disp('Timpul de realizare a metodei:');
disp(timp);

% Calculate accuracy
acuratete = sum(predictii == y_testare) / length(y_testare);
disp(['Acuratete: ', num2str(acuratete)]);

% Confusion matrix
C = confusionmat(y_testare, predictii);

numar_cifre = numel(unique(y_testare));
precizie = zeros(numar_cifre, 1);
sensibilitatea = zeros(numar_cifre, 1);
f1Score = zeros(numar_cifre, 1);

for i = 1:numar_cifre
    truePositives = C(i, i);
    falsePositives = sum(C(:, i)) - truePositives;
    falseNegatives = sum(C(i, :)) - truePositives;
    
    precizie(i) = truePositives / (truePositives + falsePositives);
    sensibilitatea(i) = truePositives / (truePositives + falseNegatives);
    f1Score(i) = 2 * (precizie(i) * sensibilitatea(i)) / (precizie(i) + sensibilitatea(i));
end

disp('Precizia:');
disp(precizie);
disp('Sensibilitatea:');
disp(sensibilitatea);
disp('F1-Score:');
disp(f1Score);

% Plot and save confusion matrix
figure;
confusionchart(C);
title('Confusion Matrix');
saveas(gcf, 'ConfusionMatrix_knn.png');

% Plot and save precision, recall, and F1-scores
figure;
bar([precizie sensibilitatea f1Score]);
title('Precizie,Sensibilitea, si F1-Score pentru fiecare cifra');
xlabel('Cifra');
ylabel('Scor');
legend('Precizie', 'Sensibilitate', 'F1-Score');
saveas(gcf, 'PSF1Score_knn.png');

%functie pentru calcularea distantei
function d = distanta(X1, X2)
    d = sqrt(sum((X1 - X2) .^ 2, 2));
end

%functie pt gasilor celor mai apropiati k vecini
function gasire_vecini = findNeighbors(X_train, X_test_row, k)
    %calculeaza distanta dintre punctul de testate si punctul de antrenare
    distante = distanta(X_train, X_test_row);
    %sortez distantele in ordine crescatoare si retin indicii
    [~, indici] = sort(distante, 'ascend');
    %selectarea primelor k indici,referitor la cei mai apropiati vecini
    gasire_vecini = indici(1:k);
end

%functie pentru prezicerea clasei
function label = predictLabel(X_train, y_train, X_test_row, k)
    %gasirea indicilor celor mai apropiati vecini ai punctului de testare
    neighbors = findNeighbors(X_train, X_test_row, k);
    %extragem etichetele
    neighbor_labels = y_train(neighbors);
    %determinarea etichetei prezise pe baza celei mai frecvente etichete a
    %vecinilor
    label = mode(neighbor_labels);
end
