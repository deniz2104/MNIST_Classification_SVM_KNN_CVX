load('test_and_train.mat');

time=tic;
X1train = X_train;
y1train = Y_train;
y1train(y1train==1)=1; 
y1train(y1train ~= 1) = -1;
X1train=X1train(1:2000,:);
y1train=y1train(1:2000,:);

X1test=X_test;
y1test = Y_test;

c=1;
%in cazul unui c mic modelul de antrenare va permite mai multe erori de
%clasificare,dar o maximizare de marja
%se vor maximiza erorile de clasificare,dar vom avea marja mult mai
%mica,poate duce la overfitting,modelul invata bine datele de
%antrenament,nu va mai generaliza bine pe date noi

[w1 , b1] = classify(X1train,y1train,c);
res1 = (X1test*w1 +b1);

X2train = X_train;
y2train = Y_train;
y2train(y2train ~= 2) = -1;
y2train(y2train==2)=1;

X2train=X2train(1:2000,:);
y2train=y2train(1:2000,:);

X2test=X_test;
y2test = Y_test;

[w2 , b2] = classify(X2train,y2train,c);
res2 = (X2test*w2 +b2);

X3train = X_train;
y3train = Y_train;
y3train(y3train ~= 3) = -1;
y3train(y3train==3)=1;

X3train=X3train(1:2000,:);
y3train=y3train(1:2000,:);

X3test=X_test;
y3test = Y_test;

[w3 , b3] = classify(X3train,y3train,c);
res3 = (X3test*w3 +b3);

X4train = X_train;
y4train = Y_train;
y4train(y4train ~= 4) = -1;
y4train(y4train==4)=1;

X4train=X4train(1:2000,:);
y4train=y4train(1:2000,:);

X4test=X_test;
y4test = Y_test;


[w4 , b4] = classify(X4train,y4train,c);
res4 = (X4test*w4 +b4);

X5train = X_train;
y5train = Y_train;
y5train(y5train ~= 5) = -1;
y5train(y5train==5)=1;

X5train=X5train(1:2000,:);
y5train=y5train(1:2000,:);

X5test=X_test;
y5test = Y_test;

[w5 , b5] = classify(X5train,y5train,c);
res5 = (X5test*w5 +b5);

X6train = X_train;
y6train = Y_train;
y6train(y6train ~= 6) = -1;
y6train(y6train==6)=1;

X6train=X6train(1:2000,:);
y6train=y6train(1:2000,:);

X6test=X_test;
y6test = Y_test;

[w6 , b6] = classify(X6train,y6train,c);
res6 = (X6test*w6 +b6);

X7train = X_train;
y7train = Y_train;
y7train(y7train ~= 7) = -1;
y7train(y7train==7)=1;

X7train=X7train(1:2000,:);
y7train=y7train(1:2000,:);

X7test=X_test;
y7test = Y_test;

[w7 , b7] = classify(X7train,y7train,c);
res7 = (X7test*w7 +b7);

X8train = X_train;
y8train = Y_train;
y8train(y8train ~= 8) = -1;
y8train(y8train==8)=1;

X8train=X8train(1:2000,:);
y8train=y8train(1:2000,:);

X8test=X_test;
y8test = Y_test;

[w8 , b8] = classify(X8train,y8train,c);
res8 = (X8test*w8 +b8);

X9train = X_train;
y9train = Y_train;
y9train(y9train ~= 9) = -1;
y9train(y9train==9)=1;

X9train=X9train(1:2000,:);
y9train=y9train(1:2000,:);

X9test=X_test;
y9test = Y_test;

[w9 , b9] = classify(X9train,y9train,c);
res9 = (X9test*w9 +b9);

X0train = X_train;
y0train = Y_train;
y0train(y0train ~= 0) = -1;
y0train(y0train==0)=1;

X0train=X0train(1:2000,:);
y0train=y0train(1:2000,:);

X0test=X_test;
y0test = Y_test;

[w0 , b0] = classify(X0train,y0train,c);
res0 = (X0test*w0 +b0);

%Se fac predicțiile finale prin alegerea celei mai mari valori din rezultatele tuturor clasificatorilor pentru fiecare exemplu de testare.
%Valorile maxime reprezintă predicțiile finale pentru fiecare exemplu.

largest=0;
yPred = zeros(10000,1);
for i=1:10000
    v = [ res0(i); res1(i); res2(i); res3(i); res4(i); res5(i); res6(i); res7(i); res8(i); res9(i)];

    largest = max(v);
    
        if ( largest==res1(i) )
        yPred(i)=1;
        
    
        elseif  (largest==res2(i) )
        yPred(i)=2;
      
    
        elseif ( largest==res3(i) )
        yPred(i)=3;
        
        
        elseif ( largest==res4(i) )
        yPred(i)=4;
        
        
        elseif ( largest==res5(i) )
        yPred(i)=5;
        
        
        elseif ( largest==res6(i) )
        yPred(i)=6;
        
        
        elseif ( largest==res7(i) )
        yPred(i)=7;
        
        
        elseif (largest==res8(i) )
        yPred(i)=8;
        
        
        elseif ( largest==res9(i) )
        yPred(i)=9;
       
        
        else
        yPred(i)=0;
        end
        
end
toc;
disp('Timpul necesar:');
disp(time);

% Calculate accuracy
accuracy = sum(yPred == Y_test) / length(Y_test);
disp(['Acuratetea: ', num2str(accuracy)]);

% Confusion matrix
C = confusionmat(Y_test, yPred);
disp('Confusion Matrix:');
disp(C);

% Precision, Recall, and F1-score
numar_cifre = numel(unique(Y_test));
precizie = zeros(numar_cifre, 1);
sensibilitate = zeros(numar_cifre, 1);
f1Score = zeros(numar_cifre, 1);

for i = 1:numar_cifre
    truePositives = C(i, i);
    falsePositives = sum(C(:, i)) - truePositives;
    falseNegatives = sum(C(i, :)) - truePositives;
    
    precizie(i) = truePositives / (truePositives + falsePositives);
    sensibilitate(i) = truePositives / (truePositives + falseNegatives);
    f1Score(i) = 2 * (precizie(i) * sensibilitate(i)) / (precizie(i) + sensibilitate(i));
end

disp('Precizie:');
disp(precizie);
disp('Sensibilitate:');
disp(sensibilitate);
disp('F1-Score:');
disp(f1Score);

figure;
confusionchart(C);
title('Confusion Matrix');
saveas(gcf, 'ConfusionMatrix_svm_linear.png');

figure;
bar([precizie sensibilitate f1Score]);
title('Precizie,Sensibilitea, si F1-Score pentru fiecare cifra');
xlabel('Cifra');
ylabel('Scor');
legend('Precizie', 'Sensibilitate', 'F1-Score');
saveas(gcf, 'PSF1Score_svm_linear.png');

