# Descriere:
  Clasificator pentru cifre scrise de mana utilizand setul de date MNIST folosind multiple metode de clasificare pentru a vizualiza indicii de performanta.Fiecare cifra este formata dintr-un grid de 28x28 in interiorul setului de date.

# Motivatia:
  Utilizarea SVM pentru un set de date cu cifre poate prezenta multe avantaje:
  - Separare optima intre clase:SVM gaseste o margine maximala intre clase,deci,poate fi benefic petru a separa cifre foarte asemanatoare (de exemplu, 3 si 5).
  - Performanta buna: Avand in vedere ca MNIST este format din 60000 de imagini de antrenament,poate avea rezultate bune comparativ cu alte metode,folosind o abordare cu kernel linear.
# Implementare:
  ## Preprocesarea datelor:
  - Citirea matricilor de testare si de antrenare.
  - Combinarea acestora intr-o singra matrice salvata si folosita pentru implementare.
  ## Atrenarea SVM-ului cu kernel liniar:
  - Functia `templeteSVM(...)` formeaza un sablon care specifica modul in care se va configura clasificatorul folosit ca functie de baza in `fitcecoc(...)`.
  - Kernelul liniar folosit pentru ca lucram cu date separabile liniar.
  - Setam parametrul de regularizare la 1.
  - `fitcecoc(...)` care are parametrii:
    - X_train reprezinta datele de antrenare  
    - Y_train etichetele
    - 'Learners', template indică faptul că fiecare clasificator binar folosește șablonul definit mai sus.
    - 'Coding', 'onevsone' specifică codificarea "unu-la-unu", ceea ce înseamnă că pentru fiecare pereche de clase se antrenează un model SVM separat. Este o metodă eficientă pentru clasificarea multi-clasă.
    - 'ClassNames' defineste clasele care trebuie clasificate
  ## SVM folosind CVX
  - In urma implementarii CVX-ului care reprezinta functia obiectiv am luat fiecare cifra in parte,am impartit componentele vectorului de antrenare in etichete binare(-1 si 1).
  - Folosind etichete binare transform problema dintr-o clasificare multipla intr-o serie de clasificari binare.
  - Din motive computationale se selecteaza primele 2000 de exemple pentru antrenament.
  - Folosesc un vector pentru toate clasele si stochez valorile prin selectarea scorului maxim din rezultatele obtinute anterior.
  ## Implementarea KNN-ului
  - Calculam distanta euclidiana intre respectivul punct si toate celelalte din setul de antrenare.
  - In urma calcularii si sortarii lor, algoritmul selecteaza cei mai apropiati 'k' vecini.
  - Se vor folosi etichetele pentru punctul de testare prin votare majoritara.
# Performante:
  ## Acuratetea :
  - SVM cu functii predefinite: 94.57%
  - CVX: 83.84%
  - KNN: 95.46%
    
  ## Motivul acuratetei mai scazute pentru CVX: 
  - CVX e folosit pentru optimizare convexa si nu este specializat pentru SVM,ceea ce poatea afecta performanta.

  ## Motivul pentru acuratetile marite:
  - SVM predefinit:
      - Utilizarea `fitcsvm(---)` si `fitcecoc(...) ` ofera acuratate ridicata datorita optimizarii performantei.
      - Functiile predefinite sunt optimizate pentru stabilitate numerica si eficienta,riscul de overfitting fiind mic si obtinand o margine optima intre clase.
  - KNN:
      - Acuratatea mare sugereaza ca datele sunt bine separate in spatiul caracteristicilor, un avantaj pentru metoda KNN.
      - Putin predispus la greseli de clasificare daca nu exista zgomot asupra datelor.
