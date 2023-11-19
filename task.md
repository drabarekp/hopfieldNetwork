# Drugi projekt: pamięć skojarzeniowa z użyciem sieci Hopfielda (10 punktów)

Temat będzie wyjaśniony i omówiony na zajęciach, na których oddawane są raporty z drugiej części pierwszego projektu.

Zaimplementować pamięć skojarzeniową z użyciem sieci Hopfielda uczonej dwiema metodami: **regułą Hebba i regułą Oji** 
(imię nazwisko w mianowniku: Erkki Oja).

Sieć powinna pozwalać na **łatwą konfigurację liczby neuronów** i **udostępniać wizualizację bieżącego stanu w formie 
prostokątnej bitmapy**. Wizualizacja powinna **pozwalać na obejrzenie kolejnych kroków zbiegania sieci do stanu 
stabilnego**. Zaimplementować **dwa warianty symulacji: synchroniczny i asynchroniczny**.

## Zagadnienia do sprawdzenia:

- skuteczność każdej z reguł uczenia i jej wpływ na liczbę stabilnych wzorców uczących,

- na początku projektu zostanie udostępniony zbór wzorców. Których spośród dostarczonych zbiorów sieć uczy się 
  poprawnie, a których nie? W których przypadkach wzorce uczące nie są stanami stabilnymi i dlaczego?
  (Uzasadnienie w raporcie odnieść do istniejących prac na temat sieci Hopfielda).

- pośród zbiorów są dwa, które zawierają prawie identyczne wzory (jeden w wersji 25×25 i drugi w wersji 25×50). 
  Dlaczego skuteczność odzyskiwania różni się znacząco na tych zbiorach?

- jaka jest skuteczność odzyskiwania oryginałów z uszkodzonego wejścia w zależności od liczności i rodzaju zbioru?

- porównać skuteczność na zbiorze, w którym są grupy elementów dość podobnych i takim, który jest całkiem zróżnicowany, 
  gdy zbiory są o podobnej liczności?

- jak nauczona sieć reaguje na podanie losowego wejścia?

- zaproponować zbiór wektorów o długości 25 (5×5), możliwie liczny, taki, żeby w nauczonej sieci wszystkie wzorce 
  z tego zbioru były stabilne.

- zaproponować zbiór uczący i wzorzec wejściowy, którego podanie do sieci zakończy się oscylacją między dwoma stanami.

- przeprowadzić eksperyment dla zbioru złożonego z dużych bitmap (co najmniej 200 × 300), 
  na przykład odpowiednio skonwertowanych zdjęć kotów.

### Uwaga!
Raport, poza informacjami jak działają reguły uczenia, powinien zawierać też wyjaśnienie, dlaczego 
te metody uczenia działają.