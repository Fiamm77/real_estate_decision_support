# Ingatlanértékelési magyarázati tudásbázis

## KSH piaci benchmark

A KSH települési, vármegyei vagy országos fajlagos ingatlanára szolgál a piaci
árszint referenciaértékeként. Amennyiben települési szintű adat nem érhető el,
a rendszer magasabb területi szintű referenciaértéket, vagyis fallbacket
használ.

## Kétkomponensű strukturális score

A modell nem közvetlenül piaci értéket becsül. A synthetic tanítóadatok alapján
külön telek-score-t és épület-score-t számít, majd ezekből képezi az ingatlan
strukturális pozícióját leíró összesített score-t.

## KSH-korrekciós faktor

A strukturális score egy 0,80 és 1,20 közötti korrekciós faktorra képződik le.
A becsült piaci érték a KSH benchmark és a korrekciós faktor szorzataként áll
elő.

## Benchmark-eltérés

A benchmark-eltérés a becsült piaci érték és a KSH benchmark különbsége.
Pozitív érték esetén az ingatlan a benchmarknál kedvezőbb, negatív érték
esetén annál kedvezőtlenebb strukturális pozícióval rendelkezik.

## Felújítás utáni szcenárió

A felújítás utáni szcenárió a célállapot mellett újraszámított strukturális
score alapján készül. A rendszer megmutatja a felújítás utáni becsült piaci
értéket, valamint az aktuális állapothoz képest várható értékváltozást.

## SHAP-magyarázat

A SHAP-értékek azt mutatják meg, hogy az egyes bemeneti jellemzők milyen
mértékben járultak hozzá az adott becsléshez. A pozitív SHAP-érték növeli, a
negatív SHAP-érték csökkenti a prediktált értéket.

## Bizonytalanság

A becslés bizonytalansága növekedhet, ha csak magasabb területi szintű KSH
referenciaérték áll rendelkezésre, illetve ha az ingatlan jellemzői jelentősen
eltérnek a tanítóadatokban megfigyelt tipikus mintázatoktól.
