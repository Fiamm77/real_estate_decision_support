# Ingatlanertekelesi magyarazati tudasbazis

## KSH piaci benchmark

A KSH telepulesi, varmegyei vagy orszagos fajlagos ingatlanarai kulon piaci
benchmarkkent jelennek meg. Ha telepulesi adat nem elerheto, a rendszer
fokozatosan magasabb szintu teruleti fallbacket hasznal.

## Ketkomponensu strukturalt score

A modell nem kozvetlen forinterteket tanul, hanem a DF-bol kepzett relativ
strukturalt poziciot. Kulon telek score es epület score keszul. A telek score
a `land_value / land_area_m2`, az epület score pedig a
`(building_value + renovation_cost) / building_area_m2` csoporton beluli
percentile rangja alapjan tanulhato jel.

## KSH korrekcios faktor

A vegso strukturalt score lakasoknal az epület score, lakohazaknal a telek
score es az epület score sulyozott atlaga. Ezt a rendszer 0.80 es 1.20 kozotti
KSH korrekcios faktorra kepezi le. A becsult piaci ertek a KSH benchmark es a
korrekcios faktor szorzata.

## Benchmark elteres

A benchmark elteres a becsult piaci ertek es a KSH benchmark kulonbsege.
Pozitiv ertek azt jelzi, hogy az ingatlan strukturalt jellemzoi a benchmark
folotti becslest tamogatnak. Negativ ertek benchmark alatti strukturalt
poziciot jelez.

## Felujitas utani szcenario

A felujitas utani ertek egy celallapotot feltetelez. A rendszer eloszor
felujitasi munkacsomagokbol koltseget becsul, majd az ingatlant a celallapot
mellett ujraertekeli. Uzleti minimum szabaly szerint a felujitas utani becsult
piaci ertek nem lehet alacsonyabb az aktualis becsult piaci erteknel.

## SHAP magyarazat

A SHAP ertekek azt mutatjak meg, hogy egy-egy feature az adott predikcio
eseteben novelte vagy csokkentette a modell becsleset. A nagy abszolut SHAP
erteku valtozok a predikcio legfontosabb lokalis magyarazo tenyezoi.

## Bizonytalansag

A becsles bizonytalansaga nagyobb lehet, ha a KSH adat csak magasabb teruleti
szinten erheto el, ha az ingatlan parameterei kilognak a tanito adatok tipikus
tartomanyabol, vagy ha hianyosak a bemeneti adatok.
