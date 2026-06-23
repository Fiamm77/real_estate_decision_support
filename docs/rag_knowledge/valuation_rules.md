# Ingatlanertekelesi magyarazati tudasbazis

## KSH piaci benchmark

A KSH telepulesi, varmegyei vagy orszagos fajlagos ingatlanara adja az
aktualis piaci árszintet. Ha telepulesi adat nem erheto el, a rendszer
magasabb teruleti szintu fallbacket hasznal.

## Ketkomponensu strukturalt score

A modell nem kozvetlen forinterteket tanul. A synthetic DF alapjan relativ
telek score-t es epület score-t becsul, majd ezekbol kepzi az ingatlan
strukturalt poziciojat.

## KSH korrekcios faktor

A strukturalt score 0.80 es 1.20 kozotti KSH-korrekcios faktorra kepzodik le.
A becsult piaci ertek a KSH benchmark es a korrekcios faktor szorzata.

## Benchmark elteres

A benchmark elteres a becsult piaci ertek es a KSH benchmark kulonbsege.
Pozitiv elteres benchmark feletti, negativ elteres benchmark alatti
strukturalt poziciot jelez.

## Felujitas utani szcenario

A felujitas utani scenario a celallapot mellett ujraprediktalt strukturalt
score-ra epul. A kimenet a felujitas utani becsult piaci erteket es az
aktualis ertekhez kepesti erteknovekmenyt mutatja.

## SHAP magyarazat

A SHAP ertekek azt jelzik, hogy az adott predikcioban mely feature-ok noveltek
vagy csokkentettek leginkabb a becslest.

## Bizonytalansag

A becsles bizonytalanabb lehet, ha csak magasabb teruleti KSH fallback erheto
el, vagy az ingatlan parameterei kilognak a tanito adatok tipikus
tartomanyabol.
