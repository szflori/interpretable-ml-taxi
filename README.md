# Interpretable Machine Learning Analysis of NYC Taxi Trip Duration

## Bemutatás

A projekt alapjául a New York-i Taxi és Limuzin Bizottság (Taxi and Limousine Commission, TLC) által közzétett nyilvános
adatok szolgálnak. A TLC New York városában működő taxis és egyéb személyszállító járművek engedélyezéséért és
felügyeletéért felelős szervezet. Az általa publikált adathalmazok nagy mennyiségű, valós közlekedési adatot
tartalmaznak, ezért jól használhatók gépi tanulási feladatokhoz.

A nyilvánosan elérhető taxiút-nyilvántartások többek között információt adnak az utazások kezdési és befejezési
időpontjáról, az indulási és érkezési helyről, a megtett távolságról, az utasszámról, valamint bizonyos pénzügyi
adatokkal is kiegészülnek, például a viteldíjjal és a fizetési móddal. Ezek az adatok megfelelő alapot biztosítanak egy
olyan modell létrehozásához, amely képes megbecsülni egy taxiút várható időtartamát.

## Adatok

Az adathalmaz több olyan változót tartalmaz, amelyek alkalmasak lehetnek a predikciós feladathoz. Ilyenek például:

- az utasfelvétel dátuma és időpontja,
- az utasleadás dátuma és időpontja,
- az indulási hely,
- az érkezési hely,
- a megtett távolság,
- az utasszám,
- a fizetési mód,
- a viteldíj és egyéb kapcsolódó összegek.

## Cél

Egy olyan gépi tanulási modell készítése, amely képes megbecsülni egy taxi út várható időtartamát az utazás alapvető
jellemzői alapján.

A célváltozó a taxi út időtartama: `trip_duration_minutes = dropoff_datetime − pickup_datetime`

A létrehozott predikciós modell ellenőzése és vizsgálata a cél, hogy tudjuk, hogy milyen tényezők alapján hozza meg a
döntéseit és, hogy ezek a döntések mennyire megbizhatóak és átláthatóak.

## Interpretálhatóság fontossága

Megvizsgálható, hogy a modell milyen jellemzőket tekint meghatározónak az utazási idő becslése során, megértsük, hogy mi
alapján hozza a döntéseit így az esetleges hibákat észrevegyük és javítsunk a torzitáson. Ez különösen fontos akkor, ha
a modellt valós döntések támogatására szeretnénk használni. Fontos, hogy a modell által adott becslések megbízhatóbbak
és hitelesebbek legyenek, valamint jobban megérthetővé váljon, hogy a predikciók milyen tényezők hatására születnek meg.

## Forrás

Link: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## Használat

### Analyze notebook

1. https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2009-01.parquet -> a letöltött fájlt helyezd a
   raw_data/2009/-ba
2. Futtasd `yellow_2009_analyze.ipynb`-t

### Preprocessing

1. Tölsd le az 2009-évi adatokat helyezd a raw_data/2009/-ba
2. Futtasd `yellow_taxi_preprocessing.ipynb`-t elkészíti a tiszta adatokat a `clean_data`-ba helyezzi