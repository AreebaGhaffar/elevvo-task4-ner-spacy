# Data exploration — CoNLL-2003 (English)

| split      |   sentences |   tokens |   entities |   PER |   ORG |   LOC |   MISC |   avg sentence length (tokens) |   max sentence length (tokens) |
|:-----------|------------:|---------:|-----------:|------:|------:|------:|-------:|-------------------------------:|-------------------------------:|
| train      |       14041 |   203621 |      23499 |  6600 |  6321 |  7140 |   3438 |                           14.5 |                            113 |
| validation |        3250 |    51362 |       5942 |  1842 |  1341 |  1837 |    922 |                           15.8 |                            109 |
| test       |        3453 |    46435 |       5648 |  1617 |  1661 |  1668 |    702 |                           13.4 |                            124 |

## Example sentences (train)

```
EU rejects German call to boycott British lamb .
entities: [('EU', 'ORG'), ('German', 'MISC'), ('British', 'MISC')]
```

```
The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep .
entities: [('European Commission', 'ORG'), ('German', 'MISC'), ('British', 'MISC')]
```

## Example sentences (validation)

```
West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship .
entities: [('West Indian', 'MISC'), ('Phil Simmons', 'PER'), ('Leicestershire', 'ORG'), ('Somerset', 'ORG')]
```

```
Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .
entities: [('Essex', 'ORG'), ('Derbyshire', 'ORG'), ('Surrey', 'ORG'), ('Kent', 'ORG'), ('Nottinghamshire', 'ORG')]
```

## Example sentences (test)

```
SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .
entities: [('JAPAN', 'LOC'), ('CHINA', 'PER')]
```

```
AL-AIN , United Arab Emirates 1996-12-06
entities: [('AL-AIN', 'LOC'), ('United Arab Emirates', 'LOC')]
```
