# peer2pro

Een project om studenten aan mentoren te koppelen met behulp van matchinglogica en machine learning-modellen. De repository bevat datasets, matching-code en hulpmiddelen voor het trainen van modellen.

## Kort overzicht
- Doel: studenten koppelen aan mentoren op basis van kenmerken en tekstuele omschrijvingen.
- Bevat scripts voor het trainen van modellen, een matching-engine en voorbeelddatasets.

## Vereisten
- Python 3.11
- Installeer requirements:

```powershell
pip install -r requirements.txt
```

## Belangrijke paden
- `DATASETS/` - voorbeeld CSV-datasets (studenten, mentoren, matches, enz.)
- `matching/` - matching-engine code, asp clingo 
- `LOG REG/`, `XGBOOST/` - scripts voor modeltraining en experimenten
- `log_reg_library/` - 

## Projectstructuur (hoog niveau)

- `main.py` - headless matching
- `gui.py` - GUI matching
- `matching/` - kernimplementatie van de matching
- `log_reg_library/` - classificatie opdracht omschrijving
- `LOG REG/`, `XGBOOST/` - modelcode en trainingsscripts
- `DATASETS/` - CSV-bestanden die door scripts en notebooks worden gebruikt

## Gebruik via de command line
Het script `main.py` verwacht enkele command line-argumenten. Hieronder staan de belangrijkste opties en voorbeelden.

- `--students-input-path` (verplicht): pad naar CSV met studenten en omschrijvingen.
- `--mentors-type1-path` (verplicht): pad naar CSV met Type-1 mentoren.
- `--mentors-type2-path` (optioneel): pad naar CSV met Type-2 mentoren (gebruik samen met `--type2-n`).
- `--type1-n` (optioneel): aantal Type-1 mentoren per student (default: 4).
- `--type2-n` (optioneel): aantal Type-2 mentoren per student (vereist `--mentors-type2-path`).
- `--export-path` (optioneel): bestemming voor het geëxporteerde matches CSV (default: `./DATASETS/matches.csv`).
- `--timeout-seconds` (optioneel): timeout voor de solver in seconden (default: 120).
- `--no-progress` (flag): geen voortgangsweergave tijdens classificatie.
- `--quiet` (flag): onderdruk DataFrame-voorbeelden in de console-output.

Belangrijke validatie: als je `--type2-n` opgeeft, moet je ook `--mentors-type2-path` meegeven.

Voorbeelden (PowerShell):

Minimaal (vereiste paden):
```powershell
python main.py --students-input-path DATASETS/studenten.csv --mentors-type1-path DATASETS/mentoren.csv
```

Met Type-2 mentoren en extra opties:
```powershell
python main.py --students-input-path DATASETS/studenten.csv --mentors-type1-path DATASETS/mentoren.csv \
	--mentors-type2-path DATASETS/mentorenB.csv --type2-n 2 --type1-n 4 --export-path DATASETS/matches.csv --timeout-seconds 180
```

Gebruik `--no-progress` of `--quiet` als je minder console-output wilt.
