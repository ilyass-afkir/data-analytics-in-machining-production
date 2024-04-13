Alle Skripte können in nummerierter Reihenfolge ausgeführt werden.
Die .csv Dateien des Koordinatenmessgeräts und der Fräsmaschine müssen in den Ordnern "cmm_data" bzw. "sliced_workpieces" gespeichert werden.
Neu definierte Funktionen, die in mehreren Skripten verwendet werden, sind in der Datei Functions.py gespeichert.

- Skript 01 lädt alle Daten, führt die Dateien der Fräsmaschine und des Koordinatenmessgeräts zusammen und speichert den neuen Datensatz im .pkl Format.
- Skript 02 (optional) zur Durchführung einer explorativen Datenanalyse, verschiedene Grafiken werden erstellt und gespeichert.
- Skript 03 vergleicht die Performance von 3 ausgewählten Regressionsalgorithmen für jedes einzelne geometrische Feature anhand 3 unterschiedlicher Bewertungskriterien. Am Ende wird für jeden Regressionsalgorithmus eine Konfusionsmatrix erstellt und gespeichert.
  - Achtung: Die Ausführung dieses Skripts kann einige Stunden in Anspruch nehmen (Grund dafür ist der GridSearch).   
- Skript 04 zeigt die Ergebnisse des besten Models (KNN), die Predictive vs. True Plots und die Feature Selection. 

Unabhängig von den übrigen Skripten:
- Skript 05 implementiert das gespeicherte finale ML-Modell (kNeighbor_corrvalue0.7.pickle) aus Skript 04, um dieses anschließend mit neuen Maschinendaten testen zu können. 
Datensätze zum Testen, die nur aus Maschinendaten bestehen, müssen dazu im Ordner "test_workpieces" abgelegt werden.
