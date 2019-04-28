# Arbeitskonzept von Luis Wirth

## Titel der Arbeit

“Bildsynthese von menschlichen Gesichtern mit KI”

## Leitfrage

“Wie können mithilfe von Machine Learning Bilder von künstlichen menschlichen
Gesichtern generiert werden?”

## Gegenstand der Untersuchung

Machine Learning ist momentan ein sehr aktueller Themenbereich der viele neue
Möglichkeiten bietet. Es vereint die beiden Gebiete der Informatik und der Mathematik, um
komplexe Problemstellungen zu lösen. Die vorliegende Arbeit soll beispielhaft eine konkrete
Anwendung von Machine Learning aufzeigen, namentlich die künstliche Bildgenerierung von
menschlichen Gesichtern.

Zu diesem Zweck gelangen künstliche Neuronale Netze zum Einsatz. Dabei wird
der Fokus auf einer spezifischen Architektur liegen: den Autoencodern. Um das Vorgehen
zu verstehen, wird die relevante Theorie erläutert.

## Fachliche Verfahren

Für das Programmieren der Applikation, welche die Bildgenerierung ausführen soll, wird
Python3 verwendet. Mithilfe von Tensorflow, ein von Google entwickeltes Framework für
datenstromorientierte Programmierung, wird das Neuronale Netz implementiert. Keras, eine
Schnittstelle von Tensorflow, wird das Erstellen des Computational Graphs des Neuronalen
Netzes erleichtern.

Dank Tensorflow und Keras müssen die komplexen Verfahren zum Trainieren von Neu-
ronalen Netzen (Gradient Descentin Kombination mit Linearer Algebra) nicht selbst
implementiert werden. Jedoch wird die Maturarbeit die dazugehörige Theorie darlegen.

## Ressourcen

Für das Trainieren des Neuronalen Netzes ist ein leistungsstarker Computer (d.h. mit guter
GPU) von Vorteil.

Des weiteren werden Trainingsdaten für das Trainieren des Neuronale Netz benötigt.
In diesem Fall handelt es sich um Fotos von menschlichen Gesichtern, welche alle ein ähnliches
Format und einen möglichst uniformen Hintergrund haben sollten. Dies könnten zum Beispiel
Jahrbuchbilder einer Schule sein oder herausgefilterte Bilder aus dem Internet.

Informationsquelle wird vor allem das Internet sein, da dieses Gebiet schwerpunktmässig
dort behandelt wird. Für die notwendige Theorie zu Gradient Descent und dergleichen werden
passende Bücher konsultiert.


## Zielsetzung

Das Ziel der Arbeit ist es, die Grundlagen und Möglichkeiten von Machine Learning anhand
eines konkreten Anwendungsbeispiel darzulegen. Dabei soll ebenfalls aufgezeigt werden,
welcher Aufwand mit einem solchen Projekt verbunden ist. Die Arbeit kann deshalb als
Orientierungshilfe für ähnliche Projekte dienen.

Gleichzeitig soll verdeutlicht werden, dass es auch für programmieraffine Laien grundsätzlich möglich ist,
Machine Learning auf komplexe Problemstellungen anzuwenden.

Durch die Identifikation von verbindenen Merkmalen eines Datensatzes, ist es möglich,
neue Daten mit den gleichen Merkmalen zu generieren. Dieses Prinzip hat unzählige Anwendungsmöglichkeiten!
Einige davon werden in der Arbeit erläutert werden. Autoencoder weisen
zudem weitere interessante Aspekte auf, von denen einige in der Arbeit behandelt werden.
