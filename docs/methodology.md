# Méthodologie

## Contrat de données

### Trafic horaire (issue #6)

**Fichier.** `data/processed/traffic/year=YYYY/part-0.parquet` : Parquet compressé (zstd), un fichier par année. Produit par `python -m traffic_prediction.data.build` (voir le README), non versionné.

Lecture : `pd.read_parquet("data/processed/traffic")`. Pandas ajoute une colonne `year` déduite du nom du dossier ; elle sert de filtre, par exemple `filters=[("year", "=", 2025)]`.

**Grille.** Une ligne par arc et par heure, de `--start` à 00:00 jusqu'à `--end` à 23:00 inclus (par défaut du 2024-01-01 au 2025-12-31). La clé (`arc`, `datetime`) est unique. Les arcs sont listés dans `configs/baseline.yaml`, section `traffic` : les arcs évalués (`evaluated_arcs`) et des arcs proxy (`proxy_arcs`).

| Colonne | Type | Unité | Contenu |
|---|---|---|---|
| `arc` | string | | Nom de l'arc. Pour les arcs évalués, le nom imposé dans le fichier de rendu : `Champs-Elysées`, `Convention`, `Saint-Pères`. |
| `role` | string | | `evalue` (arc noté par BCG X) ou `proxy` (arc voisin de substitution). |
| `iu_ac` | int32 | | Identifiant Open Data du tronçon. |
| `datetime` | timestamp sans fuseau | | Heure de la mesure, voir la convention ci-dessous. |
| `debit_horaire` | float64 | véhicules par heure | `q` d'Open Data. |
| `taux_occupation` | float64 | % du temps, de 0 à 100 | `k` d'Open Data. |
| `etat_trafic` | int8, nullable | code | 0 Inconnu, 1 Fluide (k < 15), 2 Pré-saturé (15 ≤ k < 30), 3 Saturé (30 ≤ k < 50), 4 Bloqué (k ≥ 50). |
| `etat_barre` | int8, nullable | code | 0 Inconnu, 1 Ouvert, 2 Barré, 3 Invalide. À ne pas utiliser comme variable de modèle (voir les constats). |
| `qualite_mesure` | string | | Qualité de la ligne, voir ci-dessous. |

Les valeurs manquantes sont NaN pour les mesures et `<NA>` pour les états. Aucune valeur n'est imputée.

**Convention d'heure.** `datetime` reprend `t_1h` d'Open Data sans le modifier : c'est l'heure **locale de Paris, sans fuseau**, et elle marque la **fin** de l'heure mesurée. D'après la notice, « 2019-01-01 01:00:00 » couvre de 00:00 à 01:00. Comme dans la source, chaque jour compte 24 heures :

- Au passage à l'heure d'été, la source contient une ligne pour l'heure qui n'existe pas, souvent vide. Elle est gardée.
- Au retour à l'heure d'hiver, l'heure 02:00, qui se produit deux fois, n'apparaît qu'une fois dans la source. L'une des deux heures réelles est perdue.
- Les heures de l'API récente sont en UTC avec fuseau. `normalize_raw` les convertit en heure de Paris sans fuseau. Sur la période où les deux sources se recouvrent (septembre à décembre 2025, Convention et Saint-Pères), les valeurs concordent à 100 % à la même étiquette d'heure, sans aucun décalage.
- Les heures ne sont pas décalées pour suivre le format de rendu : on ne sait pas encore si BCG X désigne le début ou la fin de l'heure. La question reste ouverte.
- Quand la période commence le 1er janvier, les archives lisibles de l'année précédente sont également recherchées pour récupérer la ligne de minuit. Une archive précédente manquante ou de compression non prise en charge produit un avertissement. L'archive 2023 est maintenant lisible avec le décodeur Deflate64 optionnel ; sans cette archive ou une autre source fournie contenant la mesure, `2024-01-01 00:00` reste `absente`. Les lignes hors période sont comptées dans le résumé.

**Qualité de la mesure (`qualite_mesure`).**

| Valeur | Règle |
|---|---|
| `ok` | `q` et `k` présents. |
| `partielle` | Une seule des deux mesures est présente. |
| `vide` | La ligne existe dans la source, mais `q` et `k` sont vides. |
| `invalide` | Au moins une valeur impossible, remplacée par NaN : valeur non numérique, `q < 0`, `k < 0`, `k > 100`, ou `q = 0` alors que `k > 0`. Dans ce dernier cas, aucun véhicule n'est compté alors que la boucle est occupée : c'est un défaut du capteur, donc `q` passe à NaN et `k` est gardé. |
| `absente` | L'heure manque dans la source ; elle a été ajoutée par la grille. |

`etat_barre` n'entre pas dans le calcul de la qualité.

Les mesures infinies sont également invalides et remplacées par NaN. Un état fourni avec un code ou un libellé inconnu arrête le traitement ; un état manquant reste manquant.

**Doublons.**

- Une archive identique à une autre (même SHA-256) n'est lue qu'une fois. C'est le cas de 2025, que le portail liste deux fois.
- Si plusieurs lignes portent le même couple (`arc`, `datetime`) avec des valeurs identiques, une seule est gardée. Si leurs valeurs diffèrent, on garde la dernière ligne entière dans l'ordre de lecture, y compris ses valeurs manquantes. Les deux cas sont comptés dans le résumé.
- Les archives sont lues par ordre de nom. Leurs membres sont lus par ordre de nom pour les ZIP ordinaires, et dans l'ordre physique de l'archive pour Deflate64 afin de décompresser en flux. Les snapshots JSONL sont lus ensuite, dans l'ordre explicite des arguments `--recent-file`. Leur nom aléatoire ne permet pas de déduire leur date d'extraction : aucun tri automatique ni mélange des valeurs de plusieurs lignes n'est effectué.

**Filtre et garde-fous.**

- Un arc est défini par son couple (nœud amont, nœud aval), dans cet ordre. Le sens inverse est un autre arc, avec un autre `iu_ac`.
- Si un couple de nœuds configuré apparaît avec un autre `iu_ac` que celui attendu, le traitement s'arrête.
- Un horodatage manquant ou qui ne correspond pas à une heure entière de Paris arrête le traitement avant la création de la grille, pour éviter de perdre silencieusement une observation.
- Un arc sans aucune mesure sur la période ne bloque pas le traitement : il produit un avertissement, il est compté, et ses heures restent NaN.
- Rien n'est écarté sans être signalé. Le résumé de fin compte les lignes lues, gardées, hors période, en double et invalides, et donne la part de valeurs manquantes par arc, par variable et par mois.

**Validation.** Avant l'écriture, le traitement vérifie les colonnes et leur ordre, le nombre de lignes (arcs × heures), l'absence de doublons et les plages de valeurs. Après l'écriture, il relit le schéma et le nombre de lignes de chaque fichier. Le traitement est déterministe : deux exécutions produisent des fichiers identiques.

## Décisions

- **Heure** : on garde `t_1h` tel quel (heure de Paris sans fuseau, fin de l'heure). On n'ajoute pas de colonne UTC : les heures du changement d'heure ne sont pas convertibles sans ambiguïté, et la source compte 24 heures par jour.
- **Sortie** : `data/processed/traffic/`, partitionnée par année. `build.py` contient une fonction par source (`build_traffic` pour l'instant). `traffic.py` réunit l'extraction API récente (issue #5) et les fonctions de lecture et de mise au format (issue #6). Sa commande en ligne extrait les données brutes ; celle de `build.py` traite les archives historiques et les snapshots JSONL explicitement fournis avec `--recent-file`. Une construction remplace le fichier complet de chaque année concernée, même si la période demandée ne couvre qu'un jour. Utiliser un autre dossier de sortie pour les essais ; la mise à jour incrémentale reste un travail distinct.
- **Arcs proxy** pour les Champs-Élysées : 4272 (La Boétie → Berri) et 4265 (Washington → Balzac), parce que l'arc évalué n'a plus de mesures depuis novembre 2025.
- **2023 / Deflate64** : l'ingestion et le traitement utilisent `stream-unzip` à la demande pour cette compression. Lecture par blocs sans extraction ; tailles et CRC sont contrôlés, y compris pour les membres ignorés par le traitement. `zipfile` reste utilisé seul pour les archives ordinaires. Les dépendances optionnelles et leur installation sont indiquées dans le README.

## Constats d'exploration

### Données trafic 2024–2025 (issue #6)

- **Les manques viennent par blocs de mois** (pannes de capteurs), pas au hasard :
  - Saint-Pères : de décembre 2024 à septembre 2025 ;
  - Convention : avril–mai 2024, septembre–novembre 2024, mars–mai 2025 ;
  - Champs-Élysées : novembre et décembre 2025.

  Pour le détail mois par mois, lancer la commande de traitement : son résumé l'affiche.
- **Champs-Elysées : dernière mesure le 2025-11-19 à 12:00.** Ni l'archive ni l'API (consultée jusqu'au 29/10/2026) ne contiennent de mesure plus récente, et aucun autre `iu_ac` n'a repris le couple Washington → Berri.
- **511 heures n'ont de données pour aucun capteur en 2024**, du 28/03 au 01/04 et fin novembre.
- **`etat_barre` est incohérent entre les deux sources.** Dans l'archive, il vaut 3 (« Invalide ») sur 99 % des lignes, y compris quand les mesures sont normales. Il vaut 2 (« Barré ») sur les Champs-Élysées lors de fermetures (14/07, 31/12, etc.). Pour les mêmes heures, l'API donne 1 (« Ouvert ») dès qu'il y a une mesure. **Il ne faut pas l'utiliser comme variable de modèle.**
- **`etat_trafic` se déduit entièrement de `k`** par les seuils de la notice : il vaut 0 si et seulement si `k` est vide. Il n'apporte rien de plus que `k`.
- **Proxy La Boétie-Berri (4272)** : on n'y trouve quasiment que le taux d'occupation. `q` manque à 96,6 % et n'est présent qu'en janvier 2024. Question ouverte : faut-il le garder comme proxy de taux seulement, ou chercher un autre proxy ? Candidats vus dans l'API en 2026, en part d'heures avec `k`, sans avoir vérifié `q` : 4274 (Berri → La Boétie, sens inverse, environ 75 %) et 4267 (Balzac → Washington, sens inverse, environ 47 %).
- **Proxy Washington-Balzac (4265)** : il est très dégradé à partir d'avril 2025. `q` manque à 85–100 % chaque mois, et `k` manque à 100 % de juillet à novembre 2025.
- **`q = 0` alors que `k > 0`** : 98 heures sur La Boétie-Berri et 101 sur Washington-Balzac, aucune sur les arcs évalués. Ces lignes sont marquées `invalide`.
- **L'archive 2023 est compressée en Deflate64**, illisible par `zipfile` seul. Cette limite est désormais prise en charge par le lecteur en flux optionnel.
- **API récente (pour l'issue #5)** :
  - elle commence le 01/09/2025 et recouvre donc l'archive jusqu'à fin 2025 ;
  - elle a elle aussi des heures absentes ;
  - ses heures sont en UTC avec fuseau ;
  - `etat_trafic` et `etat_barre` y sont en texte ;
  - `libelle` y est écrit différemment, mais les nœuds sont identiques.

## Choix du modèle
