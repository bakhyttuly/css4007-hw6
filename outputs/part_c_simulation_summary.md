| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| paper | 11 | 47 | 3 | 0 | 5.04 | 2.2297 |
| paper | 22 | 44 | 6 | 0 | 4.96 | 1.4424 |
| paper | 33 | 42 | 8 | 0 | 4.88 | 0.97 |
| extended | 11 | 50 | 0 | 0 | 5.0 | 1.502 |
| extended | 22 | 50 | 0 | 0 | 5.0 | 1.502 |
| extended | 33 | 50 | 0 | 0 | 5.0 | 1.502 |

Overall (150 episodes per map):
- paper: mean discounted return 1.5474 vs V*(start) = 1.5358; episodes that stepped on the ice tile: 150; episodes ending in the hole: 17 (of these 17 had visited the ice tile)
- extended: mean discounted return 1.5020 vs V*(start) = 1.5020; episodes that stepped on the ice tile: 0; episodes ending in the hole: 0 (of these 0 had visited the ice tile)
