# F1 Pit Stop Dataset Intelligence Report

- Train shape: (439140, 16)
- Test shape: (188165, 15)
- Target rate: 0.198982
- Duplicate train rows excluding id/target: 0
- Duplicate test rows excluding id: 0

## Column interpretation
- Categorical: Driver, Compound, Race; low-cardinality ordinal/state: Year, PitStop, Stint, Position.
- Cyclic/progression: RaceProgress is LapNumber divided by an inferred race length; lap sine/cosine and phase bins are useful.
- Tyre degradation proxies: TyreLife, LapTime_Delta, Cumulative_Degradation, degradation per tyre lap, stint-lap gap.
- Temporal ordering: Year/Race/Driver/LapNumber/Stint/TyreLife/id is the primary reconstructed sequence key.
- Synthetic fingerprints: id order, exact rational RaceProgress, high-cardinality Driver templates, and repeated group statistics.

## Top mutual information features

## Adversarial train/test drift

## Strong pairwise target interactions
             pair  groups      min      max      std
       Year|Stint      25 0.000000 0.569358 0.214352
       Race|Stint     123 0.000000 0.756017 0.180965
    Compound|Year      18 0.000000 0.538541 0.177922
        Race|Year     100 0.000000 0.760247 0.164712
    Race|Compound      87 0.008427 0.693132 0.153655
     Driver|Stint    1915 0.000000 0.775510 0.144471
    PitStop|Stint      14 0.000000 0.400441 0.143444
   Stint|Position     116 0.000000 0.455055 0.139820
      Driver|Year    1889 0.000000 0.620579 0.134614
   Compound|Stint      29 0.000000 0.447699 0.134336
     Year|PitStop       8 0.004154 0.304236 0.124601
    Year|Position      80 0.005754 0.349211 0.120743
  Driver|Compound    1767 0.000000 0.684211 0.114095
 Compound|PitStop      10 0.013889 0.340993 0.111544
      Driver|Race    8246 0.000000 0.909091 0.109942
Compound|Position      97 0.000000 0.374763 0.109143
    Race|Position     519 0.030303 0.470896 0.091485
   Driver|PitStop    1034 0.000000 0.611465 0.089854
     Race|PitStop      52 0.084585 0.397889 0.083092
  Driver|Position    6736 0.000000 0.756098 0.080071
 PitStop|Position      40 0.133375 0.300137 0.044468