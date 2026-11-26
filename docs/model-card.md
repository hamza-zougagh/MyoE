# Model card

## Model and intended use

MyoE is an original E CNN for research on personalization of recorded surface EMG. A source-pretrained encoder is adapted with labeled target calibration examples, followed by a separately fitted linear classifier. It predicts 16 gesture classes. It is not a medical device, diagnostic tool, prosthetic controller or validated live interface.

The bundled source checkpoint is the seed-2101, 20,000-update Session 1 source encoder from the 18-person FC2 pool. The source checkpoint contains no personalized classifier. The original checkpoint byte SHA-256 is `b14dcd6b09fb7bbf322ce36290c69522bebc73eb2b8233ea5576fd61a3ccc1ad`. Its historical model-state digest is `46efd84b1f06befb2e2c9ffa66dc37977cbb6d6a8e57e62d951ae99c88204178`. The repackaged `source-2101.npz` SHA-256 is `1cc9a0afec84abebd0e0b9526c5d521ccbf382e98a5997722a8b6c81404477df`. Tensor identity must be verified against the source checkpoint as part of release checks.

## Evaluation evidence

GRABMyo Session 1 exploratory Study 2 evaluated 25 people disjoint from the 18-person source pool. E macro-F1 was 0.6447 and 0.7476 at calibration budgets one and three, averaging the specified draws and three seeds. Matching A/C comparisons favored E under the exploratory protocol.

A prospective Session 2 A/C/E replication evaluated the same 25 targets on a different recording day using the original source encoders. E averaged 0.6846 and 0.7786 macro-F1 at budgets one and three. Each of the four prespecified E−A and E−C adjusted 98.75% interval lower bounds exceeded +0.01. These results support the specified within-cohort cross-session comparisons. They do not establish performance in a newly recruited cohort, different devices, electrode layouts or patients. Both studies aggregate three source seeds and calibration draws, so the means are not estimates for the seed-2101 checkpoint alone.

The E-V2 exploratory benchmark evaluated an E model trained on a **12-person** source pool, using six disjoint source-validation participants and two calibration trials. It informs architecture selection but does not directly measure the 18-person release checkpoint. See [benchmarks.md](benchmarks.md).

## Limits

The data came from healthy volunteers with 16 forearm channels and fixed gesture and repetition protocols. Overlapping calibration windows are not independent recordings. Individual performance varied. A softmax score is a model output, not established probability calibration or clinical certainty. The model and replay have not been evaluated in a clinical setting, with prosthetic devices, or as a live control loop. A separately frozen evaluation on genuinely new participants would support stronger independent-population claims.

Source data were GRABMyo v1.1.0. The official PhysioNet page specifies Creative Commons Attribution 4.0. See [NOTICE.md](../NOTICE.md) for citations and distribution boundaries.
