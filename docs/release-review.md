# Scientific release review

The original E architecture, source encoder identity and synthetic equivalence evidence received an independent read-only scientific review. The review passed for the local v0.1.0 release candidate. The review examined checkpoint provenance, original preprocessing, one complete 2,000-update synthetic personalization episode and the calibration-only final classifier.

The reviewed source checkpoint array archive has SHA-256 `1cc9a0afec84abebd0e0b9526c5d521ccbf382e98a5997722a8b6c81404477df`. The public package uses the same reviewed encoder and personalization procedure. See [release verification](release-verification.md) for measurement conditions and the scope of the evidence.

At the time of that review, Linux execution, redistribution terms and public release approval remained separate publication checks. The owner has since approved the license terms and publication, and Windows and Linux CI passed for the reviewed source commit. The source participant-disjoint and same-cohort cross-session results do not establish performance in a new participant population.
