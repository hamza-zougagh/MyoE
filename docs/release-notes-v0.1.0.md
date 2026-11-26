# MyoE v0.1.0 release notes

Initial public release of the original E research prototype.

MyoE packages the original source-pretrained E encoder and its calibration-based personalization workflow for 16-channel, 16-gesture sEMG recordings. It includes a recorded-signal interactive demonstration and Python APIs for preprocessing, personalization, prediction and model export.

The source model requires labeled target calibration and has not been evaluated as a clinical device or prosthetic controller. The example is synthetic. Reported historical outcomes concern the specific GRABMyo cohorts and protocols described in the model card. See the model card for evaluation scope and the attribution notice for data and model provenance.

## Licensing

Project-owned code and synthetic demonstration material are licensed under MIT. The project owner approved CC BY 4.0 for the source-encoder weights. GRABMyo is independently licensed under CC BY 4.0. See [NOTICE.md](../NOTICE.md) and the official GRABMyo release for attribution. Earlier references to pending owner approval in retained checkpoint metadata are historical.
