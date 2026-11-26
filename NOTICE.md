# Attribution and distribution notice

The pretrained source encoder was trained using data from the Gesture Recognition and Biometrics ElectroMyogram (GRABMyo) dataset. The official [GRABMyo v1.1.0 release](https://physionet.org/content/grabmyo/1.1.0/) lists the dataset under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The dataset authors request citations to the dataset and its original paper, and the PhysioNet citation listed on the release page.

- Jiang, N., Pradhan, A. and He, J. (2024). Gesture Recognition and Biometrics ElectroMyogram, version 1.1.0. PhysioNet. https://doi.org/10.13026/89dm-f662
- Pradhan, A., He, J. and Jiang, N. (2022). Multi-day dataset of forearm and wrist electromyogram for hand gesture recognition and biometrics. Scientific Data 9, 733. https://doi.org/10.1038/s41597-022-01836-y
- Pollard, T. et al. (2026). PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z

The package does not include GRABMyo participant recordings. The synthetic demonstration material is generated for this project, and is not a real participant recording. The original 18-person checkpoint was trained using the dataset, then represented in a package-safe payload containing its encoder tensors. Original and repackaged checkpoint digests should both be retained.

The repository license is a proposed MIT license for project-owned code, subject to owner confirmation before publication. The rights and intended terms for redistributing the trained checkpoint require the owner's release signoff. Dataset CC BY 4.0 attribution does not establish authorship of the project's code or settle all rights in learned weights. No external publication has been authorized.

For owner review, the proposed terms are MIT for project-owned code and original synthetic demonstration material, and CC BY 4.0 for the distributed source-encoder weights to the extent the project owner can grant those rights. The GRABMyo dataset is separately licensed under CC BY 4.0. No participant recordings are included. Final weight and code rights remain subject to owner confirmation.
