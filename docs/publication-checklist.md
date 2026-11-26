# Publication checklist

Before publishing a repository or distribution:

- Review the completed [independent scientific-equivalence audit](release-review.md) and exact checkpoint tensor and byte provenance.
- Confirm repository ownership and the proposed MIT license, checkpoint redistribution terms, dataset attribution, and permissions for included examples and screenshots.
- Keep research recordings and participant identifiers out of the release. Verify no raw or protected derived dataset artifacts are staged.
- Run clean installation and tests on Windows and Linux, confirm CPU-only dependencies, and verify the demo and packaged checkpoint in a clean environment.
- Retain measured package-specific benchmark conditions and label historical E-12 measurements separately from the E-18 checkpoint.
- Describe source-disjoint and cross-session evidence precisely. An independent new-person study is necessary for claims beyond the previously studied population and conditions.
- Verify registry name availability immediately before publishing. On 2026-11-25, the official `https://pypi.org/pypi/myoe/json` endpoint returned HTTP 404. This does not reserve a name or establish trademark rights.
- Complete screenshots and verify each depicts the released application and synthetic example.
- Obtain the owner's explicit approval for the exact repository, weights, release notes and publication target before publishing.


## Prepared GitHub publication

Suggested repository name is `MyoE`. Description is "Lightweight, CPU-efficient sEMG gesture recognition and cross-user personalization with minimal labeled calibration." Suggested topics are `emg`, `semg`, `gesture-recognition`, `biomedical-engineering`, `pytorch`, `personalization`, `machine-learning`, and `streamlit`.

The commands below create a public repository and release. Run them only after the owner approves the exact commit, code license and weight terms. GitHub CLI must be installed and authenticated. The commit author email will be public.

```powershell
winget install --id GitHub.cli --exact
gh auth login
gh auth status
Set-Location -LiteralPath 'D:\MyoE'
$myoeApprovedCommit = Read-Host 'Enter the reviewed and approved full commit SHA'
if ((git rev-parse HEAD) -ne $myoeApprovedCommit) { throw 'HEAD differs from the approved commit.' }
if (git status --porcelain) { throw 'Working tree has uncommitted changes.' }
gh repo create MyoE --public --source=. --remote=origin --push --description "Lightweight, CPU-efficient sEMG gesture recognition and cross-user personalization with minimal labeled calibration."
if ($LASTEXITCODE -ne 0) { throw 'Repository creation or push failed.' }
gh repo edit --add-topic emg --add-topic semg --add-topic gesture-recognition --add-topic biomedical-engineering --add-topic pytorch --add-topic personalization --add-topic machine-learning --add-topic streamlit
if ($LASTEXITCODE -ne 0) { throw 'Setting repository topics failed.' }
```

Allow GitHub Actions to start. Run the following block after the workflow appears. If no run is found, wait and run this block again. A failed CI run stops publication. The hash checks verify the two proposed release assets.

```powershell
Set-Location -LiteralPath 'D:\MyoE'
if ((git rev-parse HEAD) -ne $myoeApprovedCommit) { throw 'HEAD differs from the approved commit.' }
$myoeRun = gh run list --workflow ci.yml --commit $myoeApprovedCommit --json databaseId,headSha --jq '.[0].databaseId'
if ($LASTEXITCODE -ne 0 -or -not $myoeRun) { throw 'No CI run for the approved commit is available yet.' }
gh run watch $myoeRun --exit-status
if ($LASTEXITCODE -ne 0) { throw 'CI failed. Stop before tagging or releasing.' }
if ((Get-FileHash -Algorithm SHA256 dist/myoe-0.1.0-py3-none-any.whl).Hash -ne '2C17B6C0A368C37F17F57652442039C709BF18BC792278E41719D82D3A78CC52') { throw 'Wheel hash mismatch.' }
if ((Get-FileHash -Algorithm SHA256 myoe/data/source-2101.npz).Hash -ne '1CC9A0AFEC84ABEBD0E0B9526C5D521CCBF382E98A5997722A8B6C81404477DF') { throw 'Checkpoint hash mismatch.' }
git tag v0.1.0 $myoeApprovedCommit
if ($LASTEXITCODE -ne 0) { throw 'Tag creation failed.' }
git push origin v0.1.0
if ($LASTEXITCODE -ne 0) { throw 'Tag push failed.' }
gh release create v0.1.0 dist/myoe-0.1.0-py3-none-any.whl dist/SHA256SUMS --verify-tag --title "MyoE v0.1.0" --notes-file docs/release-notes-v0.1.0.md
if ($LASTEXITCODE -ne 0) { throw 'Release creation failed.' }
```

GitHub Actions tests Windows and Linux after the authorized push. Package-registry publication is separate and requires a fresh name check.
