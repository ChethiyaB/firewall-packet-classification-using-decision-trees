# Firewall Packet Classifier

Production-ready, replicable pipeline for classifying firewall actions (Allow/Deny/Drop/Reset-both).

## Quickstart
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export FIREWALL_DATA_URL="https://archive.ics.uci.edu/static/public/542/internet+firewall+data.zip"
python -m firewall_ml.train --cfg_path=configs/default.yaml

python -m firewall_ml.predict --model_path .artifacts/adaboost.joblib --input_csv samples/infer.csv --output_csv predictions.csv

```bash
git add .
git commit -m "feat: initial production-ready pipeline"
git push -u origin main
gh repo edit --enable-issues --enable-projects
gh api -X PUT repos/youruser/firewall-packet-classifier/branches/main/protection \
  -F required_status_checks='{"strict":true,"contexts":[]}' \
  -F enforce_admins=true \
  -F required_pull_request_reviews='{"required_approving_review_count":1}'
gh release create v0.1.0 --notes "First production-ready release"
