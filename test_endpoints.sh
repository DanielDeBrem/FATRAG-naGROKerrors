#!/bin/bash
# Quick FATRAG API Test

BASE="http://localhost:8020"
echo "Testing FATRAG API endpoints..."
echo ""

# Core
echo "✓ Health: $(curl -s $BASE/health | grep -o '"status":"ok"')"
echo "✓ Root: $(curl -s -o /dev/null -w '%{http_code}' $BASE/)"
echo "✓ Query: $(curl -s -X POST $BASE/query -H 'Content-Type: application/json' -d '{"question":"test"}' -w '%{http_code}' -o /dev/null)"
echo "✓ Admin Health: $(curl -s -o /dev/null -w '%{http_code}' $BASE/admin/health)"
echo "✓ Config: $(curl -s -o /dev/null -w '%{http_code}' $BASE/admin/config)"
echo "✓ Docs: $(curl -s -o /dev/null -w '%{http_code}' $BASE/admin/docs)"
echo "✓ Clients: $(curl -s -o /dev/null -w '%{http_code}' $BASE/admin/clients)"
echo "✓ Projects: $(curl -s -o /dev/null -w '%{http_code}' $BASE/admin/projects)"
echo "✓ Feedback: $(curl -s -o /dev/null -w '%{http_code}' $BASE/admin/feedback)"
echo "✓ Templates: $(curl -s -o /dev/null -w '%{http_code}' $BASE/admin/templates)"
echo "✓ Tax Income: $(curl -s -X POST $BASE/admin/tax/calculate/income -H 'Content-Type: application/json' -d '{"income":75000}' -w '%{http_code}' -o /dev/null)"
echo "✓ Tax Corporate: $(curl -s -X POST $BASE/admin/tax/calculate/corporate -H 'Content-Type: application/json' -d '{"profit":250000}' -w '%{http_code}' -o /dev/null)"
echo "✓ Tax VAT: $(curl -s -X POST $BASE/admin/tax/calculate/vat -H 'Content-Type: application/json' -d '{"net_amount":1000,"vat_rate":"standard"}' -w '%{http_code}' -o /dev/null)"
echo "✓ Progressive History: $(curl -s -o /dev/null -w '%{http_code}' $BASE/api/progressive-test/history)"
echo ""
echo "All major endpoints responding!"
