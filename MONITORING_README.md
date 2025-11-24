# 📊 FATRAG Monitoring & Optimization System

Compleet real-time monitoring en auto-testing systeem om de optimale configuratie voor elk analyse type te vinden.

## 🎯 Doel

Automatisch testen van verschillende model configuraties (temperature, tokens, modellen) en real-time monitoren van performance om **kwaliteit binnen redelijke tijd** te produceren.

## 📦 Components

### 1. **MetricsStore** (`metrics_store.py`)
- Track timing, tokens, quality scores voor alle analyses
- Aggregeer statistieken per analyse type
- Compare verschillende configuraties
- Recommend beste settings

### 2. **Auto-Test Framework** (`scripts/auto_test_analyses.py`)
- Test Flash Analysis met verschillende configs
- Test Grondige Analyse met verschillende configs  
- Test Template Reports met verschillende configs
- Genereer comparison reports

### 3. **Monitoring Dashboard** (`static/admin/monitor.html`)
- Real-time stats per analyse type
- Performance charts over tijd
- Config comparison visualisatie
- Test runner met één klik

## 🚀 Quick Start

### Stap 1: Test een analyse type

```bash
cd /home/daniel/Projects/FATRAG

# Test alleen Flash Analysis (snel)
python scripts/auto_test_analyses.py --flash-only

# Test alleen Templates
python scripts/auto_test_analyses.py --templates-only

# Run alle tests (duurt lang!)
python scripts/auto_test_analyses.py
```

### Stap 2: Bekijk Dashboard

Open in je browser:
```
http://10.0.1.227:8020/admin/monitor.html
```

Dashboard features:
- **Quick Stats**: Active jobs, avg duration, success rate
- **Tabs per analyse type**: Flash, Grondige, Templates, FATRAG
- **Performance Chart**: Duration over tijd
- **Recent Runs Table**: Laatste 5 runs met config
- **Config Comparison**: Vergelijk model, temperature, tokens
- **Test Runner**: Start nieuwe tests vanuit UI

### Stap 3: Analyseer Resultaten

```bash
# Bekijk statistieken voor een analyse type
python scripts/auto_test_analyses.py --analyze flash_analysis
python scripts/auto_test_analyses.py --analyze grondige_analysis
python scripts/auto_test_analyses.py --analyze template_holding_analysis
```

## 📝 Test Configuraties

### Flash Analysis
```python
Config 1: llama3.1:8b, temp=0.1, max_tokens=2000 (snelst, minst creative)
Config 2: llama3.1:8b, temp=0.2, max_tokens=3000 (balanced)
Config 3: llama3:8b, temp=0.15, max_tokens=2500 (alternatief model)
```

### Grondige Analyse
```python
Config 1: llama3.1:70b, temp=0.1, chunk=2000, overlap=200 (meest precies)
Config 2: llama3.1:70b, temp=0.15, chunk=3000, overlap=300 (grotere chunks)
Config 3: llama3.1:70b, temp=0.05, chunk=2500, overlap=250 (minste randomness)
```

### Template Reports
```python
Config 1: llama3.1:70b, temp=0.1, parallel=True (snel, parallel)
Config 2: llama3.1:70b, temp=0.15, parallel=True (iets meer creative)
Config 3: llama3.1:70b, temp=0.1, parallel=False (sequential, langzaam maar betrouwbaar)
```

## 📊 Metrics Opgeslagen

Voor elke test run wordt opgeslagen:

```json
{
  "run_id": "flash_analysis_1699999999",
  "job_id": "flash_test_xxx",
  "analysis_type": "flash_analysis",
  "config": {"model": "llama3.1:8b", "temperature": 0.1},
  "start_time": "2025-11-10T20:00:00",
  "end_time": "2025-11-10T20:02:30",
  "duration_seconds": 150.5,
  "stages": [
    {"name": "flash_generation", "duration_seconds": 150, "tokens_used": 2500}
  ],
  "tokens": {"total": 2500},
  "quality_scores": {},
  "status": "completed"
}
```

Metrics worden opgeslagen in: `metrics/`

## 🎯 Gebruik Cases

### Use Case 1: Vind snelste Flash config
```bash
# Run tests
python scripts/auto_test_analyses.py --flash-only

# Analyseer
python scripts/auto_test_analyses.py --analyze flash_analysis

# Output toont beste config op basis van duration
```

### Use Case 2: Vergelijk Template parallel vs sequential
Dashboard:
1. Ga naar Templates tab
2. Bekijk Recent Runs
3. Filter op `sections_parallel`
4. Zie verschil in duration

### Use Case 3: Monitor production runs
1. Dashboard auto-refresht elke 30s
2. Zie active jobs real-time
3. Check success rate
4. Alert bij drops

## 🔧 API Endpoints (TODO)

Je moet deze nog toevoegen aan `main.py`:

```python
# GET /api/metrics/stats?type=flash_analysis
# GET /api/metrics/compare?type=flash_analysis&key=model
# POST /api/run-tests {test_type: "flash"}
```

## 📈 Optimization Workflow

1. **Baseline**: Run 3 tests per config per analyse type
2. **Analyze**: Bekijk avg duration, tokens, success rate
3. **Iterate**: Pas best-performing configs aan
4. **Production**: Gebruik winning config in main.py
5. **Monitor**: Dashboard tracks production performance
6. **Repeat**: Weekly re-test met nieuwe configs

## 🎨 Custom Configs

Edit `scripts/auto_test_analyses.py`:

```python
FLASH_CONFIGS = [
    {"model": "llama3.1:8b", "temperature": 0.1, "max_tokens": 2000},
    # Add your config:
    {"model": "llama3.1:8b", "temperature": 0.3, "max_tokens": 1500},
]
```

## 📊 Expected Timing (geschat)

| Analyse Type | Config Snelst | Config Langzaamst | Verschil |
|--------------|---------------|-------------------|----------|
| Flash | 30s (8B, low temp) | 90s (8B, high temp) | 3x |
| Grondige | 5m (70B, small chunks) | 15m (70B, large chunks) | 3x |
| Template | 10m (parallel) | 30m (sequential) | 3x |
| FATRAG Full | 30m (8 GPU) | 2h (1 GPU) | 4x |

## 🚨 Monitoring Alerts (toekomstig)

- Duration > 2x avg → Alert
- Success rate < 90% → Alert
- Tokens > budget → Warning
- GPU util < 50% → Inefficiency warning

## 🛠 Next Steps

1. ✅ MetricsStore gebouwd
2. ✅ Auto-test framework gemaakt
3. ✅ Dashboard gebouwd
4. ⏳ **TODO: API endpoints toevoegen aan main.py**
5. ⏳ **TODO: Integreer metrics in bestaande analyses**
6. ⏳ **TODO: Run eerste test suite**
7. ⏳ **TODO: Optimize configuraties**

## 💡 Tips

- Start met **Flash only** tests (snelst)
- Run tests 's nachts (GPU usage)
- Test incrementeel (1 analyse type per keer)
- Monitor token usage vs quality trade-off
- Document winning configs in comments

## 📞 Support

Bij vragen over het monitoring systeem:
- Check metrics/ directory voor raw data
- Check test_results_*.json voor full test output
- Dashboard logs errors in browser console
