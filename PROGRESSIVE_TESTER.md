# 🧪 Progressive Quality Tester

**De gemakkelijkste manier om optimale configuraties te vinden!**

## 🎯 Wat doet het?

Start met de **kleinste, snelste** configuratie en schaalt automatisch op naar betere kwaliteit:

```
Level 1: Minimal  ⚡ → Snel (30s-5m)
  ↓ (bij succes)
Level 2: Balanced ⚙️ → Gemiddeld (60s-10m)
  ↓ (bij succes)
Level 3: Maximum  🚀 → Beste kwaliteit (90s-30m)
```

**Stopt automatisch** bij falen, zodat je niet onnodig tijd verliest!

## 📍 Open de Tester

```
http://10.0.1.227:8020/admin/progressive-test.html
```

## 🎮 Hoe te gebruiken

1. **Kies analyse type:**
   - ⚡ Flash Analysis (3 levels: 30s → 60s → 90s)
   - 📚 Grondige Analyse (3 levels: 5m → 10m → 15m)
   - 📋 Template Report (3 levels: 10m → 20m → 30m)

2. **Klik "Start Progressive Test"**
   - Level 1 draait automatisch
   - Bij succes → Level 2 draait automatisch
   - Bij succes → Level 3 draait automatisch
   - Bij falen → Stopt en toont resultaten

3. **Bekijk resultaten:**
   - Tijd per level
   - Tokens verbruik
   - Quality score
   - **Aanbevolen configuratie** voor productie

4. **Klik "Bekijk in Dashboard"** voor historische data

## 🎨 Features

✅ **Auto-progression** - Geen manual clicks tussen levels
✅ **Smart stopping** - Stopt bij eerste fail
✅ **Visual feedback** - Spinners, progress bar, status icons
✅ **Real-time results** - Zie metrics direct per level
✅ **Recommendation engine** - Krijg beste config aanbevolen
✅ **Stop button** - Cancel test mid-run
✅ **Reset** - Start opnieuw met één klik

## 📊 Test Configs per Type

### Flash Analysis (llama3.1:8b)
```javascript
Level 1: Mini      → temp=0.1, tokens=1500  (snelst)
Level 2: Standard  → temp=0.15, tokens=2500 (balanced)
Level 3: Max       → temp=0.2, tokens=3000  (beste)
```

###  Grondige Analyse (llama3.1:70b)
```javascript
Level 1: Snel      → temp=0.05, chunks=1500 (snelst)
Level 2: Balanced  → temp=0.1, chunks=2500  (balanced)
Level 3: Diep      → temp=0.15, chunks=3500 (beste)
```

### Template Report (llama3.1:70b)
```javascript
Level 1: Basis    → temp=0.05, sequential   (snelst)
Level 2: Plus     → temp=0.1, parallel      (balanced)
Level 3: Premium  → temp=0.15, parallel     (beste)
```

## 🎯 Typische Workflow

1. **Flash test eerst** (snel, 2-6 minuten)
   - Verifieer systeem werkt
   - Vind baseline

2. **Grondige als Flash succesvol** (15-45 minuten)
   - Test 70B model
   - Vind optimale chunk size

3. **Template voor finale tuning** (30-90 minuten)
   - Test parallel vs sequential
   - Vind optimale temperature

## 💡 Wat de test output betekent

**Bij Level 1 fail:**
- Ollama problemen
- Model niet beschikbaar
- Configuration errors
→ Fix infrastructuur eerst

**Bij Level 2 fail:**
- Model overload
- Timeout issues
- Memory problemen
→ Level 1 is je productie config

**Bij Level 3 fail:**
- Edge case van model
- Extreme parameters
→ Level 2 is je productie config

**Alles succesvol:**
- 🎉 Level 3 is je productie config!
- Beste kwaliteit haalbaar

## 🚀 Na testen

1. Noteer aanbevolen config
2. Update je productie code met die settings
3. Monitor performance in Dashboard
4. Herhaal test weekly voor optimization

## 🔧 Mode: Demo vs Production

**Nu: DEMO MODE**
- Simulated tests (3-5s per level)
- Random success/fail
- Fake metrics

**TODO: Production Mode**
- Daadwerkelijke analyse runs
- Echte Ollama calls
- Real metrics naar MetricsStore

Vervang in JavaScript:
```javascript
// Replace deze simulate code:
await sleep(3000);
const success = Math.random() > 0.1;

// Met dit:
const result = await fetch('/api/progressive-test', {
  method: 'POST',
  body: JSON.stringify({type, level, config})
});
```

## 📂 Files

```
static/admin/progressive-test.html  → Main UI
metrics_store.py                    → Metrics opslag
scripts/auto_test_analyses.py      → CLI versie
```

## 🎨 Design Features

- Gradient background
- Animated spinners
- Smooth progress bar
- Color-coded feedback:
  - 🟡 Yellow = Waiting
  - 🔵 Blue = Running
  - 🟢 Green = Success
  - 🔴 Red = Failed
- Responsive mobile-first
- Tailwind CSS styling

## 📞 Quick Links

- Progressive Tester: `/admin/progressive-test.html`
- Monitoring Dashboard: `/admin/monitor.html`
- Main Admin: `/admin/index.html`
- Docs: `MONITORING_README.md`

---

**TIP:** Begin altijd met Flash Analysis om je systeem te verifiëren! 🚀

## 🔧 Parameter Tuning (Multi‑GPU, 8× RTX 3060 Ti)

Gebruik de tuner om automatisch de beste parameters te vinden en op te slaan voor de FATRAG pipeline met 1 GPU per trial (kleine modellen) en GPU‑bewuste instellingen.

- Endpoint: `POST /api/progressive-test/tune`
- Vereisten:
  1) Start 8 Ollama workers (1 GPU per port 11434..11441):
     `bash scripts/start_ollama_workers.sh`
  2) Zorg dat de FastAPI server draait op poort 8020
- Voorbeeld body:
```json
{
  "project_id": "project-XXXX",
  "search_space": {
    "model": ["llama3.1:8b"],
    "temperature": [0.1, 0.15, 0.2],
    "max_tokens": [1536, 2048, 3072],
    "max_chunks": [15, 25, 35],
    "chunk_size": [600, 800, 1000, 1200],
    "chunk_overlap": [25, 50, 100, 200],
    "concurrency": [1, 2]
  },
  "objective": "maximize_chunks_per_second",
  "budget": { "max_trials": 8, "max_total_runtime_seconds": 1800, "early_stopping_rounds": 3 },
  "persist": true
}
```
- Smoke test:
  `python3 scripts/smoke_tuner.py --project-id project-XXXX --persist`
- Resultaat:
  - Beste configuratie + score + volledige trial‑historie
  - Bij `"persist": true` worden de winnende instellingen onder `FLASH_TUNING` in `config/config.json` opgeslagen en runtime herladen

GPU‑best practices (3060 Ti, 8 GB VRAM):
- Kleine modellen (7B/8B) voor snelle trials; `concurrency` op 1–2 houden
- `max_tokens` 1536–3072, `chunk_size` 600–1200, `chunk_overlap` 25–200
- 70B alleen voor finale synthese indien gewenst; zonder NVLink is single‑GPU 70B traag — paralleliseer overige taken over de andere 7 GPU’s
