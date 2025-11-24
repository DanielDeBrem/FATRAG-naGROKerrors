# FATRAG - Want to Have Features

Dit document bevat toekomstige features voor de FATRAG applicatie die later geïmplementeerd kunnen worden.

## 🚀 Geplande Features (Fase 3-4)

### 7. Automated Alert System 🚨

**Beschrijving:**
Notification systeem voor financiële professionals met real-time alerts.

**Functionaliteit:**
- Nieuwe wetgeving impact analyse (RSS feed Belastingdienst)
- Deadline tracking (aangiftes, jaarstukken, rapportages)
- Afwijkingen in analyses detectie (unusually high/low values)
- Document expiry notifications (huurcontracten, licenties)
- Regulatory changes monitoring (NL fiscale regelgeving)

**Implementatie:**
```python
# Cron jobs voor:
# - Daily: Check aangiftdeadlines
# - Weekly: Scan Belastingdienst updates
# - Monthly: Contract expiry checks
# 
# Notifications via:
# - Email (SMTP)
# - Webhook (Slack/Teams integration)
# - In-app notifications (bell icon)
```

**Prioriteit:** Medium
**Geschatte tijd:** 3-4 dagen
**Dependencies:** 
- Celery/APScheduler voor background jobs
- Email server configuratie
- RSS parser voor Belastingdienst feed

---

### 8. Client Portal 👥

**Beschrijving:**
Separate frontend voor eindklanten met beperkte toegang.

**Functionaliteit:**
- View-only access tot eigen projecten
- Download analyses & organograms (PDF/Excel export)
- Upload nieuwe documenten (drag & drop)
- Q&A chat gefilterd op eigen client data
- Digital signature workflow voor akkoord
- Document versiegeschiedenis
- Feedback/comments op rapporten

**Implementatie:**
```python
# Multi-tenant architecture:
# - JWT authentication per client
# - Row-level security in database
# - Separate routes: /client/*
# - Client-specific vectorstore filtering
# 
# Frontend:
# - Vue.js of React SPA
# - Mobile-responsive design
# - Progressive Web App (offline capability)
```

**Prioriteit:** High (voor scaling)
**Geschatte tijd:** 2-3 weken
**Dependencies:**
- JWT auth system
- Email verification
- File upload restrictions
- Rate limiting per client

---

### 9. Excel/PowerPoint Export 📊

**Beschrijving:**
Professional export formats voor presentaties en analyses.

**Functionaliteit:**

**Excel Export:**
- Evidence tables met formulas & pivots
- Financial metrics dashboard
- Multi-sheet workbook (Summary, Details, Evidence, Entities)
- Conditional formatting voor risks
- Charts & graphs embedded
- Auto-calculation formulas (ROI, NPV, etc.)

**PowerPoint Export:**
- Auto-generated slide deck:
  1. Cover slide (branding)
  2. Executive summary (bullets)
  3. Key findings (visuals)
  4. Organogram (embedded image)
  5. Financial metrics (charts)
  6. Recommendations (action items)
  7. Appendix (details)
- Corporate template support
- Custom styling per client

**Word Export:**
- Formatted report met TOC
- Headers/footers met paginanummering
- Professional styling
- Embedded images & tables
- Track changes support

**Implementatie:**
```python
# Libraries:
# - openpyxl: Excel generation
# - python-pptx: PowerPoint generation  
# - python-docx: Word generation
# 
# Endpoints:
# POST /admin/projects/{id}/export/excel
# POST /admin/projects/{id}/export/powerpoint
# POST /admin/projects/{id}/export/word
# 
# Template engine:
# - Jinja2 for content interpolation
# - Custom styling per client
```

**Prioriteit:** Medium-High
**Geschatte tijd:** 1-2 weken
**Dependencies:**
- openpyxl, python-pptx, python-docx
- Template storage (filesystem or S3)
- Large file handling

---

### 10. Collaborative Annotations 💬

**Beschrijving:**
Team features voor samenwerkende adviseurs.

**Functionaliteit:**
- Inline comments op document chunks/sections
- @mention other advisors (notifications)
- Task assignment (review this section, verify calculation)
- Approval workflow (partner sign-off required)
- Discussion threads per finding
- Comment history & audit trail
- Resolve/unresolve status
- Priority tagging (urgent, review, question)

**Implementatie:**
```python
# Database schema:
# - comments table (content, user_id, doc_id, chunk_id)
# - mentions table (comment_id, user_id)
# - tasks table (assignee, status, due_date)
# - approvals table (project_id, approver, status)
# 
# Real-time updates:
# - WebSocket connections (socket.io)
# - Push notifications
# 
# Frontend:
# - Comment sidebar (per document)
# - Inline highlighting
# - Notification bell icon
# - Task dashboard
```

**Prioriteit:** Medium (for teams)
**Geschatte tijd:** 2-3 weken
**Dependencies:**
- WebSocket server (Socket.IO or FastAPI WebSocket)
- User management system
- Notification system
- Real-time sync

---

## 📋 Implementatie Volgorde (Suggesties)

**Fase 3 (Enhanced Collaboration):**
1. Document Comparison (Suggestie 5) ← AL GEÏMPLEMENTEERD
2. Automated Alerts (Suggestie 7)
3. Collaborative Annotations (Suggestie 10)

**Fase 4 (Scale & Export):**
1. Excel/PowerPoint Export (Suggestie 9)
2. Client Portal (Suggestie 8)

---

## 🎯 Prioriteit Matrix

| Feature | Business Value | Complexity | Priority |
|---------|---------------|-----------|----------|
| Client Portal | Hoog | Hoog | P1 |
| Excel/PPT Export | Hoog | Medium | P1 |
| Automated Alerts | Medium | Medium | P2 |
| Collaborative Annotations | Medium | Hoog | P2 |

---

## 💡 Extra Nice-to-Have's

### A. API Rate Limiting & Quotas
- Per-client API limits
- Usage tracking
- Billing integration

### B. Advanced Search
- Full-text search across all documents
- Semantic search (vector similarity)
- Filters by date, amount, entity, type

### C. Audit Logging
- Complete audit trail (who did what when)
- Export audit logs
- Compliance reporting

### D. Machine Learning Enhancements
- Auto-categorization van documenten
- Anomaly detection in financial data
- Predictive analytics (cash flow forecasting)

### E. Integration Marketplace
- Exact Online (accounting software)
- Belastingdienst API (direct filing)
- KVK API (company data)
- RDW API (vehicle data for lease/fleet)

---

**Laatste update:** 2025-11-10
**Versie:** 1.0
**Maintainer:** FATRAG Development Team
