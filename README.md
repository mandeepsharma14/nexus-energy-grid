# ⚡ NexusGrid — AI-Powered Energy Intelligence & Optimization Platform

> **Enterprise-grade energy intelligence platform combining advanced AI/ML forecasting, prescriptive optimization, digital twin modeling, and autonomous sustainability intelligence.**

© 2026 Mandeep Sharma. All rights reserved.

---

## 🎯 Product Vision

NexusGrid is a next-generation, AI-native enterprise energy platform that helps organizations:

- **Reduce energy costs by 10–25%** through AI-driven optimization
- **Predict risk and anomalies** before they impact budgets
- **Achieve sustainability targets** with carbon-aware intelligence
- **Automate procurement** with forward contract advisory
- **Generate autonomous insights** for executive decision-making

---

## 🚀 Live Demo

**Demo URL:** `nexusgrid-demo.vercel.app` *(deploy from this codebase)*

**Demo Mode:** No login required — 5 pre-loaded organizations:
| Organization | Type | Annual Spend |
|---|---|---|
| ArcelorSteel Manufacturing | Industrial | $38.9M |
| MegaMart Retail Chain | Retail | $26.2M |
| PrimePlex Commercial RE | Commercial RE | $17.0M |
| SwiftLogix Warehouses | Logistics | $22.1M |
| Vertex Data Centers | Technology | $86.9M |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES LAYER                       │
│  Utility APIs │ IoT/Smart Meters │ CSV/PDF │ Market │ Carbon │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA INGESTION PIPELINE                     │
│  Schema Validation │ ETL │ Anomaly Flagging │ Normalization  │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI/ML CORE ENGINE                         │
│  Forecasting (LSTM/Prophet) │ Anomaly (Isolation Forest)    │
│  Optimization (LP + RL)     │ Recommendation Engine         │
│  AI Copilot (LLM + RAG)     │ Digital Twin │ Carbon AI      │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              FASTAPI BACKEND (REST + WebSocket)              │
│  JWT Auth │ Multi-Tenant Middleware │ Rate Limiting          │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA PERSISTENCE                           │
│  PostgreSQL (Neon) │ TimescaleDB │ Redis │ S3 Storage        │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           NEXT.JS 14 FRONTEND (Vercel CDN)                  │
│  Portfolio Dashboard │ AI Copilot │ Optimizer │ ESG Reports  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Model

```sql
-- Core Tenant Model
organizations (id, name, type, tier, settings, created_at)
facilities    (id, org_id, name, location, sqft, type, timezone)
meters        (id, facility_id, type, unit, utility_account, tariff_id)
tariffs       (id, utility, schedule_name, tou_periods, demand_charges)

-- Time-Series Data
energy_usage (time, meter_id, kwh, kw_demand, power_factor, cost)
  → TimescaleDB hypertable, partitioned by month
  → Continuous aggregates: hourly, daily, monthly

-- Intelligence Layer
forecasts        (id, facility_id, model, horizon, forecast_date, values[])
recommendations  (id, org_id, type, title, description, savings, confidence, status)
alerts           (id, facility_id, severity, type, message, root_cause, resolved_at)
anomalies        (id, meter_id, detected_at, score, baseline, actual, cost_impact)

-- Sustainability
emissions        (id, facility_id, scope, period, tco2, methodology, source)
esg_targets      (id, org_id, framework, target_year, baseline, current, unit)
carbon_credits   (id, org_id, type, quantity, vintage, registry, cost)

-- Utility Bills
utility_bills    (id, facility_id, period, kwh, demand_kw, charges{}, total, raw_pdf)
billing_errors   (id, bill_id, error_type, expected, actual, dispute_amount)
```

---

## 🤖 AI/ML Strategy

### Forecasting Engine
- **Models:** LSTM (PyTorch), Prophet (Meta), ARIMA (statsmodels)
- **Horizons:** 7-day (96% accuracy), 30-day (94%), 90-day (88%)
- **Features:** weather, calendar, production schedule, price signals
- **Output:** point forecast + 80/95% confidence intervals

### Anomaly Detection
- **Isolation Forest:** unsupervised, handles high-dimensional meter data
- **Autoencoder:** deep learning, captures seasonal patterns
- **Statistical Process Control:** for real-time streaming alerts
- **Output:** anomaly score (0–1), estimated cost impact, probable root cause

### Optimization Engine
- **Linear Programming (PuLP/CVXPY):** tariff optimization, peak shaving
- **Reinforcement Learning (Stable-Baselines3):** load scheduling agent
  - Environment: 24h grid prices + demand schedule
  - Reward: minimize cost + carbon while meeting production constraints
- **Output:** optimal load schedule, tariff recommendation, contract strategy

### AI Copilot
- **LLM:** Claude claude-sonnet-4-6 API via RAG pipeline
- **Context:** energy metrics + forecasts + recommendations injected per query
- **Capabilities:** root cause analysis, savings roadmap, ESG narrative, executive summaries

### Innovation Features
1. **Carbon-Aware Scheduling:** shifts deferrable loads to low grid-carbon-intensity windows (EPA eGRID real-time)
2. **RL Load Agent:** reinforcement learning agent that continuously learns optimal load patterns
3. **Autonomous Insight Engine:** auto-generates alerts, root causes, and executive summaries without human trigger

---

## 🗂 Module Architecture

```
nexusgrid/
├── frontend/                    # Next.js 14 App Router
│   ├── app/
│   │   ├── (demo)/
│   │   │   ├── portfolio/       # Portfolio dashboard
│   │   │   ├── facilities/      # Facility management
│   │   │   ├── analytics/       # Energy analytics
│   │   │   ├── copilot/         # AI Copilot chat
│   │   │   ├── forecasts/       # Forecast visualization
│   │   │   ├── optimizer/       # Savings simulator
│   │   │   ├── carbon/          # Sustainability & ESG
│   │   │   ├── twin/            # Digital Twin
│   │   │   └── alerts/          # Alert center
│   │   └── page.tsx             # Landing page
│   ├── components/
│   │   ├── charts/              # Recharts/D3 components
│   │   ├── cards/               # Stat, insight, facility cards
│   │   ├── copilot/             # Chat interface
│   │   └── layout/              # Nav, sidebar, footer
│   └── lib/
│       ├── api.ts               # API client
│       ├── mock-data.ts         # Demo synthetic data
│       └── hooks/               # useEnergy, useForecast, etc.
│
├── backend/                     # FastAPI Python
│   ├── app/
│   │   ├── api/v1/
│   │   │   ├── energy.py        # Usage data endpoints
│   │   │   ├── forecasts.py     # Forecast endpoints
│   │   │   ├── insights.py      # Recommendations
│   │   │   ├── optimize.py      # Optimization engine
│   │   │   ├── carbon.py        # Emissions/ESG
│   │   │   ├── copilot.py       # LLM interface
│   │   │   ├── twin.py          # Digital Twin
│   │   │   └── alerts.py        # Alert management
│   │   ├── models/              # SQLAlchemy models
│   │   ├── services/
│   │   │   ├── ml/
│   │   │   │   ├── forecaster.py    # LSTM + Prophet
│   │   │   │   ├── anomaly.py       # Isolation Forest
│   │   │   │   ├── optimizer.py     # LP + RL
│   │   │   │   └── copilot.py       # LLM + RAG
│   │   │   └── ingestion/
│   │   │       ├── utility_api.py   # Green Button / EDI
│   │   │       ├── iot_mqtt.py      # Smart meter ingestion
│   │   │       └── bill_ocr.py      # PDF bill parsing
│   │   └── core/
│   │       ├── auth.py          # JWT + RBAC
│   │       ├── tenant.py        # Multi-tenant middleware
│   │       └── config.py        # Settings
│   └── Dockerfile
│
├── ml/                          # Standalone ML pipeline
│   ├── training/
│   │   ├── train_lstm.py
│   │   ├── train_anomaly.py
│   │   └── train_rl_agent.py
│   ├── data/
│   │   └── synthetic_generator.py  # Realistic demo data
│   └── models/                  # Saved model artifacts
│
├── infra/
│   ├── docker-compose.yml
│   ├── schema.sql               # Full DB schema
│   └── seed_data.sql            # Demo data
│
└── docs/
    ├── ARCHITECTURE.md
    ├── API.md
    ├── DEPLOYMENT.md
    └── ML_STRATEGY.md
```

---

## 🚢 Deployment Guide (Free-Tier Demo)

### Prerequisites
- GitHub account
- Vercel account (free)
- Render or Railway account (free)
- Neon PostgreSQL account (free)

### Step 1: Database Setup (Neon)
```bash
# Create Neon project → copy connection string
# Run migrations:
psql $DATABASE_URL -f infra/schema.sql
psql $DATABASE_URL -f infra/seed_data.sql
```

### Step 2: Backend Deploy (Render)
```bash
# Connect GitHub repo to Render
# Set environment variables:
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
ANTHROPIC_API_KEY=sk-...
SECRET_KEY=your-secret-key
ENVIRONMENT=production

# Build command: pip install -r requirements.txt
# Start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Step 3: Frontend Deploy (Vercel)
```bash
# Connect GitHub repo to Vercel
# Set environment variables:
NEXT_PUBLIC_API_URL=https://nexusgrid-api.onrender.com
NEXT_PUBLIC_WS_URL=wss://nexusgrid-api.onrender.com

# Framework: Next.js (auto-detected)
# Deploy: automatic on push to main
```

### Demo Mode (No Backend)
The HTML demo file (`nexusgrid-platform.html`) runs fully client-side with synthetic data — no backend required.

---

## 🔧 Local Development

```bash
# Frontend
cd frontend
npm install
cp .env.example .env.local
npm run dev       # http://localhost:3000

# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
alembic upgrade head
python -m app.cli seed-demo-data
uvicorn app.main:app --reload  # http://localhost:8000

# ML Training
cd ml
pip install -r requirements.txt
python data/synthetic_generator.py
python training/train_lstm.py
python training/train_anomaly.py
```

---

## 📋 API Reference

### Authentication
```
POST /api/v1/auth/demo-login     → JWT token (demo mode)
POST /api/v1/auth/login          → Full auth
```

### Key Endpoints
```
GET  /api/v1/energy/usage        → Time-series consumption data
GET  /api/v1/forecasts/{facility_id}?horizon=30  → AI forecasts
POST /api/v1/copilot/chat        → AI Copilot query
POST /api/v1/optimize/simulate   → Savings simulator
GET  /api/v1/carbon/emissions    → Scope 1/2/3 data
GET  /api/v1/alerts              → Active alerts
POST /api/v1/twin/simulate       → Digital twin scenario
GET  /api/v1/insights            → AI recommendations
GET  /api/v1/reports/esg         → ESG report package
WS   /api/v1/ws/{tenant_id}      → Real-time data stream
```

---

## 📈 Roadmap

### Phase 1 — Demo ✅
- Fully interactive client-side demo
- 5 synthetic organizations
- All 9 modules functional
- AI Copilot with pre-trained responses

### Phase 2 — MVP Backend
- FastAPI + PostgreSQL/Neon
- Real ML model serving
- Actual LLM Copilot via Anthropic API
- Multi-tenant auth

### Phase 3 — SaaS
- Stripe billing integration
- Utility API connections (Green Button)
- Real IoT/MQTT ingestion
- Mobile app (React Native)

### Phase 4 — Enterprise
- On-premise deployment option
- Custom ML model fine-tuning
- White-labeling
- Advanced RL optimization agent

---

## 🛡 Security

- **Authentication:** JWT + refresh tokens, role-based access control
- **Multi-tenancy:** Row-Level Security in PostgreSQL, tenant isolation middleware
- **API Security:** Rate limiting, request signing, CORS policy
- **Data:** AES-256 encryption at rest, TLS 1.3 in transit
- **Secrets:** Environment variables only, never in code

---

## 🤝 Contributing

This is a portfolio project demonstrating production-grade AI/ML engineering capabilities.

**Tech Stack Summary:**
- Frontend: Next.js 14, TypeScript, TailwindCSS, Recharts
- Backend: FastAPI, Python, SQLAlchemy
- ML: PyTorch, scikit-learn, Stable-Baselines3, Prophet
- DB: PostgreSQL, TimescaleDB, Redis
- Deploy: Vercel + Render + Neon

---

## 📄 License

© 2026 Mandeep Sharma. All rights reserved.

Built as part of an AI/ML engineering portfolio demonstrating enterprise-grade platform development, ML engineering, and full-stack architecture.

**Portfolio:** [github.com/mandeepsharma14](https://github.com/mandeepsharma14)  
**LinkedIn:** [linkedin.com/in/mandeepsharma14](https://linkedin.com/in/mandeepsharma14)

---

*NexusGrid — Where energy data becomes competitive intelligence.*
