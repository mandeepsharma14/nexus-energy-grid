"""
NexusGrid — AI Copilot API Router
LLM-powered natural language interface for energy intelligence queries.
Uses Anthropic Claude with RAG (energy context injection).

© 2026 Mandeep Sharma. All rights reserved.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

from app.core.config import settings

router = APIRouter()


# ── Request / Response Models ─────────────────────────────
class CopilotMessage(BaseModel):
    role: str        # "user" | "assistant"
    content: str

class CopilotRequest(BaseModel):
    message: str
    org_id: str = "arcelor_steel"
    facility_id: Optional[str] = None
    conversation_history: List[CopilotMessage] = []

class CopilotResponse(BaseModel):
    answer: str
    citations: List[str] = []
    data_points: dict = {}
    recommended_actions: List[str] = []
    confidence: float = 0.85


# ── Context Builder ───────────────────────────────────────
def build_energy_context(org_id: str, facility_id: Optional[str] = None) -> str:
    """
    Build structured energy context to inject into the LLM prompt.
    This is the RAG layer — relevant data is pulled and formatted
    for the model to reason over.
    """
    ctx_parts = []

    # Load bills data
    bills_path = os.path.join(settings.DATA_DIR, "utility_bills.csv")
    if os.path.exists(bills_path):
        bills = pd.read_csv(bills_path)
        bills_org = bills[bills.get("org_id", pd.Series(dtype=str)) == org_id]
        if facility_id:
            bills_org = bills_org[bills_org["facility_id"] == facility_id]

        if len(bills_org) > 0:
            recent = bills_org.tail(3)
            ctx_parts.append(f"RECENT BILLING (last 3 months):\n{recent[['facility_id','month','total_kwh','total_bill','peak_kw','blended_rate']].to_string(index=False)}")

    # Load recommendations
    recs_path = os.path.join(settings.DATA_DIR, "recommendations.csv")
    if os.path.exists(recs_path):
        recs = pd.read_csv(recs_path)
        recs_org = recs[recs.get("org_id", pd.Series(dtype=str)) == org_id]
        if len(recs_org) > 0:
            ctx_parts.append(f"AI RECOMMENDATIONS:\n{recs_org[['type','title','annual_savings_usd','confidence','effort']].head(5).to_string(index=False)}")

    # Load emissions
    em_path = os.path.join(settings.DATA_DIR, "emissions.csv")
    if os.path.exists(em_path):
        em = pd.read_csv(em_path)
        em_org = em[em.get("org_id", pd.Series(dtype=str)) == org_id]
        if len(em_org) > 0:
            ctx_parts.append(f"EMISSIONS DATA:\n{em_org.tail(3).to_string(index=False)}")

    if ctx_parts:
        return "\n\n".join(ctx_parts)
    else:
        return "Portfolio data: 48.2 GWh consumed, $3.24M spent this month across 5 facilities. 7 anomalies detected. AI savings potential: $1.06M/year."


def build_system_prompt(energy_context: str) -> str:
    return f"""You are the NexusGrid AI Energy Copilot — an expert AI assistant embedded in an enterprise energy intelligence platform.

You help energy managers, sustainability leads, and CFOs understand their energy data, reduce costs, achieve sustainability targets, and make smart procurement decisions.

CURRENT PORTFOLIO DATA:
{energy_context}

YOUR CAPABILITIES:
1. Analyze consumption trends, cost drivers, and anomalies
2. Explain AI recommendations with supporting data
3. Guide optimization strategies (tariff switching, peak shaving, load shifting)
4. Provide carbon/ESG insights and net-zero pathway advice
5. Give procurement timing recommendations based on price forecasts
6. Generate executive summaries

RESPONSE STYLE:
- Be concise, data-driven, and action-oriented
- Always quantify savings, risks, and impacts in dollar and carbon terms
- Format key numbers prominently
- End with 1-3 concrete next actions when relevant
- Be confident but acknowledge uncertainty where appropriate

You are not a general chatbot — stay focused on energy, carbon, and sustainability topics.
© 2026 Mandeep Sharma — NexusGrid AI Platform"""


# ── Main Chat Endpoint ────────────────────────────────────
@router.post("/chat", response_model=CopilotResponse, summary="AI Copilot chat")
async def copilot_chat(request: CopilotRequest):
    """
    Natural language energy intelligence interface.
    Injects portfolio context and routes to LLM.
    Falls back to rule-based responses if API unavailable.
    """
    energy_context = build_energy_context(request.org_id, request.facility_id)
    system_prompt = build_system_prompt(energy_context)

    # Build message history for multi-turn
    messages = [{"role": m.role, "content": m.content}
                for m in request.conversation_history[-6:]]  # last 3 turns
    messages.append({"role": "user", "content": request.message})

    # ── Try Anthropic API ──────────────────────────────────
    if settings.ANTHROPIC_API_KEY:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=settings.LLM_MODEL,
                max_tokens=settings.LLM_MAX_TOKENS,
                system=system_prompt,
                messages=messages,
            )
            answer = response.content[0].text
            return CopilotResponse(
                answer=answer,
                citations=["NexusGrid AI Analysis", f"Data as of {datetime.now().strftime('%Y-%m-%d')}"],
                confidence=0.92,
            )
        except Exception as e:
            # Fall through to rule-based
            pass

    # ── Rule-based fallback (demo mode) ───────────────────
    answer = _rule_based_response(request.message, energy_context)
    return CopilotResponse(
        answer=answer,
        citations=["NexusGrid Demo Mode — Connect API key for live AI"],
        data_points={"mode": "rule_based"},
        confidence=0.78,
    )


def _rule_based_response(question: str, context: str) -> str:
    """Deterministic responses for common energy queries (demo mode)."""
    q = question.lower()

    if any(w in q for w in ["cost", "increase", "expensive", "bill"]):
        return ("Your March cost of **$3.24M** is up 7.4% vs February. Primary drivers:\n\n"
                "1. **Market prices +9.2%** — PJM averaged $48.20/MWh vs $44.10\n"
                "2. **Demand spike at Plant Gamma** — triggered $3,200 demand ratchet\n"
                "3. **Production increase +12%** — added 2.1 GWh consumption\n\n"
                "**Recommendation:** Switch Plant Alpha to TOU Rate D immediately — saves **$18,400/month** with zero capex.")

    if any(w in q for w in ["reduce", "save", "15%", "optimize"]):
        return ("Achievable. Here's your AI-generated roadmap to **15%+ cost reduction (~$486K/year)**:\n\n"
                "**Action 1 — Tariff Switch (Month 1):** Switch to TOU Rate D → **$412K/yr** (zero capex)\n"
                "**Action 2 — Peak Shaving (Month 1-2):** Pre-cool HVAC 2-4 AM → **$183K/yr**\n"
                "**Action 3 — Demand Response (Month 2):** Enroll in PJM DR → **$172K/yr** incentives\n"
                "**Action 4 — Forward Contracts:** Lock 30% load before Q2 spike → **$290K** savings\n\n"
                "Combined: **$1.06M/yr (32% reduction)**. I can auto-execute Actions 1-3 with approval.")

    if any(w in q for w in ["carbon", "co2", "emission", "sustainability", "esg"]):
        return ("Your portfolio emissions are trending **favorably**:\n\n"
                "- **Scope 2:** 9,000 tCO₂/month — down **7.2% YoY** from grid decarbonization\n"
                "- **Carbon intensity:** 0.266 tCO₂/MWh — improved **6.3% YoY**\n\n"
                "At current trajectory, you reach net-zero by **2048**. To hit your SBTi 2040 target:\n"
                "1. Increase renewable to 42% by 2028 (currently at 38%)\n"
                "2. Activate carbon-aware scheduling (AI auto-shifts loads to low-carbon windows)\n"
                "3. Purchase additional RECs for HQ facility (8% gap to 100%)")

    if any(w in q for w in ["facility", "worst", "inefficient", "poor"]):
        return ("**Plant Gamma (Detroit, MI)** is your least efficient facility — AI score **41/100**:\n\n"
                "- EUI: **195 kBtu/sqft** vs industry median 145\n"
                "- 4 critical anomalies in 30 days\n"
                "- 3 billing errors = **$14,200 overbilling** (dispute immediately)\n"
                "- Power factor 0.82 → monthly penalty **$1,400**\n\n"
                "Monthly overrun: **$96,400** vs benchmark peers.\n\n"
                "**Top actions:** Fix capacitor bank + dispute bills + HVAC retrofit = **$37K/month** savings.")

    if any(w in q for w in ["contract", "procurement", "forward", "price"]):
        return ("**YES — act within 48 hours.**\n\n"
                "LSTM model forecasts **+14% price increase Q2** based on:\n"
                "- Natural gas futures up 18%\n"
                "- PJM tight summer capacity margins\n"
                "- Above-normal temperature forecast\n\n"
                "**Strategy:** Lock 30% of baseload (25 MW) at current $41.20/MWh\n"
                "Value: 25 MW × 8,760h × 50% LF × $5.80/MWh delta = **$289,620 saved**\n"
                "Confidence: **81%**. Escalate to procurement team today.")

    return ("I've analyzed your portfolio data:\n\n"
            "**Portfolio KPIs (MTD):**\n"
            "- Consumption: 48.2 GWh\n"
            "- Total Cost: $3.24M\n"
            "- AI Savings Realized: $412K\n"
            "- Anomalies Detected: 7\n\n"
            "**Top opportunity:** $1.06M/year in AI-identified savings across tariff optimization, "
            "peak shaving, and demand response. Ask me about any specific area for a deep-dive analysis.")


# ── Generate Executive Summary ────────────────────────────
@router.post("/executive-summary", summary="Auto-generate executive summary")
async def generate_summary(org_id: str = "arcelor_steel"):
    """Autonomously generate a board-ready executive energy summary."""
    context = build_energy_context(org_id)
    prompt = (f"Generate a concise executive summary (200 words) for the energy portfolio. "
              f"Focus on: total spend, key wins, risks, and top 3 recommended actions.\n\n{context}")

    if settings.ANTHROPIC_API_KEY:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            resp = client.messages.create(
                model=settings.LLM_MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return {"summary": resp.content[0].text, "generated_at": datetime.now().isoformat()}
        except Exception:
            pass

    return {
        "summary": (
            "Energy spend is 8.1% below budget driven by AI-optimized tariff selection saving $412K. "
            "Three anomalies were auto-detected and remediated. Carbon intensity improved 6.3% YoY. "
            "Plant Gamma requires immediate attention — inefficiency score 41/100 represents $96K/month overrun. "
            "Q2 procurement recommendation: lock 30% of load at current rates before forecast +14% price spike. "
            "Forecast accuracy: 94.2% on 30-day horizon."
        ),
        "generated_at": datetime.now().isoformat(),
        "mode": "rule_based",
    }
