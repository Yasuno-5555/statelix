
from typing import Dict, Any, List, Optional
import re

class AnalystEngine:
    """
    Grounded Dialogue Engine for diagnostic explanations.
    Uses DiagnosticReport and Storyteller context to answer queries.
    """
    
    def __init__(self, report: Any, storyteller_summary: str):
        self.report = report
        self.summary = storyteller_summary
        
    def answer(self, query: str) -> str:
        q = query.lower()
        
        # 1. Why rejected? / Reliability
        if any(w in q for w in ["why", "reject", "reliable", "trust", "mci"]):
            return self._explain_mci()
            
        # 2. How to fix? / Suggestions
        if any(w in q for w in ["how", "fix", "improve", "better", "suggestion"]):
            return self._explain_suggestions()
            
        # 3. Assumptions / Risks
        if any(w in q for w in ["assumption", "risk", "bias", "confound", "valid"]):
            return self._explain_assumptions()

        # 4. Success / Facts
        if any(w in q for w in ["fact", "result", "coef", "what happened", "success"]):
            # Extract facts from storyteller summary (usually section 1)
            facts = self.summary.split("### 2.")[0]
            return f"Here are the core facts from the analysis:\n{facts}"

        return "I can explain the MCI score, internal objections, model assumptions, or improvement suggestions. What would you like to know?"

    def _explain_mci(self) -> str:
        mci = self.report.mci
        header = f"The model has a Credibility Score (MCI) of {mci.total_score:.2f}."
        
        if mci.total_score > 0.8:
            status = "It is considered Trustworthy by Statelix standards."
        elif mci.total_score > 0.5:
            status = "It is in a Warning state. Some contracts are barely met."
        else:
            status = "It has been Rejected. I cannot endorse this result."
            
        reasons = "\n".join([f"- {msg}" for msg in self.report.messages])
        return f"{header}\n{status}\n\nSpecific Objections:\n{reasons if reasons else 'None. The model is mathematically sound.'}"

    def _explain_suggestions(self) -> str:
        if not self.report.suggestions:
            return "The model is already optimal according to current diagnostics. No further refinements are suggested."
            
        suggs = "\n".join([f"- {s}" for s in self.report.suggestions])
        return f"Based on the diagnostic collapse, I suggest the following strategies:\n{suggs}\n\nYou can preview these in the 'Suggestions' or 'Manifold' tabs."

    def _explain_assumptions(self) -> str:
        # Extract assumptions from storyteller summary (usually section 2)
        if "### 2. Interpretation Hints" in self.summary:
            hints = self.summary.split("### 2.")[1].split("### 3.")[0]
            return f"Causal validity depends on several critical assumptions:\n{hints}"
        return "I don't have detailed assumption documentation for this specific model type yet."
