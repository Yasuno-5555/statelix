"""
Truth Collapse Simulator: How Does Your Conclusion Die?

Systematically destroys assumptions to reveal:
- The order in which truth collapses
- When claims become unsustainable
- How irreversible each step is
- The archetype of your analysis's death

"この結論は、どんな順番で、どれくらい惨めに死ぬか"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from copy import deepcopy

try:
    from .assumption_path import AssumptionPath, AssumptionState, PathPoint
    from .assumption_tube import AssumptionTube, TubeMetrics
    from .claim_budget import ClaimBudget, ClaimStrength, ClaimBudgetCalculator
    from .honest_sentence import HonestSentence, HonestSentenceGenerator
    from .one_way_door import OneWayDoor, OneWayDoorAnalyzer, DoorType
except ImportError:
    from statelix_py.core.assumption_path import AssumptionPath, AssumptionState, PathPoint
    from statelix_py.core.assumption_tube import AssumptionTube, TubeMetrics
    from statelix_py.core.claim_budget import ClaimBudget, ClaimStrength, ClaimBudgetCalculator
    from statelix_py.core.honest_sentence import HonestSentence, HonestSentenceGenerator
    from statelix_py.core.one_way_door import OneWayDoor, OneWayDoorAnalyzer, DoorType


# =============================================================================
# Collapse Archetype
# =============================================================================

class CollapseArchetype(Enum):
    """
    Death style classifications.
    """
    GLASS_TOWER = "glass_tower"      # Strong initially, shatters instantly
    MARSHMALLOW = "marshmallow"      # Never strong, survives but says nothing
    STEEL_ROD = "steel_rod"          # Survives long, thin claims
    PAPER_TIGER = "paper_tiger"      # Looks strong, tube is paper-thin
    PHOENIX = "phoenix"              # Recovers after collapse
    UNKNOWN = "unknown"
    
    @property
    def icon(self) -> str:
        icons = {
            CollapseArchetype.GLASS_TOWER: "🗼",
            CollapseArchetype.MARSHMALLOW: "☁️",
            CollapseArchetype.STEEL_ROD: "🔩",
            CollapseArchetype.PAPER_TIGER: "🐯",
            CollapseArchetype.PHOENIX: "🔥",
            CollapseArchetype.UNKNOWN: "❓"
        }
        return icons.get(self, "❓")
    
    @property
    def description(self) -> str:
        descriptions = {
            CollapseArchetype.GLASS_TOWER: "初期は強いが、1仮定で即死",
            CollapseArchetype.MARSHMALLOW: "ずっと曖昧で最後まで生き残るが何も言えない",
            CollapseArchetype.STEEL_ROD: "最後まで耐えるが主張は細い",
            CollapseArchetype.PAPER_TIGER: "有意だが Tube が最初から薄い",
            CollapseArchetype.PHOENIX: "崩壊後に回復する稀有なケース",
            CollapseArchetype.UNKNOWN: "分類不能"
        }
        return descriptions.get(self, "分類不能")


# =============================================================================
# Collapse Stage
# =============================================================================

@dataclass
class CollapseStage:
    """
    A single stage in the collapse sequence.
    """
    stage_index: int
    destroyed_assumption: str
    state_before: AssumptionState
    state_after: AssumptionState
    
    # Metrics at this stage
    robustness_radius: float
    truth_thickness: float
    claim_strength: ClaimStrength
    honest_sentence: str
    
    # One-way doors
    doors_crossed: List[str] = field(default_factory=list)
    
    # Derived metrics
    is_terminal: bool = False  # Can no longer make claims
    sentence_degraded: bool = False  # Sentence got weaker
    
    @property
    def is_alive(self) -> bool:
        """Is the analysis still making meaningful claims?"""
        return self.claim_strength not in [ClaimStrength.NONE, ClaimStrength.MINIMAL]
    
    def __repr__(self) -> str:
        status = "DEAD" if self.is_terminal else ("WEAK" if not self.is_alive else "OK")
        return f"Stage[{self.stage_index}] {self.destroyed_assumption} → {self.claim_strength.value} [{status}]"


@dataclass
class CollapseSchedule:
    """
    Complete collapse timeline.
    """
    stages: List[CollapseStage]
    archetype: CollapseArchetype
    
    # Summary metrics
    initial_strength: ClaimStrength
    final_strength: ClaimStrength
    death_stage: Optional[int]  # When claims became NONE
    critical_assumption: Optional[str]  # Which assumption killed it
    
    # Irreversibility
    irreversibility_index: float  # 0-1, higher = harder to recover
    reversibility_tested: bool = False
    
    def __len__(self) -> int:
        return len(self.stages)
    
    def get_stage(self, index: int) -> Optional[CollapseStage]:
        """Get a specific stage."""
        if 0 <= index < len(self.stages):
            return self.stages[index]
        return None
    
    def strength_trajectory(self) -> List[str]:
        """Get trajectory of claim strengths."""
        return [s.claim_strength.value for s in self.stages]
    
    def sentence_trajectory(self) -> List[str]:
        """Get trajectory of honest sentences."""
        return [s.honest_sentence for s in self.stages]
    
    def __repr__(self) -> str:
        return f"CollapseSchedule({self.archetype.value}, stages={len(self.stages)}, death@{self.death_stage})"


# =============================================================================
# Collapse Simulator
# =============================================================================

class TruthCollapseSimulator:
    """
    Systematically destroys assumptions to reveal how conclusions die.
    
    The final judgment device for Statelix.
    
    Example:
        >>> sim = TruthCollapseSimulator()
        >>> report = sim.simulate(tube, path)
        >>> 
        >>> print(f"Archetype: {report.schedule.archetype.icon} {report.schedule.archetype.description}")
        >>> print(f"Death at stage: {report.schedule.death_stage}")
        >>> print(f"Critical assumption: {report.schedule.critical_assumption}")
        >>> 
        >>> for stage in report.schedule.stages:
        ...     print(f"{stage.stage_index}: {stage.destroyed_assumption} → {stage.claim_strength.value}")
    """
    
    # Default destruction order (epistemologically motivated)
    DEFAULT_DESTRUCTION_ORDER = [
        'normality',        # First: distributional assumptions
        'homoscedasticity', # Second: variance assumptions
        'stationarity',     # Third: temporal assumptions
        'exogeneity',       # Fourth: causal assumptions
        'independence',     # Fifth: structural assumptions
        'linearity'         # Last: functional form
    ]
    
    def __init__(self, destruction_order: Optional[List[str]] = None):
        """
        Initialize simulator.
        
        Args:
            destruction_order: Custom order for assumption destruction
        """
        self.destruction_order = destruction_order or self.DEFAULT_DESTRUCTION_ORDER
        self.budget_calc = ClaimBudgetCalculator()
        self.sentence_gen = HonestSentenceGenerator()
        self.door_analyzer = OneWayDoorAnalyzer()
    
    def simulate(
        self,
        tube: Optional[AssumptionTube] = None,
        path: Optional[AssumptionPath] = None,
        model: Optional[Any] = None,
        data: Optional[Dict] = None,
        test_reversibility: bool = True
    ) -> 'CollapseReport':
        """
        Run full collapse simulation.
        
        Args:
            tube: Base AssumptionTube
            path: Base AssumptionPath
            model: Optional model for regeneration
            data: Optional data for regeneration
            test_reversibility: Whether to test if collapse is reversible
        
        Returns:
            CollapseReport with complete analysis
        """
        # Get initial state
        if path and path.points:
            initial_state = path.points[0].state
        else:
            initial_state = AssumptionState.classical()
        
        # Compute initial metrics
        if tube:
            initial_metrics = tube.compute_metrics()
            initial_budget = self.budget_calc.from_tube_metrics(initial_metrics)
        else:
            initial_metrics = TubeMetrics(
                robustness_radius=0.5,
                truth_thickness=0.3,
                truth_thickness_location=0.5,
                total_variance=0.2
            )
            initial_budget = self.budget_calc.from_tube_metrics(initial_metrics)
        
        initial_sentence = self.sentence_gen.generate(tube=tube, path=path, budget=initial_budget)
        
        # Run destruction sequence
        stages = self._run_destruction_sequence(
            initial_state=initial_state,
            initial_metrics=initial_metrics,
            initial_budget=initial_budget,
            initial_sentence=initial_sentence.sentence,
            tube=tube,
            path=path
        )
        
        # Analyze results
        death_stage, critical_assumption = self._find_death_point(stages)
        archetype = self._classify_archetype(stages, initial_budget)
        
        # Test reversibility
        irreversibility = 0.0
        if test_reversibility and death_stage is not None:
            irreversibility = self._compute_irreversibility(
                stages, death_stage, initial_state
            )
        
        schedule = CollapseSchedule(
            stages=stages,
            archetype=archetype,
            initial_strength=initial_budget.max_strength,
            final_strength=stages[-1].claim_strength if stages else initial_budget.max_strength,
            death_stage=death_stage,
            critical_assumption=critical_assumption,
            irreversibility_index=irreversibility,
            reversibility_tested=test_reversibility
        )
        
        return CollapseReport(
            schedule=schedule,
            initial_metrics=initial_metrics,
            initial_budget=initial_budget,
            initial_sentence=initial_sentence.sentence,
            final_sentence=stages[-1].honest_sentence if stages else initial_sentence.sentence
        )
    
    def _run_destruction_sequence(
        self,
        initial_state: AssumptionState,
        initial_metrics: TubeMetrics,
        initial_budget: ClaimBudget,
        initial_sentence: str,
        tube: Optional[AssumptionTube],
        path: Optional[AssumptionPath]
    ) -> List[CollapseStage]:
        """Run the destruction sequence."""
        stages = []
        current_state = deepcopy(initial_state)
        prev_sentence = initial_sentence
        prev_strength = initial_budget.max_strength
        
        dims = ['linearity', 'independence', 'stationarity',
                'normality', 'homoscedasticity', 'exogeneity']
        
        for i, assumption in enumerate(self.destruction_order):
            if assumption not in dims:
                continue
            
            # Record state before
            state_before = deepcopy(current_state)
            
            # Destroy assumption (set to 0)
            dim_idx = dims.index(assumption)
            vec = current_state.to_vector()
            vec[dim_idx] = 0.0
            current_state = AssumptionState.from_vector(vec)
            
            # Compute metrics at this stage
            # Simulate decreased robustness as assumptions are relaxed
            destruction_factor = (len(self.destruction_order) - i) / len(self.destruction_order)
            degraded_metrics = TubeMetrics(
                robustness_radius=initial_metrics.robustness_radius * destruction_factor * 0.8,
                truth_thickness=initial_metrics.truth_thickness * destruction_factor * 0.7,
                truth_thickness_location=initial_metrics.truth_thickness_location,
                total_variance=initial_metrics.total_variance / max(destruction_factor, 0.1)
            )
            
            # Compute budget and sentence
            budget = self.budget_calc.from_tube_metrics(degraded_metrics)
            sentence_obj = self.sentence_gen.generate(budget=budget)
            
            # Check doors
            doors = self.door_analyzer.analyze(current_state)
            crossed = [d.door_type.value for d in doors if d.is_crossed]
            
            # Check if terminal
            is_terminal = budget.max_strength == ClaimStrength.NONE
            sentence_degraded = budget.max_strength != prev_strength
            
            stage = CollapseStage(
                stage_index=i,
                destroyed_assumption=assumption,
                state_before=state_before,
                state_after=deepcopy(current_state),
                robustness_radius=degraded_metrics.robustness_radius,
                truth_thickness=degraded_metrics.truth_thickness,
                claim_strength=budget.max_strength,
                honest_sentence=sentence_obj.sentence,
                doors_crossed=crossed,
                is_terminal=is_terminal,
                sentence_degraded=sentence_degraded
            )
            
            stages.append(stage)
            prev_sentence = sentence_obj.sentence
            prev_strength = budget.max_strength
            
            # Stop if terminal
            if is_terminal:
                break
        
        return stages
    
    def _find_death_point(
        self,
        stages: List[CollapseStage]
    ) -> Tuple[Optional[int], Optional[str]]:
        """Find when analysis dies (NONE or MINIMAL claims)."""
        for stage in stages:
            if stage.claim_strength in [ClaimStrength.NONE, ClaimStrength.MINIMAL]:
                return stage.stage_index, stage.destroyed_assumption
        return None, None
    
    def _classify_archetype(
        self,
        stages: List[CollapseStage],
        initial_budget: ClaimBudget
    ) -> CollapseArchetype:
        """Classify the death style."""
        if not stages:
            return CollapseArchetype.UNKNOWN
        
        initial_str = initial_budget.max_strength
        death_stage = None
        for s in stages:
            if not s.is_alive:
                death_stage = s.stage_index
                break
        
        # Check initial thickness
        initial_thin = initial_budget.robustness_score < 0.3
        
        # Strong start, quick death
        if initial_str in [ClaimStrength.STRONG, ClaimStrength.DEFINITIVE]:
            if death_stage is not None and death_stage <= 1:
                return CollapseArchetype.GLASS_TOWER
        
        # Paper tiger: looks strong but thin
        if initial_str in [ClaimStrength.STRONG, ClaimStrength.DEFINITIVE] and initial_thin:
            return CollapseArchetype.PAPER_TIGER
        
        # Marshmallow: never strong, survives
        if initial_str in [ClaimStrength.WEAK, ClaimStrength.MINIMAL]:
            if death_stage is None or death_stage >= len(stages) - 1:
                return CollapseArchetype.MARSHMALLOW
        
        # Steel rod: survives but thin claims
        if death_stage is None and initial_str in [ClaimStrength.MODERATE, ClaimStrength.WEAK]:
            return CollapseArchetype.STEEL_ROD
        
        # Check for recovery (phoenix)
        if len(stages) >= 2:
            strengths = [s.claim_strength for s in stages]
            for i in range(1, len(strengths)):
                if (self._strength_rank(strengths[i]) > 
                    self._strength_rank(strengths[i-1])):
                    return CollapseArchetype.PHOENIX
        
        return CollapseArchetype.UNKNOWN
    
    def _strength_rank(self, strength: ClaimStrength) -> int:
        """Numeric rank for claim strength."""
        ranks = {
            ClaimStrength.NONE: 0,
            ClaimStrength.MINIMAL: 1,
            ClaimStrength.WEAK: 2,
            ClaimStrength.MODERATE: 3,
            ClaimStrength.STRONG: 4,
            ClaimStrength.DEFINITIVE: 5
        }
        return ranks.get(strength, 0)
    
    def _compute_irreversibility(
        self,
        stages: List[CollapseStage],
        death_stage: int,
        initial_state: AssumptionState
    ) -> float:
        """
        Compute how irreversible the collapse is.
        
        Higher = harder to recover.
        """
        if death_stage is None or death_stage >= len(stages):
            return 0.0
        
        death = stages[death_stage]
        
        # Count doors crossed
        n_doors = len(death.doors_crossed)
        
        # How much of the assumption space is destroyed?
        final_state = death.state_after
        initial_vec = initial_state.to_vector()
        final_vec = final_state.to_vector()
        
        destruction_ratio = 1.0 - (np.sum(final_vec) / max(np.sum(initial_vec), 1.0))
        
        # Combined irreversibility
        irreversibility = min(1.0, (n_doors * 0.2) + (destruction_ratio * 0.8))
        
        return irreversibility


# =============================================================================
# Collapse Report
# =============================================================================

@dataclass
class CollapseReport:
    """
    Complete collapse analysis report.
    """
    schedule: CollapseSchedule
    
    # Initial state
    initial_metrics: TubeMetrics
    initial_budget: ClaimBudget
    initial_sentence: str
    
    # Final state
    final_sentence: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'archetype': self.schedule.archetype.value,
            'archetype_icon': self.schedule.archetype.icon,
            'archetype_description': self.schedule.archetype.description,
            'death_stage': self.schedule.death_stage,
            'critical_assumption': self.schedule.critical_assumption,
            'irreversibility_index': self.schedule.irreversibility_index,
            'initial_strength': self.initial_budget.max_strength.value,
            'final_strength': self.schedule.final_strength.value,
            'initial_sentence': self.initial_sentence,
            'final_sentence': self.final_sentence,
            'n_stages': len(self.schedule.stages),
            'trajectory': self.schedule.strength_trajectory()
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        s = self.schedule
        
        md = f"""# Truth Collapse Report

## Archetype: {s.archetype.icon} {s.archetype.value.replace('_', ' ').title()}

> {s.archetype.description}

---

## Summary

| Metric | Value |
|--------|-------|
| Initial Strength | `{self.initial_budget.max_strength.value}` |
| Final Strength | `{s.final_strength.value}` |
| Death Stage | {s.death_stage if s.death_stage is not None else 'Survived'} |
| Critical Assumption | {s.critical_assumption or 'None'} |
| Irreversibility | {s.irreversibility_index:.2f} |

---

## The Last Honest Sentence

**Before collapse:**
> {self.initial_sentence}

**After collapse:**
> {self.final_sentence}

---

## Collapse Timeline

"""
        for stage in s.stages:
            status = "💀" if stage.is_terminal else ("📉" if stage.sentence_degraded else "✓")
            md += f"### Stage {stage.stage_index}: Destroy `{stage.destroyed_assumption}` {status}\n\n"
            md += f"- **Claim Strength:** {stage.claim_strength.value}\n"
            md += f"- **Robustness:** {stage.robustness_radius:.3f}\n"
            md += f"- **Thickness:** {stage.truth_thickness:.3f}\n"
            if stage.doors_crossed:
                md += f"- **Doors Crossed:** {', '.join(stage.doors_crossed)}\n"
            md += f"\n> {stage.honest_sentence}\n\n"
        
        return md
    
    def __repr__(self) -> str:
        return f"CollapseReport({self.schedule.archetype.value}, death@{self.schedule.death_stage})"


# =============================================================================
# Convenience Function
# =============================================================================

def simulate_collapse(
    tube: Optional[AssumptionTube] = None,
    path: Optional[AssumptionPath] = None,
    destruction_order: Optional[List[str]] = None
) -> CollapseReport:
    """
    Convenience function to simulate truth collapse.
    
    Example:
        >>> report = simulate_collapse(tube)
        >>> print(f"{report.schedule.archetype.icon} {report.schedule.archetype.description}")
    """
    sim = TruthCollapseSimulator(destruction_order=destruction_order)
    return sim.simulate(tube=tube, path=path)
