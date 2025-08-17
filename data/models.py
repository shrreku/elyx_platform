from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class HealthMetrics:
    blood_sugar: float
    blood_pressure: str
    weight: float
    a1c: Optional[float] = None


@dataclass
class WeeklyReport:
    week: int
    metrics: HealthMetrics
    events: List[str]
    agent_recommendations: Dict[str, str]
    user_adherence: float
    notes: str


@dataclass
class DiagnosticPanel:
    week: int
    results: Dict[str, str]
    delta: Dict[str, str]


@dataclass
class MicroReplan:
    week: int
    version: str
    reason: str
    changes: List[str]


@dataclass
class JourneyState:
    current_week: int
    total_weeks: int
    active_conditions: List[str]
    current_medications: List[str]
    exercise_plan: Dict
    nutrition_plan: Dict
    diagnostic_panels: List[DiagnosticPanel]
    micro_replans: List[MicroReplan]


