"""Pure freshness-based component health evaluation."""

from dataclasses import dataclass, field

OK = 0
WARN = 1
ERROR = 2


@dataclass(frozen=True)
class HealthResult:
    """One evaluated hardware state without ROS message dependencies."""

    level: int
    state: str
    message: str
    age_sec: float | None
    missing_topics: tuple[str, ...]


@dataclass
class ComponentHealth:
    """Track receipt times for every output required from one component.

    Args:
        component_id: Stable UI and diagnostic identifier.
        kind: Hardware or subsystem category.
        enabled: Whether YAML enables the component.
        required: Whether missing data is an error instead of a warning.
        topics: Complete tuple of expected ROS topic names.
        timeout_sec: Maximum receipt age in seconds, or ``None`` for latched state.
        startup_grace_sec: Seconds allowed before the first sample.
    """

    component_id: str
    kind: str
    enabled: bool
    required: bool
    topics: tuple[str, ...]
    timeout_sec: float | None
    startup_grace_sec: float
    started_at: float
    last_seen: dict[str, float] = field(default_factory=dict)

    def observe(self, topic: str, now: float) -> None:
        """Record receipt of one expected topic at monotonic time ``now``."""

        if topic in self.topics:
            self.last_seen[topic] = float(now)

    def evaluate(self, now: float) -> HealthResult:
        """Return disabled, starting, healthy or stale component state."""

        if not self.enabled:
            return HealthResult(OK, "DISABLED", "Disabled by YAML", None, ())
        missing = tuple(topic for topic in self.topics if topic not in self.last_seen)
        if missing and now - self.started_at <= self.startup_grace_sec:
            return HealthResult(WARN, "STARTING", "Waiting for data", None, missing)
        if self.timeout_sec is None:
            stale = missing
        else:
            stale = tuple(
                topic
                for topic in self.topics
                if topic not in self.last_seen
                or now - self.last_seen[topic] > self.timeout_sec
            )
        if stale:
            level = ERROR if self.required else WARN
            ages = [now - self.last_seen[topic] for topic in self.topics if topic in self.last_seen]
            return HealthResult(
                level,
                "STALE",
                "Required output is stale" if self.required else "Optional output is stale",
                max(ages) if ages else None,
                stale,
            )
        ages = [now - self.last_seen[topic] for topic in self.topics]
        return HealthResult(OK, "OK", "Connected", max(ages) if ages else 0.0, ())
