"""Validate warm-up sequencing and local speech configuration."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SpeechConfig:
    """Configure queued local text-to-speech announcements.

    Attributes are scalar command-line settings for ``espeak-ng``. ``phrases``
    maps stable event names to user-facing text; episode phrases may contain an
    ``{episode}`` integer placeholder.
    """

    enabled: bool
    command: str
    voice: str
    speed: int
    volume: int
    timeout_sec: float
    phrases: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SpeechConfig":
        """Return validated speech settings from the top-level YAML mapping."""

        data = _mapping(value, "speech")
        command = _non_empty_string(data.get("command", "espeak-ng"), "speech.command")
        voice = _non_empty_string(data.get("voice", "en-us"), "speech.voice")
        speed = _integer_range(data.get("speed", 160), "speech.speed", 80, 450)
        volume = _integer_range(data.get("volume", 100), "speech.volume", 0, 200)
        defaults = {
            "warmup_started": "warm up",
            "warmup_finished": "warm up finished",
            "episode_started": "episode {episode} started",
            "episode_finished": "episode {episode} finished",
            "episode_discarded": "episode {episode} discarded",
            "reset_started": "resetting",
            "reset_finished": "reset finished",
        }
        phrases = dict(defaults)
        phrases.update(_mapping(data.get("phrases"), "speech.phrases"))
        for name, phrase in phrases.items():
            _non_empty_string(phrase, f"speech.phrases.{name}")
        return cls(
            enabled=_boolean(data.get("enabled", True), "speech.enabled"),
            command=command,
            voice=voice,
            speed=speed,
            volume=volume,
            timeout_sec=_positive_number(
                data.get("timeout_sec", 10.0), "speech.timeout_sec"
            ),
            phrases=phrases,
        )

    def phrase(self, event: str, *, episode: int | None = None) -> str:
        """Format one configured announcement for an optional episode index."""

        phrase = self.phrases[event]
        return phrase.format(episode=episode) if episode is not None else phrase


@dataclass(frozen=True)
class WarmupConfig:
    """Configure the pre-recording hardware readiness state machine.

    Time values are seconds. The coordinator waits for every enabled camera and
    tactile output, resets all actuators to ``xarm.initial_joint_positions``,
    then holds the stable pose before unlocking recorder controls.
    """

    enabled: bool
    sensor_ready_stable_time_sec: float
    hardware_timeout_sec: float
    finished_delay_sec: float
    require_all_enabled_devices: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "WarmupConfig":
        """Return validated warm-up timing and readiness policy settings."""

        data = _mapping(value, "warmup")
        return cls(
            enabled=_boolean(data.get("enabled", False), "warmup.enabled"),
            sensor_ready_stable_time_sec=_positive_number(
                data.get("sensor_ready_stable_time_sec", 0.5),
                "warmup.sensor_ready_stable_time_sec",
            ),
            hardware_timeout_sec=_positive_number(
                data.get("hardware_timeout_sec", 30.0),
                "warmup.hardware_timeout_sec",
            ),
            finished_delay_sec=_positive_number(
                data.get("finished_delay_sec", 2.0),
                "warmup.finished_delay_sec",
            ),
            require_all_enabled_devices=_boolean(
                data.get("require_all_enabled_devices", True),
                "warmup.require_all_enabled_devices",
            ),
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return a YAML mapping or raise a field-specific error."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _boolean(value: Any, name: str) -> bool:
    """Return a strict YAML boolean rather than accepting truthy strings."""

    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _non_empty_string(value: Any, name: str) -> str:
    """Return a stripped non-empty YAML string."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_number(value: Any, name: str) -> float:
    """Return a positive finite scalar number in seconds."""

    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _integer_range(value: Any, name: str, minimum: int, maximum: int) -> int:
    """Return an integer inside one inclusive command-line range."""

    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value
