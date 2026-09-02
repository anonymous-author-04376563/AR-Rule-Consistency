from dataclasses import dataclass


VALID_INCONSISTENCY_TYPES = {"consistent", "omission", "commission", "confusion"}


@dataclass(frozen=True)
class ExperimentConfig:
    scene: int
    rule: int
    inconsistency_type: str
    model: str = "gpt-5.6-luna"
    thinking: str = "low"
    debug_save_images: bool = False
    save_every_n: int = 0
    max_debug_images: int = 0

    def __post_init__(self) -> None:
        normalized_type = self.inconsistency_type.strip().casefold()
        if self.scene < 1 or self.rule < 1:
            raise ValueError("scene and rule must be positive integers")
        if normalized_type not in VALID_INCONSISTENCY_TYPES:
            raise ValueError(
                f"inconsistency_type must be one of {sorted(VALID_INCONSISTENCY_TYPES)}"
            )
        if self.save_every_n < 0 or self.max_debug_images < 0:
            raise ValueError("debug image limits cannot be negative")
        object.__setattr__(self, "inconsistency_type", normalized_type)