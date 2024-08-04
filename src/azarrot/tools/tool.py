import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class LocalizedToolParameter:
    name: str
    type: str
    description: str | None
    required: bool


@dataclass
class LocalizedToolDescription:
    name: str
    display_name: str | None
    description: str | None
    parameters: list[LocalizedToolParameter]

    def parameters_json(self) -> str:
        return json.dumps([p.__dict__ for p in self.parameters])


@dataclass
class ToolParameter:
    name: str
    type: Literal["string", "integer", "number", "object", "array", "boolean", "null"]
    description: dict[str, str]
    required: bool

    def get_description(self, locale: str) -> str:
        return self.description[locale]

    def to_localized(self, locale: str) -> LocalizedToolParameter:
        return LocalizedToolParameter(
            name=self.name, type=self.type, description=self.get_description(locale), required=self.required
        )


@dataclass
class ToolDescription:
    name: str
    default_locale: str
    display_name: dict[str, str]
    description: dict[str, str]
    parameters: list[ToolParameter]

    def get_display_name(self, locale: str | None = None) -> str:
        return self.display_name[locale if locale is not None else self.default_locale]

    def get_description(self, locale: str | None = None) -> str:
        return self.description[locale if locale is not None else self.default_locale]

    def to_localized(self, locale: str | None) -> LocalizedToolDescription:
        if locale is None:
            locale = self.default_locale

        return LocalizedToolDescription(
            name=self.name,
            display_name=self.get_display_name(locale),
            description=self.get_description(locale),
            parameters=[p.to_localized(locale) for p in self.parameters],
        )


class Tool(ABC):
    @abstractmethod
    def description(self) -> ToolDescription:
        pass

    @abstractmethod
    def execute(self, **kwargs: dict[str, Any]) -> Any:
        pass
