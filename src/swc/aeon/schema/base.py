"""Base classes for defining experiment configuration and data models."""

import datetime
import os
import sys
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Literal, Self, TypeVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from pydantic.alias_generators import to_camel, to_pascal
from pydantic.json_schema import JsonSchemaValue

from swc.aeon.io.reader import Reader

_TYPENAME_KEY = "x-sgen-typename"


def bind_typename(schema: JsonSchemaValue, typename: str) -> JsonSchemaValue:
    """Applies the `x-sgen-typename` tag, binding a definition to an existing type.

    Args:
        schema: The JSON schema definition to tag, modified in place.
        typename: Fully qualified name of the type to bind.

    Returns:
        The same schema, so it can be used inline in a `model_config` declaration.
    """
    schema[_TYPENAME_KEY] = typename
    return schema


def _inherited_typename(cls: type) -> str | None:
    """Returns the type name a class carries from the nearest base configuring one."""
    for base in cls.__mro__[1:]:
        config = getattr(base, "model_config", None)
        if isinstance(config, dict):
            extra = config.get("json_schema_extra")
            return extra.get(_TYPENAME_KEY) if isinstance(extra, dict) else None
    return None


class DiscriminatorTypeMixin:
    """Sets `discriminator_type` to the subclass name, for types in a discriminated union."""

    def __init_subclass__(cls, **kwargs):
        """Injects `discriminator_type` as a Literal of the subclass name."""
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        cls.__annotations__["discriminator_type"] = Literal[name]
        cls.discriminator_type = name


class BaseSchema(BaseModel):
    """The base class for all experiment configuration and data models."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        arbitrary_types_allowed=True,
        field_title_generator=lambda n, _: to_pascal(n),
        populate_by_name=True,
        from_attributes=True,
    )

    _container_prefix: str = ""
    _container: "BaseSchema | None" = None

    def __init_subclass__(cls, sgen_namespace: str | None = None, **kwargs):
        """Accepts the optional `sgen_namespace` keyword, which `object` would reject."""
        super().__init_subclass__(**kwargs)

    @classmethod
    def __pydantic_init_subclass__(cls, sgen_namespace: str | None = None, **kwargs):
        """Binds the subclass to a type name in the namespace declared by its module.

        The namespace is the `SGEN_NAMESPACE` of the declaring module, or `sgen_namespace`
        for a model describing a type owned elsewhere. A module declaring neither leaves
        its models untagged, dropping any name inherited from a base so that a subclass
        never claims to be the type of its parent. A model generating its schema
        extension itself is left alone.
        """
        super().__pydantic_init_subclass__(**kwargs)
        extra = cls.model_config.get("json_schema_extra")
        if callable(extra):
            return

        module = sys.modules.get(cls.__module__)
        namespace = sgen_namespace or getattr(module, "SGEN_NAMESPACE", None)
        extra = dict(extra or {})
        if namespace is not None:
            typename = f"{namespace}.{cls.__name__}"
            cls.model_config["json_schema_extra"] = bind_typename(extra, typename)
        elif _TYPENAME_KEY in extra and extra[_TYPENAME_KEY] == _inherited_typename(cls):
            del extra[_TYPENAME_KEY]
            cls.model_config["json_schema_extra"] = extra

    def _join_pattern_prefix(self, pattern_prefix: str) -> str:
        return self._container_prefix

    def _resolve_pattern_prefix(self) -> str:
        container = self._container
        pattern_prefix = self._container_prefix
        while container is not None:
            pattern_prefix = container._join_pattern_prefix(pattern_prefix)
            container = container._container

        return pattern_prefix

    @model_validator(mode="after")
    def _validate_container_prefix(self) -> Self:
        for name in self.__class__.model_fields:
            f = getattr(self, name)
            if isinstance(f, dict):
                for nk, nv in f.items():
                    if isinstance(nv, BaseSchema):
                        nv._container_prefix = nk
                        nv._container = self
            elif isinstance(f, BaseSchema):
                f._container_prefix = to_pascal(name)
                f._container = self
        return self


class Experiment(BaseSchema):
    """The base class for creating experiment models."""

    workflow: str = Field(description="Path to the workflow running the experiment.")
    commit: str = Field(description="Commit hash of the experiment repo.")
    repository_url: str = Field(
        description="The URL of the git repository used to version experiment source code."
    )


class Dataset(BaseSchema):
    """The base class for creating dataset models."""

    def _join_pattern_prefix(self, pattern_prefix: str) -> str:
        return os.path.join(self._container_prefix, pattern_prefix)


ModelT = TypeVar("ModelT", bound=BaseSchema)


class Metadata(Reader):
    """Extracts metadata information from all epochs in the dataset."""

    def __init__(self, type: type[ModelT], pattern="Metadata"):
        """Initialize the reader object with the specified model type and optional pattern."""
        super().__init__(pattern, columns=["metadata", "epoch"], extension="json")
        self.type = TypeAdapter(type)

    def read(self, path: Path) -> pd.DataFrame:
        """Returns metadata for the epoch associated with the specified file."""
        epoch_str = path.parts[-2]
        date_str, time_str = epoch_str.split("T")
        time = datetime.datetime.fromisoformat(date_str + "T" + time_str.replace("-", ":"))
        metadata = path.read_text()
        data = {"metadata": [self.type.validate_json(metadata)], "epoch": [epoch_str]}
        return pd.DataFrame(data, index=pd.Series(time), columns=self.columns)


_SelfBaseSchema = TypeVar("_SelfBaseSchema", bound=BaseSchema)
_ReaderT = TypeVar("_ReaderT", bound=Reader)


def data_reader(func: Callable[[_SelfBaseSchema, str], _ReaderT]) -> cached_property[_ReaderT]:
    """Decorator to include a data reader as `cached_property` in experiment dataset models."""

    def decorator(self: _SelfBaseSchema) -> _ReaderT:
        pattern_prefix = self._resolve_pattern_prefix()  # pyright: ignore[reportPrivateUsage]
        return func(self, pattern_prefix)

    return cached_property(decorator)
