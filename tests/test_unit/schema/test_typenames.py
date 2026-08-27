"""Tests for binding schema types and enumeration members to generated type names."""

import sys
from enum import IntEnum, StrEnum
from typing import Annotated, Literal

import pytest
from pydantic import Field, TypeAdapter

from swc.aeon.schema import BaseSchema, DiscriminatorTypeMixin, SchemaEnum, bind_typename

NAMESPACE = "Aeon.Test"


@pytest.fixture
def namespaced(monkeypatch):
    """Declares a namespace on this module for the duration of a test."""
    monkeypatch.setattr(sys.modules[__name__], "SGEN_NAMESPACE", NAMESPACE, raising=False)
    return monkeypatch


def typename(schema_type):
    """Returns the type name bound to a model or enumeration, if any."""
    return TypeAdapter(schema_type).json_schema().get("x-sgen-typename")


def member_names(schema_type):
    """Returns the member names supplied for an enumeration, if any."""
    return TypeAdapter(schema_type).json_schema().get("x-enumNames")


def test_bind_typename_tags_in_place():
    """Test that `bind_typename` tags in place and returns the schema it was given."""
    schema = {"type": "object"}
    result = bind_typename(schema, "Aeon.Test.Thing")
    assert result is schema
    assert schema["x-sgen-typename"] == "Aeon.Test.Thing"


def test_typename_derived_from_module_namespace(namespaced):
    """Test that a model is named in the namespace declared by its module."""

    class Derived(BaseSchema):
        """A model in a namespaced module."""

    assert typename(Derived) == f"{NAMESPACE}.Derived"


def test_typename_absent_without_module_namespace():
    """Test that a model in a module declaring no namespace is left untagged."""

    class Untagged(BaseSchema):
        """A model in a module with no namespace."""

    assert typename(Untagged) is None


def test_typename_recomputed_for_subclass(namespaced):
    """Test that a subclass takes its own name rather than the name of its base."""

    class Base(BaseSchema):
        """The base."""

    class Sub(Base):
        """The subclass."""

    assert typename(Base) == f"{NAMESPACE}.Base"
    assert typename(Sub) == f"{NAMESPACE}.Sub"


def test_typename_overridden_by_class_keyword():
    """Test that `sgen_namespace` names a model owned by another package."""

    class Foreign(BaseSchema, sgen_namespace="OpenEphys.Onix1"):
        """A model describing a type owned elsewhere."""

    assert typename(Foreign) == "OpenEphys.Onix1.Foreign"


def test_class_keyword_overrides_module_namespace(namespaced):
    """Test that `sgen_namespace` wins over the namespace declared by the module."""

    class Foreign(BaseSchema, sgen_namespace="OpenEphys.Onix1"):
        """A model describing a type owned elsewhere."""

    assert typename(Foreign) == "OpenEphys.Onix1.Foreign"


def test_inherited_typename_dropped_without_namespace(namespaced):
    """Test that a subclass declaring no namespace does not claim the name of its base."""

    class Base(BaseSchema):
        """Defined while the module declares a namespace."""

    namespaced.undo()

    class Naive(Base):
        """Defined after the namespace is gone, as a consumer module would be."""

    assert typename(Base) == f"{NAMESPACE}.Base"
    assert typename(Naive) is None


def test_explicit_typename_survives_without_namespace(namespaced):
    """Test that a name written into the model config is kept rather than dropped."""

    class Base(BaseSchema):
        """Defined while the module declares a namespace."""

    namespaced.undo()

    class Explicit(Base):
        """Names itself, as `bind_typename` documents."""

        model_config = {"json_schema_extra": bind_typename({}, "Third.Party.Explicit")}

    assert typename(Explicit) == "Third.Party.Explicit"


@pytest.mark.parametrize("subclass_namespace", [True, False], ids=["tagged", "untagged"])
def test_base_typename_unchanged_by_subclass(namespaced, subclass_namespace):
    """Test that defining a subclass leaves the name of its base alone.

    Pydantic shares the `json_schema_extra` dictionary between a model and its base, so
    both branches have to copy it before writing.
    """

    class Base(BaseSchema):
        """The base."""

    if not subclass_namespace:
        namespaced.undo()

    class Sub(Base):
        """The subclass."""

    assert typename(Sub) == (f"{NAMESPACE}.Sub" if subclass_namespace else None)
    assert typename(Base) == f"{NAMESPACE}.Base"


def test_generated_schema_extension_preserved(namespaced):
    """Test that a model generating its own schema extension keeps it and stays untagged."""

    def generate(schema):
        schema["x-generated"] = True

    class Generated(BaseSchema):
        """Supplies a callable rather than a mapping, which cannot be merged into."""

        model_config = {"json_schema_extra": generate}

    schema = TypeAdapter(Generated).json_schema()
    assert schema["x-generated"] is True
    assert "x-sgen-typename" not in schema


def test_sibling_extension_keys_preserved(namespaced):
    """Test that binding a name leaves other schema extension entries in place."""

    class Annotated(BaseSchema):
        """Carries an unrelated extension entry."""

        model_config = {"json_schema_extra": {"x-unrelated": "kept"}}

    schema = TypeAdapter(Annotated).json_schema()
    assert schema["x-sgen-typename"] == f"{NAMESPACE}.Annotated"
    assert schema["x-unrelated"] == "kept"


def test_enum_typename_derived_from_module_namespace(namespaced):
    """Test that an enumeration is named in the namespace declared by its module."""

    class Colour(SchemaEnum):
        """An enumeration in a namespaced module."""

        RED = "Red"

    assert typename(Colour) == f"{NAMESPACE}.Colour"


def test_enum_typename_absent_without_module_namespace():
    """Test that an enumeration in a module declaring no namespace is left untagged."""

    class Colour(SchemaEnum):
        """An enumeration in a module with no namespace."""

        RED = "Red"

    assert typename(Colour) is None


def test_enum_typename_overridden_by_class_keyword():
    """Test that `sgen_namespace` names an enumeration owned by another package."""

    class Foreign(SchemaEnum, sgen_namespace="OpenEphys.Onix1"):
        """An enumeration describing a type owned elsewhere."""

        A = 1

    assert typename(Foreign) == "OpenEphys.Onix1.Foreign"


def test_member_names_supplied_for_integer_values():
    """Test that an integer enumeration carries its member names in Pascal case."""

    class Colour(SchemaEnum, IntEnum):
        """Members an integer value cannot name."""

        RED = 0
        DARK_BLUE = 1

    assert member_names(Colour) == ["Red", "DarkBlue"]


def test_member_names_absent_for_string_values():
    """Test that a string enumeration is left to name its own members.

    Supplying them would replace a name the generated code already takes from the value,
    which the YAML round trip depends on.
    """

    class Colour(SchemaEnum, StrEnum):
        """Members a string value already names."""

        RED = "Red"
        DARK_BLUE = "DarkBlue"

    assert member_names(Colour) is None


def test_aliased_member_names_align_with_values():
    """Test that an alias does not shift the member names onto the wrong values.

    An alias appears in the values but not when iterating the class, so the names have to
    come from `__members__`.
    """

    class Colour(SchemaEnum, IntEnum):
        """Carries an alias for its first member."""

        RED = 0
        CRIMSON = 0
        BLUE = 1

    schema = TypeAdapter(Colour).json_schema()
    assert schema["x-enumNames"] == ["Red", "Crimson", "Blue"]
    assert len(schema["x-enumNames"]) == len(schema["enum"])


def test_pascal_case_member_name_unchanged():
    """Test that a name outside the upper case convention is passed through."""

    class Colour(SchemaEnum, IntEnum):
        """Names its members in Pascal case rather than upper case."""

        DarkBlue = 0

    assert member_names(Colour) == ["DarkBlue"]


def test_discriminator_type_selects_union_member():
    """Test that `DiscriminatorTypeMixin` gives each type the literal a union selects on."""

    class Headstage(BaseSchema):
        """The common base."""

    class Alpha(DiscriminatorTypeMixin, Headstage):
        """One member of the union."""

    class Beta(DiscriminatorTypeMixin, Headstage):
        """Another member of the union."""

    assert Alpha.model_fields["discriminator_type"].annotation == Literal["Alpha"]
    assert Beta().discriminator_type == "Beta"

    union = TypeAdapter(Annotated[Alpha | Beta, Field(discriminator="discriminator_type")])
    assert isinstance(union.validate_python({"discriminatorType": "Beta"}), Beta)
