import pytest
import logging
from texteller.globals import Globals


def test_singleton_pattern():
    """Test that Globals uses the singleton pattern correctly."""
    # Create two instances
    globals1 = Globals()
    globals2 = Globals()

    # Both variables should reference the same object
    assert globals1 is globals2

    # Modifying one should affect the other
    globals1.test_attr = "test_value"
    assert globals2.test_attr == "test_value"

    # Clean up after test
    delattr(globals1, "test_attr")


def test_predefined_attributes():
    """Test predefined attributes have correct default values."""
    globals_instance = Globals()
    assert globals_instance.repo_name == "OleehyO/TexTeller"
    assert globals_instance.logging_level == logging.INFO


def test_attribute_modification():
    """Test that attributes can be modified."""
    globals_instance = Globals()

    # Modify existing attribute
    original_repo_name = globals_instance.repo_name
    globals_instance.repo_name = "NewRepo/NewName"
    assert globals_instance.repo_name == "NewRepo/NewName"

    assert Globals().logging_level == logging.INFO
    Globals().logging_level = logging.DEBUG
    assert Globals().logging_level == logging.DEBUG

    # Reset for other tests
    globals_instance.repo_name = original_repo_name
    globals_instance.logging_level = logging.INFO


def test_dynamic_attributes():
    """Test that new attributes can be added dynamically."""
    globals_instance = Globals()

    # Add new attribute
    globals_instance.new_attribute = "new_value"
    assert globals_instance.new_attribute == "new_value"

    # Clean up after test
    delattr(globals_instance, "new_attribute")


def test_representation():
    """Test the string representation of Globals."""
    globals_instance = Globals()
    repr_string = repr(globals_instance)

    # Check that repr contains class name and is formatted as expected
    assert repr_string.startswith("<Globals:")
    assert "repo_name" in repr_string
    assert "logging_level" in repr_string
