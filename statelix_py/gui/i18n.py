"""
Statelix Internationalization (i18n) Module

Provides a simple translation system for Japanese/English dual language support.

Usage:
    from statelix_py.gui.i18n import t, set_language, get_language
    
    # In widgets:
    label = QLabel(t("panel.model"))
    
    # Change language:
    set_language("en")
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

# Global state
_current_language = "ja"  # Default to Japanese
_translations: Dict[str, Dict[str, str]] = {}
_observers = []  # Callbacks for language change notifications


def _load_translations():
    """Load all translation files from locales directory."""
    global _translations
    
    locales_dir = Path(__file__).parent / "locales"
    
    for lang_file in locales_dir.glob("*.json"):
        lang_code = lang_file.stem  # e.g., "ja" from "ja.json"
        try:
            with open(lang_file, "r", encoding="utf-8") as f:
                _translations[lang_code] = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {lang_file}: {e}")
            _translations[lang_code] = {}


def t(key: str, **kwargs) -> str:
    """
    Translate a key to the current language.
    
    Parameters
    ----------
    key : str
        Translation key (e.g., "panel.model", "btn.run")
    **kwargs : dict
        Format arguments for string interpolation
        
    Returns
    -------
    str
        Translated string, or the key itself if not found
        
    Examples
    --------
    >>> t("btn.run")
    "実行" or "Run"
    
    >>> t("data.status.loaded", filename="test.csv", rows=100, cols=5)
    "✅ データ: test.csv (100行 x 5列)"
    """
    if not _translations:
        _load_translations()
    
    lang_dict = _translations.get(_current_language, {})
    text = lang_dict.get(key, key)  # Return key if not found
    
    # String interpolation
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass  # Return unformatted text if formatting fails
    
    return text


def set_language(lang: str) -> bool:
    """
    Set the current language.
    
    Parameters
    ----------
    lang : str
        Language code ("ja" or "en")
        
    Returns
    -------
    bool
        True if language was changed successfully
    """
    global _current_language
    
    if not _translations:
        _load_translations()
    
    if lang not in _translations:
        print(f"Warning: Language '{lang}' not available. Available: {list(_translations.keys())}")
        return False
    
    old_lang = _current_language
    _current_language = lang
    
    # Save to settings file
    _save_language_setting(lang)
    
    # Notify observers
    if old_lang != lang:
        for callback in _observers:
            try:
                callback(lang)
            except Exception as e:
                print(f"Warning: Language change callback error: {e}")
    
    return True


def get_language() -> str:
    """Get the current language code."""
    return _current_language


def get_available_languages() -> list:
    """Get list of available language codes."""
    if not _translations:
        _load_translations()
    return list(_translations.keys())


def add_language_observer(callback):
    """
    Register a callback to be notified when language changes.
    
    The callback receives the new language code as argument.
    """
    _observers.append(callback)


def remove_language_observer(callback):
    """Remove a language change observer."""
    if callback in _observers:
        _observers.remove(callback)


def _save_language_setting(lang: str):
    """Save language setting to config file."""
    config_dir = Path.home() / ".statelix"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "settings.json"
    
    settings = {}
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except Exception:
            pass
    
    settings["language"] = lang
    
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save language setting: {e}")


def _load_language_setting():
    """Load language setting from config file."""
    global _current_language
    
    config_file = Path.home() / ".statelix" / "settings.json"
    
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                settings = json.load(f)
                _current_language = settings.get("language", "ja")
        except Exception:
            pass


# Initialize on import
_load_translations()
_load_language_setting()
