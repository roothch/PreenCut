#!/usr/bin/env python3

"""
Written by Ulf Hermjakob, USC/ISI  March-June 2024
uroman is a universal romanizer. It converts text in any script to the Latin alphabet.
This script is a Python reimplementation of an earlier Perl script, with some improvements.
The tool has been tested on 250 languages, with 100 or more sentences each.
This script is still under development and large-scale testing. Feedback welcome.
This script provides token-size caching (for faster runtimes).
Output formats include
  (1) best romanization string
  (2) best romanization edges ("best path"; incl. start and end positions with respect to the original string)
  (3) best romanization with alternatives (as applicable for ambiguous romanization)
  (4) best romanization full lattice (all edges, including superseded sub-edges)
See below for 'sample calls' under main()
"""


from __future__ import annotations
import argparse
from collections import defaultdict
# from memory_profiler import profile
import datetime
from enum import Enum
from fractions import Fraction
import gc
import json
import math
import os
from pathlib import Path
import pstats
import regex
import sys
from typing import List, Tuple
import unicodedata as ud
from __init__ import __version__, last_mod_date
DEFAULT_ROM_MAX_CACHE_SIZE = 65536
PROFILE_FLAG = "--profile"  # also used in argparse processing
if PROFILE_FLAG in sys.argv:
    import cProfile

# UTILITIES


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        print(f"Calling: {func.__name__}{args}")
        print(f"Start time: {start_time:%A, %B %d, %Y at %H:%M}")
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        time_diff = (end_time-start_time).total_seconds()
        print(f"End time: {end_time:%A, %B %d, %Y at %H:%M}")
        print(f"Duration: {time_diff} seconds")
        return result
    return wrapper


def slot_value_in_double_colon_del_list(line: str, slot: str, default: str | list | None = None) -> str | list | None:
    """For a given slot, e.g. 'cost', get its value from a line such as '::s1 of course ::s2 ::cost 0.3' -> 0.3
    The value can be an empty string, as for ::s2 in the example above."""
    m = regex.match(fr'(?:.*\s)?::{slot}(|\s+\S.*?)(?:\s+::\S.*|\s*)$', line)
    return m.group(1).strip() if m else default


def has_value_in_double_colon_del_list(line: str, slot: str) -> bool:
    return isinstance(slot_value_in_double_colon_del_list(line, slot), str)


def dequote_string(s: str) -> str:
    if isinstance(s, str):
        m = regex.match(r'''\s*(['"“])(.*)(['"”])\s*$''', s)
        if m and ((m.group(1) + m.group(3)) in ("''", '""', '“”')):
            return m.group(2)
    return s


def last_chr(s: str) -> str:
    if len(s):
        return s[len(s)-1]
    else:
        ''


def ud_numeric(char: str) -> int | float | None:
    try:
        num_f = ud.numeric(char)
        return int(num_f) if num_f.is_integer() else num_f
    except (ValueError, TypeError):
        return None


def robust_str_to_num(num_s: str, filename: str = None, line_number: int | None = None, silent: bool = False) \
        -> int | float | None:
    if isinstance(num_s, str):
        try:
            return float(num_s) if "." in num_s else int(num_s)
        except ValueError:
            if not silent:
                sys.stderr.write(f'Cannot convert "{num_s}" to a number')
                if line_number:
                    sys.stderr.write(f' line: {line_number}')
                if filename:
                    sys.stderr.write(f' file: {filename}')
                sys.stderr.write(f'\n')
    elif isinstance(num_s, float) or isinstance(num_s, int):
        return num_s
    return None


def first_non_none(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def any_not_none(*args) -> bool:
    for arg in args:
        if arg is not None:
            return True
    return False


def add_non_none_to_dict(d: dict, key: str, value) -> None:
    if value is not None:
        d[key] = value


def fraction_char2fraction(fraction_char: str, fraction_value: float | None = None,
                           uroman: Uroman | None = None) -> Fraction | None:
    s = ''
    fraction = None
    for ud_decomp_elem in ud.decomposition(fraction_char).split():
        try:
            s += chr(int(ud_decomp_elem, 16))
        except ValueError:
            s += ud_decomp_elem
    if m := regex.match(r'<fraction>(\d+)⁄(\d+)$', s):
        numerator_s, denominator_s = m.group(1, 2)
        try:
            fraction = Fraction(int(numerator_s), int(denominator_s))
        except ValueError:
            fraction = None
    if (fraction is None) and uroman and fraction_value:
        if numerator_denominator := uroman.unicode_float2fraction(fraction_value):
            try:
                fraction = Fraction(numerator_denominator[0], numerator_denominator[1])
            except ValueError:
                fraction = None
    return fraction


def chr_name(char: str) -> str:
    """robust version of ud.name; see related Uroman.char_name() that includes names not included in UnicodeData.txt"""
    try:
        return ud.name(char)
    except (ValueError, TypeError):
        return ''


def args_get(key: str, args: argparse.Namespace | None = None):
    return vars(args)[key] if args and (key in args) else None


class DictClass:
    def __init__(self, **kw_args):
        for kw_arg in kw_args:
            kw_arg2 = kw_arg.replace('_', '-')
            value = kw_args[kw_arg]
            if not (value in (None, [], False)):
                self.__dict__[kw_arg2] = value

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, key, default=None):
        return self.__dict__[key] if key in self.__dict__ else default

    def __bool__(self):
        return len(self.__dict__) > 0


class RomRule(DictClass):
    # key: source string
    # typical attributes: s (source), t (target), prov (provenance), lcodes (language codes)
    # t_alts=t_alts (target alternatives), use_only_at_start_of_word, dont_use_at_start_of_word,
    # use_only_at_end_of_word, dont_use_at_end_of_word, use_only_for_whole_word
    pass


class Script(DictClass):
    # key: lower case script_name
    # typical attributes: script_name, direction, abugida_default_vowels, alt_script_names, languages
    pass


class RomFormat(Enum):
    """Output format of romanization"""
    STR = 'str'          # simple string
    EDGES = 'edges'      # list of edges (includes character offsets in original string)
    ALTS = 'alts'        # lattice including alternative edges
    LATTICE = 'lattice'  # lattice including alternative and superseded edges

    def __str__(self):
        return self.value


class Uroman:
    """This class loads and maintains uroman data independent of any specific text corpus.
    Typically, only a single instance will be used. (In contrast to multiple lattice instances, one per text.)
    Methods include some testing. And finally methods to romanize a string (romanize_string()) or an entire file
    (romanize_file())."""
    def __init__(self, data_dir: Path | None = None, **args):  # args: load_log, rebuild_ud_props
        self.data_dir = data_dir or self.default_data_dir()
        self.rom_rules = defaultdict(list)
        self.scripts = defaultdict(Script)
        self.dict_bool = defaultdict(bool)
        self.dict_str = defaultdict(str)
        self.dict_int = defaultdict(int)
        self.dict_num = defaultdict(lambda: None)   # values are int (most common), float, or str ("1/2")
        # num_props key: txt
        # values:  {"txt": "\u137b", "rom": "100", "value": 100, "type": "base", "mult": 1, "script": "Ethiopic"}
        self.num_props = defaultdict(dict)
        self.dict_set = defaultdict(set)
        self.fraction_connectors = {}
        self.minus_signs = {}
        self.plus_signs = {}
        self.float2fraction = {}  # caching
        gc.disable()
        self.rom_cache = {}   # key: (s, lcode) value: t
        self.rom_cache_size = 0
        self.rom_max_cache_size = args.get('cache_size', 0)
        self.cache_p = (self.rom_max_cache_size != 0)
        self.hangul_rom = {}
        self.stats = defaultdict(int)  # stats, e.g. for unprocessed numbers
        self.abugida_cache = {}  # key: (script, char_rom) value: (base_rom, base_rom_plus_abugida_vowel, modified rom)
        self.load_resource_files(data_dir, args.get('load_log', False),
                                 args.get('rebuild_ud_props', False),
                                 args.get('rebuild_num_props', False))
        gc.enable()

    @staticmethod
    def default_data_dir() -> Path:
        return Path(__file__).parent / "data"

    def reset_cache(self, cache_size: int = DEFAULT_ROM_MAX_CACHE_SIZE):
        self.rom_cache = {}
        self.rom_cache_size = 0
        self.rom_max_cache_size = cache_size
        self.cache_p = (self.rom_max_cache_size != 0)

    # noinspection SpellCheckingInspection
    def second_rom_filter(self, c: str, rom: str, name: str | None) -> Tuple[str | None, str]:
        """Much of this code will eventually move the old Perl code to generate cleaner primary data"""
        if rom and (' ' in rom):
            if name is None:
                name = self.chr_name(c)
            if "MYANMAR VOWEL SIGN KAYAH" in name:
                if m := regex.search(r'kayah\s+(\S+)\s*$', rom):
                    return m.group(1), name
            if "MENDE KIKAKUI SYLLABLE" in name:
                if m := regex.search(r'm\d+\s+(\S+)\s*$', rom):
                    return m.group(1), name
            if regex.search(r'\S\s+\S', rom):
                return c, name
        return None, name

    def load_rom_file(self, filename: str, provenance: str, file_format: str = None, load_log: bool = True):
        """Reads in and processes the 3 main romanization data files: (1) romanization-auto-table.txt
        which was automatically generated from UnicodeData.txt (2) UnicodeDataOverwrite.txt that "corrects"
        some entries in romanization-auto-table.txt and (3) romanization-table.txt which was largely manually
        created and allows complex romanization rules, some for specific languages, some for specific contexts."""
        n_entries = 0
        try:
            f = open(filename)
        except FileNotFoundError:
            sys.stderr.write(f'Cannot open file {filename}\n')
            return
        with (f):
            for line_number, line in enumerate(f, 1):
                if line.startswith('#'):
                    continue
                if regex.match(r'^\s*$', line):  # blank line
                    continue
                line = regex.sub(r'\s{2,}#.*$', '', line)
                if file_format == 'u2r':
                    t_at_end_of_syllable = None
                    u = dequote_string(slot_value_in_double_colon_del_list(line, 'u'))
                    try:
                        cp = int(u, 16)
                        s = chr(cp)
                    except ValueError:
                        continue
                    t = dequote_string(slot_value_in_double_colon_del_list(line, 'r'))
                    if name := slot_value_in_double_colon_del_list(line, 'name'):
                        self.dict_str[('name', s)] = name
                    if pic := slot_value_in_double_colon_del_list(line, 'pic'):
                        self.dict_str[('pic', s)] = pic
                    if tone_mark := slot_value_in_double_colon_del_list(line, 'tone-mark'):
                        self.dict_str[('tone-mark', s)] = tone_mark
                    if syllable_info := slot_value_in_double_colon_del_list(line, 'syllable-info'):
                        self.dict_str[('syllable-info', s)] = syllable_info
                else:
                    s = dequote_string(slot_value_in_double_colon_del_list(line, 's'))
                    t = dequote_string(slot_value_in_double_colon_del_list(line, 't'))
                    t_at_end_of_syllable = dequote_string(slot_value_in_double_colon_del_list(line,
                                                                                              't-end-of-syllable'))
                if (num_s := slot_value_in_double_colon_del_list(line, 'num')) is not None:
                    num = robust_str_to_num(num_s)
                    self.dict_num[s] = (num_s if (num is None) else num)
                is_minus_sign = has_value_in_double_colon_del_list(line, 'is-minus-sign')
                if is_minus_sign:
                    self.minus_signs[s] = True
                is_plus_sign = has_value_in_double_colon_del_list(line, 'is-plus-sign')
                if is_plus_sign:
                    self.plus_signs[s] = True
                is_decimal_point = has_value_in_double_colon_del_list(line, 'is-decimal-point')
                is_large_power = has_value_in_double_colon_del_list(line, 'is-large-power')
                fraction_connector = slot_value_in_double_colon_del_list(line, 'fraction-connector')
                if fraction_connector:
                    self.fraction_connectors[s] = True
                percentage_marker = slot_value_in_double_colon_del_list(line, 'percentage-marker')
                int_frac_connector = slot_value_in_double_colon_del_list(line, 'int-frac-connector')
                lcode_s = slot_value_in_double_colon_del_list(line, 'lcode')
                lcodes = regex.split(r'[,;]\s*', lcode_s) if lcode_s else []
                use_only_at_start_of_word = has_value_in_double_colon_del_list(line, 'use-only-at-start-of-word')
                dont_use_at_start_of_word = has_value_in_double_colon_del_list(line, 'dont-use-at-start-of-word')
                use_only_at_end_of_word = has_value_in_double_colon_del_list(line, 'use-only-at-end-of-word')
                dont_use_at_end_of_word = has_value_in_double_colon_del_list(line, 'dont-use-at-end-of-word')
                use_only_for_whole_word = has_value_in_double_colon_del_list(line, 'use-only-for-whole-word')
                num_s = slot_value_in_double_colon_del_list(line, 'num')
                num = robust_str_to_num(num_s, filename, line_number, silent=False)
                t_alt_s = slot_value_in_double_colon_del_list(line, 't-alt')
                t_alts = regex.split(r'[,;]\s*', t_alt_s) if t_alt_s else []
                t_alts = list(map(dequote_string, t_alts))
                t_mod, name2 = self.second_rom_filter(s, t, None)
                if t_mod and (t_mod != t):
                    if t != s:
                        pass  # sys.stderr.write(f"UPDATE: {s} {name2} {t} -> {t_mod}\n")
                    t = t_mod
                if s is not None:
                    for bool_key in ('is-large-power', 'is-minus-sign', 'is-plus-sign', 'is-decimal-point'):
                        bool_value = eval(bool_key.replace('-', '_'))
                        if bool_value:
                            self.dict_bool[(bool_key, s)] = True
                    if any_not_none(t, num, is_minus_sign, is_plus_sign, is_decimal_point, is_large_power,
                                    fraction_connector, percentage_marker, int_frac_connector):
                        self.register_s_prefix(s)
                        n_entries += 1
                        # if regex.match(r'[\u2800-\u28FF]', s): print("Braille", s, t)
                        restrictions = [lcodes, use_only_at_start_of_word, dont_use_at_start_of_word,
                                        use_only_at_end_of_word, dont_use_at_end_of_word, use_only_for_whole_word]
                        n_restrictions = len([restr for restr in restrictions if restr])
                        provenance2 = provenance
                        if (t is None) and (num is not None) and (provenance2 == "rom"):
                            provenance2 = "num"
                        new_rom_rule = RomRule(s=s, t=t, prov=provenance2, lcodes=lcodes, t_alts=t_alts, num=num,
                                               use_only_at_start_of_word=use_only_at_start_of_word,
                                               dont_use_at_start_of_word=dont_use_at_start_of_word,
                                               use_only_at_end_of_word=use_only_at_end_of_word,
                                               dont_use_at_end_of_word=dont_use_at_end_of_word,
                                               use_only_for_whole_word=use_only_for_whole_word,
                                               t_at_end_of_syllable=t_at_end_of_syllable,
                                               n_restr=n_restrictions,
                                               is_minus_sign=is_minus_sign,
                                               is_plus_sign=is_plus_sign,
                                               is_decimal_point=is_decimal_point,
                                               fraction_connector=fraction_connector,
                                               percentage_marker=percentage_marker,
                                               int_frac_connector=int_frac_connector,
                                               is_large_power=is_large_power)
                        old_rom_rules = self.rom_rules[s]
                        if ((len(old_rom_rules) == 1) and (old_rom_rules[0]['prov'] in ('ud', 'ow'))
                                and not (lcodes or use_only_at_start_of_word or dont_use_at_start_of_word
                                         or use_only_at_end_of_word or dont_use_at_end_of_word
                                         or use_only_for_whole_word)):
                            self.rom_rules[s] = [new_rom_rule]  # overwrite
                        else:
                            self.rom_rules[s].append(new_rom_rule)
        # Thai
        thai_cancellation_mark = '\u0E4C'
        # cancellation applies to preceding letter incl. any vowel modifier letter
        # noinspection SpellCheckingInspection (e.g. ศักดิ์สิทธิ์ -> saksit)
        for cp in range(0x0E01, 0x0E4C):   # Thai
            c = chr(cp)
            s = c + thai_cancellation_mark
            new_rom_rule = RomRule(s=s, t='', prov='auto cancel letter')
            if not self.rom_rules[s]:
                self.rom_rules[s] = [new_rom_rule]
                self.register_s_prefix(s)
        thai_consonants = list(map(chr, range(0x0E01, 0x0E2F)))
        thai_vowel_modifiers = ['\u0E31', '\u0E47'] + list(map(chr, range(0x0E33, 0x0E3B)))
        for c1 in thai_consonants:
            for v in thai_vowel_modifiers:
                s = c1 + v + thai_cancellation_mark
                new_rom_rule = RomRule(s=s, t='', prov='auto cancel syllable')
                if not self.rom_rules[s]:
                    self.rom_rules[s] = [new_rom_rule]
                    self.register_s_prefix(s)
        if load_log:
            sys.stderr.write(f'Loaded {n_entries} from {filename}\n')

    def load_script_file(self, filename: str, load_log: bool = True):
        """Reads in (typically from Scripts.txt) information about various scripts such as Devanagari,
        incl. information such as the default abugida vowel letter (e.g. "a")."""
        n_entries, max_n_script_name_components = 0, 0
        try:
            f = open(filename)
        except FileNotFoundError:
            sys.stderr.write(f'Cannot open file {filename}\n')
            return
        with f:
            for line_number, line in enumerate(f, 1):
                if line.startswith('#'):
                    continue
                if regex.match(r'^\s*$', line):  # blank line
                    continue
                line = regex.sub(r'\s{2,}#.*$', '', line)
                if script_name := slot_value_in_double_colon_del_list(line, 'script-name'):
                    lc_script_name = script_name.lower()
                    if lc_script_name in self.scripts:
                        sys.stderr.write(f'** Ignoring duplicate script "{script_name}" '
                                         f'in line {line_number} of {filename}\n')
                    else:
                        n_entries += 1
                        direction = slot_value_in_double_colon_del_list(line, 'direction')
                        abugida_default_vowel_s = slot_value_in_double_colon_del_list(line,
                                                                                      'abugida-default-vowel')
                        abugida_default_vowels = regex.split(r'[,;]\s*', abugida_default_vowel_s) \
                            if abugida_default_vowel_s else []
                        alt_script_name_s = slot_value_in_double_colon_del_list(line, 'alt-script-name')
                        alt_script_names = regex.split(r'[,;]\s*', alt_script_name_s) if alt_script_name_s else []
                        language_s = slot_value_in_double_colon_del_list(line, 'language')
                        languages = regex.split(r'[,;]\s*', language_s) if language_s else []
                        new_script = Script(script_name=script_name, alt_script_names=alt_script_names,
                                            languages=languages, direction=direction,
                                            abugida_default_vowels=abugida_default_vowels)
                        self.scripts[lc_script_name] = new_script
                        for language in languages:
                            self.dict_set[('scripts', language)].add(script_name)
                        for alt_script_name in alt_script_names:
                            lc_alt_script_name = alt_script_name.lower()
                            if lc_alt_script_name in self.scripts:
                                sys.stderr.write(f'** Ignoring duplicate alternative script name "{script_name}" '
                                                 f'in line {line_number} of {filename}\n')
                            else:
                                self.scripts[lc_alt_script_name] = new_script
                    n_script_name_components = len(script_name.split())
                    if n_script_name_components > max_n_script_name_components:
                        max_n_script_name_components = n_script_name_components
        if max_n_script_name_components:
            self.dict_int['max_n_script_name_components'] = max_n_script_name_components
        if load_log:
            sys.stderr.write(f'Loaded {n_entries} script descriptions from {filename}'
                             f' (max_n_scripts_name_components: {max_n_script_name_components})\n')

    def extract_script_name(self, script_name_plus: str, full_char_name: str = None) -> str | None:
        """Using info from Scripts.txt, this script selects the script name from a Unicode,
        e.g. given "OLD HUNGARIAN CAPITAL LETTER A", extract "Old Hungarian"."""
        if full_char_name and script_name_plus == full_char_name:
            return None
        while script_name_plus:
            if script_name_plus.lower() in self.scripts:
                if script := self.scripts[script_name_plus.lower()]:
                    if script_name := script['script-name']:
                        return script_name
            script_name_plus = regex.sub(r'\s*\S*\s*$', '', script_name_plus)
        return None

    def load_unicode_data_props(self, filename: str, load_log: bool = True):
        """Loads Unicode derived data from (1) UnicodeDataProps.txt, (2) UnicodeDataPropsHangul.txt
        and UnicodeDataPropsCJK.txt with a list of valid script-specific characters."""
        n_script, n_script_char, n_script_vowel_sign, n_script_medial_consonant_sign, n_script_virama = 0, 0, 0, 0, 0
        try:
            f = open(filename)
        except FileNotFoundError:
            sys.stderr.write(f'Cannot open file {filename}\n')
            return
        with f:
            for line_number, line in enumerate(f, 1):
                if line.startswith('#'):
                    continue
                if regex.match(r'^\s*$', line):  # blank line
                    continue
                line = regex.sub(r'\s{2,}#.*$', '', line)
                if script_name := slot_value_in_double_colon_del_list(line, 'script-name'):
                    n_script += 1
                    for char in slot_value_in_double_colon_del_list(line, 'char', []):
                        self.dict_str[('script', char)] = script_name
                        n_script_char += 1
                    for char in slot_value_in_double_colon_del_list(line, 'numeral', []):
                        self.dict_str[('script', char)] = script_name
                        n_script_char += 1
                    for char in slot_value_in_double_colon_del_list(line, 'vowel-sign', []):
                        self.dict_bool[('is-vowel-sign', char)] = True
                        n_script_vowel_sign += 1
                    for char in slot_value_in_double_colon_del_list(line, 'medial-consonant-sign', []):
                        self.dict_bool[('is-medial-consonant-sign', char)] = True
                        n_script_medial_consonant_sign += 1
                    for char in slot_value_in_double_colon_del_list(line, 'sign-virama', []):
                        self.dict_bool[('is-virama', char)] = True
                        n_script_virama += 1
        if load_log:
            sys.stderr.write(f'Loaded from {filename} mappings of {n_script_char:,d} characters '
                             f'to {n_script} script{"" if n_script == 1 else "s"}')
            if n_script_vowel_sign or n_script_virama or n_script_medial_consonant_sign:
                sys.stderr.write(f', with a total of {n_script_vowel_sign} vowel signs, '
                                 f'{n_script_medial_consonant_sign} medial consonant signs '
                                 f'and {n_script_virama} viramas')
            sys.stderr.write('.\n')

    def load_num_props(self, filename: str, load_log: bool = True):
        """Loads Unicode derived data from (1) UnicodeDataProps.txt, (2) UnicodeDataPropsHangul.txt
        and UnicodeDataPropsCJK.txt with a list of valid script-specific characters."""
        n_entries = 0
        try:
            f = open(filename)
        except FileNotFoundError:
            sys.stderr.write(f'Cannot open file {filename}\n')
            return
        with f:
            for line_number, line in enumerate(f, 1):
                if line.startswith('#'):
                    continue
                if regex.match(r'^\s*$', line):  # blank line
                    continue
                d = json.loads(line)
                if isinstance(d, dict):
                    if txt := d.get('txt'):
                        self.num_props[txt] = d
                        n_entries += 1
                    else:
                        sys.stderr.write(f'Missing txt in l.{line_number} in file {filename}: {line.strip()}\n')
                    for bool_key in ('is-large-power',):
                        if d.get(bool_key):
                            self.dict_bool[(bool_key, txt)] = True
                else:
                    sys.stderr.write(f'json in l.{line_number} in file {filename} not a dict: {line.strip()}\n')
        if load_log:
            sys.stderr.write(f'Loaded {n_entries} entries from {filename}\n')

    @staticmethod
    def de_accent_pinyin(s: str) -> str:
        """De-accents a string from "liú" to "liu" and "ü" to "u" (to help process file Chinese_to_Pinyin.txt)."""
        result = ''
        for char in s:
            if decomp := ud.decomposition(char).split():
                try:
                    decomp_chars = [chr(int(x, 16)) for x in decomp]
                    letters = [x for x in decomp_chars if ud.category(x).startswith('L')]
                except ValueError:
                    sys.stderr.write(f'Cannot decode {decomp}\n')
                    continue
                if len(letters) == 1:
                    result += letters[0]
                else:
                    sys.stderr.write(f'Cannot decode {decomp} (expected 1 letter)\n')
            else:
                result += char
        result = result.replace('ü', 'u')
        return result

    def register_s_prefix(self, s: str):
        for prefix_len in range(1, len(s) + 1):
            self.dict_bool[('s-prefix', s[:prefix_len])] = True

    def load_chinese_pinyin_file(self, filename: str, load_log: bool = True):
        """Loads file Chinese_to_Pinyin.txt which maps Chinese characters to their Latin form."""
        n_entries = 0
        try:
            f = open(filename)
        except FileNotFoundError:
            sys.stderr.write(f'Cannot open file {filename}\n')
            return
        with f:
            for line_number, line in enumerate(f, 1):
                if line.startswith('#'):
                    continue
                if regex.match(r'^\s*$', line):  # blank line
                    continue
                try:
                    chinese, pinyin = line.rstrip().split()
                    rom = self.de_accent_pinyin(pinyin)
                except ValueError:
                    sys.stderr.write(f'Cannot process line {line_number} in file {filename}: {line}')
                else:
                    s = chinese
                    new_rom_rule = RomRule(s=s, t=rom, prov='rom pinyin', lcodes=[])
                    self.rom_rules[chinese].append(new_rom_rule)
                    self.register_s_prefix(s)
                    n_entries += 1
        if load_log:
            sys.stderr.write(f'Loaded {n_entries} script descriptions from {filename}\n')

    @staticmethod
    def add_char_to_rebuild_unicode_data_dict(d: dict, script_name: str, prop_class: str, char: str):
        d['script-names'].add(script_name)
        key = (script_name, prop_class)
        if key in d:
            d[key].append(char)
        else:
            d[key] = [char]

    def rebuild_unicode_data_props(self, out_filename: str, cjk: str = None, hangul: str = None):
        """This functions rebuilds UnicodeDataProps*.txt This might be useful when a new UnicodeData.txt
        version is released, or additional information is extracted from Unicode to UnicodeDataProps.txt
        Regular users normally never have to call this function."""
        d = {'script-names': set()}
        vowel_s = ''
        n_script_refs = 0
        codepoint = -1
        prop_classes = {'char'}
        while codepoint < 0xF0000:
            codepoint += 1
            c = chr(codepoint)
            if not (char_name := self.chr_name(c)):
                continue
            # noinspection SpellCheckingInspection
            for prop_name_comp2 in ('VOWEL SIGN',
                                    ('MEDIAL CONSONANT SIGN', 'CONSONANT SIGN MEDIAL', 'CONSONANT SIGN SHAN MEDIAL',
                                     'CONSONANT SIGN MON MEDIAL'),
                                    ('SIGN VIRAMA', 'SIGN ASAT', 'AL-LAKUNA', 'SIGN COENG', 'SIGN PAMAAEH',
                                     'CHARACTER PHINTHU'),
                                    ('NUMERAL', 'NUMBER', 'DIGIT', 'FRACTION')):
                if prop_name_comp2 and isinstance(prop_name_comp2, tuple):
                    prop_list = prop_name_comp2
                else:
                    prop_list = (prop_name_comp2,)
                for prop_name_comp in prop_list:
                    prop_class = prop_list[0].lower().replace(' ', '-')
                    if prop_class not in prop_classes:
                        prop_classes.add(prop_class)
                    script_name_cand = regex.sub(fr'\s+{prop_name_comp}\b.*$', '', char_name)
                    if script_name := self.extract_script_name(script_name_cand, char_name):
                        self.add_char_to_rebuild_unicode_data_dict(d, script_name, prop_class, c)
            script_name_cand = regex.sub(r'\s+(CONSONANT|LETTER|LIGATURE|SIGN|SYLLABLE|SYLLABICS|VOWEL|'
                                         r'IDEOGRAPH|HIEROGLYPH|POINT|ACCENT|CHARACTER|TIPPI|ADDAK|IRI|URA|'
                                         r'SYMBOL GENITIVE|SYMBOL COMPLETED|SYMBOL LOCATIVE|SYMBOL AFOREMENTIONED|'
                                         r'AU LENGTH MARK)\b.*$', '',
                                         char_name)
            if script_name := self.extract_script_name(script_name_cand, char_name):
                self.add_char_to_rebuild_unicode_data_dict(d, script_name, 'char', c)
                n_script_refs += 1
            rom = self.romanize_string(c)
            # noinspection SpellCheckingInspection
            if regex.match(r'^[aeiou]*[aeiouy]$', rom, regex.IGNORECASE):
                vowel_s += c

        # print(sorted(d['script-names']))
        prop_classes = sorted(prop_classes)
        out_filenames = [x for x in [out_filename, cjk, hangul] if x]
        cjk2 = cjk if cjk else out_filename
        hangul2 = hangul if hangul else out_filename
        for out_file in out_filenames:
            try:
                f_out = open(out_file, 'w')
            except OSError:
                sys.stderr.write(f'Cannot write to file {out_file}\n')
                continue
            with f_out:
                for script_name in sorted(d['script-names']):
                    if script_name == 'CJK':
                        if out_file != cjk2:
                            continue
                    elif script_name == 'Hangul':
                        if out_file != hangul2:
                            continue
                    else:
                        if out_file != out_filename:
                            continue
                    prop_components = [f"::script-name {script_name}"]
                    for prop_class in prop_classes:
                        key = (script_name, prop_class)
                        if key in d:
                            if chars := ''.join(d[key]):
                                if prop_class in ('char',):
                                    prop_components.append(f"::n-{prop_class} {len(chars)}")
                                prop_components.append(f"::{prop_class} {chars}")
                    f_out.write(f"{' '.join(prop_components)}\n")
                if (out_file == out_filename) and vowel_s:
                    f_out.write(f'::vowels {vowel_s}\n')
        sys.stderr.write(f"Rebuilt {out_filenames} with {n_script_refs} characters "
                         f"for {len(d['script-names'])} scripts.\n")

    def rebuild_num_props(self, out_filename: str, err_filename: str):
        n_out, n_err = 0, 0
        with open(out_filename, 'w') as f_out, open(err_filename, 'w') as f_err:
            codepoint = -1
            while codepoint < 0xF0000:
                codepoint += 1
                char = chr(codepoint)
                num = first_non_none(ud_numeric(char),  # robust ud.numeric
                                     self.num_value(char))  # uroman table includes extra num values, e.g. for Egyptian
                if num is None:
                    continue
                result_dict = {}
                orig_txt = char
                value: int | float | None = None  # non-fraction-value(3 1/2) = 3
                fraction: Fraction | None = None  # fraction(3 1/2) = Fraction(1, 2)
                num_base = None  # num_base(500) = 100
                base_multiplier = None  # base_multiplier(500) = 5
                script = None
                is_large_power = self.dict_bool[('is-large-power', char)]
                # num_base is typically a power of 10: 1, 10, 100, 1000, 10000, 100000, 1000000, ...
                # exceptions might include 12 for the 'dozen' in popular English 'two dozen and one' (2*12+1=25)
                # exceptions might include 20 for the 'score' in archaic English 'four score and seven' (4*20+7=87)
                # noinspection SpellCheckingInspection   exceptions might include 20 for the 'vingt'
                # noinspection SpellCheckingInspection   as in standard French 'quatre-vingt-treize' (4*20+13=93)
                if script_name := self.chr_script_name(char):
                    script = script_name
                elif char in '0123456789':
                    script = 'ascii-digit'
                name = self.chr_name(char)
                exclude_from_number_processing = False
                for scrypt_type in ('SUPERSCRIPT', 'SUBSCRIPT',
                                    'CIRCLED', 'PARENTHESIZED', 'SEGMENTED', 'MATHEMATICAL', 'ROMAN NUMERAL',
                                    'FULL STOP', 'COMMA'):
                    if scrypt_type in name:
                        script = '*' + scrypt_type.lower().replace(' ', '-')
                        exclude_from_number_processing = True
                        break
                for scrypt_type in ('VULGAR FRACTION',):
                    if scrypt_type in name:
                        script = scrypt_type.lower().replace(' ', '-')
                        break
                if exclude_from_number_processing:
                    continue
                if isinstance(num, int):
                    value = num
                    if 0 <= num <= 9:
                        num_base = 1
                        base_multiplier = num
                        if "DIGIT" in name:
                            num_type = 'digit'
                        else:
                            # Chinese numbers 零 (0), 一 (1), ... 九 (9) have numeric values,
                            # but are NOT (full) digits
                            num_type = 'digit-like'
                    elif m := regex.match(r'([0-9]+?)(0*)$', str(num)):
                        base_multiplier = int(m.group(1))  # non_base_value(500) = 5
                        num_base = int('1' + m.group(2))
                        num_type = 'base' if base_multiplier == 1 else 'multi'
                    else:
                        num_type = 'other-int'  # Do such cases exist?
                elif ("FRACTION" in name) and (fraction := fraction_char2fraction(char, num, self)):
                    fraction = fraction
                    num_type = 'fraction'
                else:
                    num_type = 'other-num'  # Do such cases exist? Yes. Bengali currency numerators, ...
                value_s = '' if value is None else str(value)
                fraction_s = '' if fraction is None else f'{fraction.numerator}/{fraction.denominator}'
                fraction_list = None if fraction is None else [fraction.numerator, fraction.denominator]
                delimiter_s = ' ' if value_s and fraction_s else ''
                rom = (value_s + delimiter_s + fraction_s) or orig_txt
                add_non_none_to_dict(result_dict, 'txt', orig_txt)
                add_non_none_to_dict(result_dict, 'rom', rom)
                add_non_none_to_dict(result_dict, 'value', value)
                add_non_none_to_dict(result_dict, 'fraction', fraction_list)
                add_non_none_to_dict(result_dict, 'type', num_type)
                if is_large_power:
                    result_dict['is-large-power'] = True
                add_non_none_to_dict(result_dict, 'base', num_base)
                add_non_none_to_dict(result_dict, 'mult', base_multiplier)
                add_non_none_to_dict(result_dict, 'script', script)
                if num_type.startswith('other'):
                    add_non_none_to_dict(result_dict, 'name', name)
                    f_err.write(json.dumps(result_dict) + '\n')
                    n_err += 1
                else:
                    if not script:
                        add_non_none_to_dict(result_dict, 'name', name)
                    f_out.write(json.dumps(result_dict) + '\n')
                    n_out += 1
        sys.stderr.write(f'Processed {codepoint} codepoints,\n  wrote {n_out} lines to {out_filename}\n'
                         f'    and {n_err} lines to {err_filename}\n')

    def load_resource_files(self, data_dir: Path, load_log: bool = False,
                            rebuild_ud_props: bool = False, rebuild_num_props: bool = False):
        """Loads all resource files needed for romanization."""
        data_dir = data_dir
        if not isinstance(data_dir, Path):
            sys.stderr.write(f'Error: data_dir is of {type(data_dir)}, not a Path.\n'
                             f'       Cannot load any resource files.\n')
            return
        self.load_rom_file(os.path.join(data_dir, "romanization-auto-table.txt"),
                           'ud', file_format='rom', load_log=load_log)
        self.load_rom_file(os.path.join(data_dir, "UnicodeDataOverwrite.txt"),
                           'ow', file_format='u2r', load_log=load_log)
        self.load_rom_file(os.path.join(data_dir, "romanization-table.txt"),
                           'man', file_format='rom', load_log=load_log)
        self.load_chinese_pinyin_file(os.path.join(data_dir, "Chinese_to_Pinyin.txt"), load_log=load_log)
        self.load_script_file(os.path.join(data_dir, "Scripts.txt"), load_log=load_log)
        self.load_num_props(os.path.join(data_dir, "NumProps.jsonl"), load_log=load_log)
        for base_file in ("UnicodeDataProps.txt", "UnicodeDataPropsCJK.txt", "UnicodeDataPropsHangul.txt"):
            self.load_unicode_data_props(os.path.join(data_dir, base_file), load_log=load_log)
        if rebuild_ud_props:
            self.rebuild_unicode_data_props(os.path.join(data_dir, "UnicodeDataProps.txt"),
                                            cjk=os.path.join(data_dir, "UnicodeDataPropsCJK.txt"),
                                            hangul=os.path.join(data_dir, "UnicodeDataPropsHangul.txt"))
        if rebuild_num_props:
            self.rebuild_num_props(os.path.join(data_dir, "NumProps.jsonl"),
                                   os.path.join(data_dir, "NumPropsRejects.jsonl"))

    def unicode_hangul_romanization(self, s: str, pass_through_p: bool = False):
        """Special algorithmic solution to convert (Korean) Hangul characters to the Latin alphabet."""
        if cached_rom := self.hangul_rom.get(s, None):
            return cached_rom
        leads = "g gg n d dd r m b bb s ss - j jj c k t p h".split()
        vowels = "a ae ya yae eo e yeo ye o wa wai oe yo u weo we wi yu eu yi i".split()
        tails = "- g gg gs n nj nh d l lg lm lb ls lt lp lh m b bs s ss ng j c k t p h".split()
        result = ""
        for c in s:
            cp = ord(c)
            if 0xAC00 <= cp <= 0xD7A3:
                code = cp - 0xAC00
                lead_index = int(code / (28 * 21))
                vowel_index = int(code / 28) % 21
                tail_index = code % 28
                rom = leads[lead_index] + vowels[vowel_index] + tails[tail_index]
                rom = rom.replace('-', '')
                self.hangul_rom[c] = rom
                result += rom
            elif pass_through_p:
                result += c
        return result

    @staticmethod
    def char_is_nonspacing_mark(s) -> bool:
        """ Checks whether a character is a nonspacing mark, e.g. combining accents, points, vowel signs"""
        return (len(s) == 1) and (ud.category(s) == 'Mn')

    @staticmethod
    def char_is_format_char(s) -> bool:
        """ Checks whether a character is a formatting character, e.g. a zero-with joiner/non-joiner"""
        return (len(s) == 1) and (ud.category(s) == 'Cf')

    @staticmethod
    def char_is_space_separator(s) -> bool:
        """ Checks whether a character is a space,
            e.g. ' ', non-breakable space, en space, ideographic (Chinese) space, Ogham space mark
            but excluding \t, \r, \n"""
        return (len(s) == 1) and (ud.category(s) == 'Zs')

    def chr_name(self, char: str) -> str:
        try:
            return ud.name(char)
        except (ValueError, TypeError):
            if name := self.dict_str[('name', char)]:
                return name
        return ''

    def num_value(self, s: str) -> int | float | Fraction | None:
        """rom_rules include numeric values beyond UnicodeData.txt, e.g. for Egyptian numerals"""
        for rom_rule in self.rom_rules[s]:
            if (num := rom_rule['num']) is not None:
                return num
        return None

    def rom_rule_value(self, s: str, key: str):
        for rom_rule in self.rom_rules[s]:
            if (value := rom_rule.get(key)) is not None:
                return value
        return None

    def unicode_float2fraction(self, num: float, precision: float = 0.000001) -> Tuple[int, int] | None:
        """only for common unicode fractions"""
        if cached_value := self.float2fraction.get(num, None):
            return cached_value
        for numerator in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
            for denominator in (2, 3, 4, 5, 6, 8, 12, 16, 20, 32, 40, 64, 80, 160, 320):
                if abs(numerator / denominator - num) < precision:
                    result = numerator, denominator
                    self.float2fraction[num] = result
                    return result
        return None

    def chr_script_name(self, char: str) -> str:
        """For letters, diacritics, numerals etc."""
        return self.dict_str[('script', char)]

    def test_output_of_selected_scripts_and_rom_rules(self):
        """Low level test function that checks and displays romanization information."""
        output = ''
        for s in ("Oriya", "Chinese"):
            d = self.scripts[s.lower()]
            output += f'SCRIPT {s} {d}\n'
        for s in ('ƿ', 'β', 'и', 'μπ', '⠹', '亿', 'ちょ', 'и', '𓍧', '正', '分之', 'ऽ', 'ศ', 'ด์', 'ढ़', 'ड़'):
            d = self.rom_rules[s]
            output += f'DICT {s} {d}\n'
        for s in ('ƿ', 'β', 'न', 'ु'):
            output += f'SCRIPT-NAME {s} {self.chr_script_name(s)}\n'
        for s in ('万', '\uF8F7', '\U00013368', '\U0001308B', '\u0E48', '\u0E40'):
            name = self.chr_name(s)
            num = self.dict_num[s]
            pic = self.dict_str[('pic', s)]
            tone_mark = self.dict_str[('tone-mark', s)]
            syllable_info = self.dict_str[('syllable-info', s)]
            is_large_power = self.dict_bool[('is-large-power', s)]
            output += f'PROPS {s}'
            if name:
                output += f'  name: {name}'
            if num:
                output += f'  num: {num} ({type(num).__name__})'
            if pic:
                output += f'  pic: {pic}'
            if tone_mark:
                output += f'  tone-mark: {tone_mark}'
            if syllable_info:
                output += f'  syllable-info: {syllable_info}'
            if is_large_power:
                output += f'  is-large-power: {is_large_power}'
            output += '\n'
        mayan12 = '\U0001D2EC'
        egyptian600 = '𓍧'
        runic90 = '𐍁'
        klingon2 = '\uF8F2'
        for offset, c in enumerate(f'9九万萬百፲፱፻፸¾0²₂AⅫ⑫൵{runic90}{mayan12}{egyptian600}{klingon2}'):
            output += f'NUM-EDGE: {NumEdge(offset, offset+1, c, self)}\n'
        for s in ('\u00bc', '\u0968'):
            output += f'NUM-PROPS: {self.num_props[s]}\n'
        print(output)

    def test_romanization(self, **args):
        """A few full cases of romanization testing."""
        # noinspection SpellCheckingInspection
        tests = [('ألاسكا', None), ('यह एक अच्छा अनुवाद है.', 'hin'), ('ちょっとまってください', 'kor'),
                 ('Μπανγκαλόρ', 'ell'), ('Зеленський', 'ukr'), ('കേരളം', 'mal')]
        for test in tests:
            s = test[0]
            lcode = test[1] if len(test) >= 2 else None
            rom = self.romanize_string(s, lcode=lcode, **args)
            sys.stderr.write(f'ROM {s} -> {rom}\n')
        n_alerts = 0
        codepoint = -1
        while codepoint < 0xF0000:
            codepoint += 1
            c = chr(codepoint)
            rom = self.romanize_string(c)
            if regex.search(r'\s', rom) and regex.search(r'\S', rom):
                name = self.chr_name(c)
                sys.stderr.write(f'U+{codepoint:04X} {c} {name}  {rom}\n')
                n_alerts += 1
        sys.stderr.write(f'{n_alerts} alerts for roms with spaces\n')

    def romanize_file(self, input_filename: str | None = None, output_filename: str | None = None,
                      lcode: str | None = None, direct_input: List[str] = None, **args):
        """Script to apply romanization to an entire file. Input and output files needed.
        Language code (lcode) recommended."""
        f_in_to_be_closed, f_out_to_be_closed = False, False
        if direct_input and (input_filename is None):
            f_in = direct_input  # list of lines
        elif isinstance(input_filename, str):
            try:
                f_in = open(input_filename)
                f_in_to_be_closed = True
            except OSError:
                sys.stderr.write(f'Error in romanize_file: Cannot open file {input_filename}\n')
                f_in = None
        elif input_filename is None:
            f_in = sys.stdin
        else:
            sys.stderr.write(f"Error in romanize_file: argument 'input_filename' {input_filename} "
                             f"is of wrong type: {type(input_filename)} (should be str)\n")
            f_in = None
        if isinstance(output_filename, str):
            try:
                f_out = open(str(output_filename), 'w')
                f_out_to_be_closed = True
            except OSError:
                sys.stderr.write(f'Error in romanize_file: Cannot write to file {output_filename}\n')
                f_out = None
        elif output_filename is None:
            f_out = sys.stdout
        else:
            sys.stderr.write(f"Error in romanize_file: argument 'output_filename' {output_filename} "
                             f"is of wrong type: {type(output_filename)} (should be str)\n")
            f_out = None
        if f_in and f_out:
            max_lines = args.get('max_lines')
            progress_dots_output = False
            for line_number, line in enumerate(f_in, 1):
                if m := regex.match(r'(::lcode\s+)([a-z]{3})(\s+)(.*)$', line):
                    lcode_kw, lcode2, space, snt = m.group(1, 2, 3, 4)
                    rom_result = self.romanize_string(snt, lcode2 or lcode, **args)
                    if args.get('rom_format', RomFormat.STR) == RomFormat.STR:
                        lcode_prefix = f"{lcode_kw}{lcode2}{space}"
                        f_out.write(lcode_prefix + rom_result + '\n')
                    else:
                        lcode_prefix = f'[0, 0, "", "lcode: {lcode2}"]'  # meta edge with lcode info
                        prefixed_edges = [lcode_prefix] + self.romanize_string(snt, lcode2 or lcode, **args)
                        f_out.write(Edge.json_str(prefixed_edges) + '\n')
                else:
                    f_out.write(Edge.json_str(self.romanize_string(line.rstrip('\n'), lcode, **args)) + '\n')
                if not args.get('silent'):
                    if line_number % 100 == 0:
                        if line_number % 1000 == 0:
                            sys.stderr.write(str(line_number))
                        else:
                            sys.stderr.write('.')
                        progress_dots_output = True
                        sys.stderr.flush()
                        gc.collect()
                if max_lines and line_number >= max_lines:
                    break
            if progress_dots_output:
                sys.stderr.write('\n')
                sys.stderr.flush()
        if f_in_to_be_closed:
            f_in.close()
        if f_out_to_be_closed:
            f_out.close()

    @staticmethod
    def apply_any_offset_to_cached_rom_result(cached_rom_result: str | List[Edge], offset: int = 0) \
            -> str | List[Edge]:
        if isinstance(cached_rom_result, str):
            return cached_rom_result
        elif offset == 0:
            return cached_rom_result
        else:
            return [Edge(edge.start + offset, edge.end + offset, edge.txt, edge.type) for edge in cached_rom_result]

    @staticmethod
    def decode_unicode_escapes(s: str) -> str:
        if regex.search(r'\\[xuU][0-9A-Fa-f]{2}', s):
            result = ''
            rest = s
            while m := regex.match(r'(.*?)(\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})(.*)$', rest):
                pre, core, rest = m.group(1, 2, 3)
                cp = int(core[2:], 16)
                # escape only for non-ASCII, specifically not for \x22, \x25 (quote, apostrophe)
                if cp > 0x80:
                    result += pre + chr(cp)
                else:
                    result += pre + core
            result += rest + ('\n' if s.endswith('\n') else '')
            return result
        else:
            return s

    def romanize_string_core(self, s: str, lcode: str | None, rom_format: RomFormat, offset: int = 0, **args) \
            -> str | List[Edge]:
        """Script to support token-by-token romanization with caching for higher speed."""
        if self.cache_p:
            cached_rom = self.rom_cache.get((s, lcode, rom_format))
            if cached_rom is not None:
                return self.apply_any_offset_to_cached_rom_result(cached_rom, offset)
        lat = Lattice(s, uroman=self, lcode=lcode)
        lat.pick_tibetan_vowel_edge(**args)
        lat.prep_braille(**args)
        lat.add_romanization(**args)
        lat.add_numbers(self, **args)
        lat.add_braille_numbers(**args)
        lat.add_rom_fall_back_singles(**args)
        if rom_format == RomFormat.LATTICE:
            all_edges = lat.all_edges(0, len(s))
            lat.add_alternatives(all_edges)
            if self.rom_cache_size < self.rom_max_cache_size:
                self.rom_cache[(s, lcode, rom_format)] = all_edges
                self.rom_cache_size += 1
            result = self.apply_any_offset_to_cached_rom_result(all_edges, offset)
        else:
            best_edges = lat.best_rom_edge_path(0, len(s))
            if rom_format in (RomFormat.EDGES, RomFormat.ALTS):
                if rom_format == RomFormat.ALTS:
                    lat.add_alternatives(best_edges)
                if self.rom_cache_size < self.rom_max_cache_size:
                    self.rom_cache[(s, lcode, rom_format)] = best_edges
                    self.rom_cache_size += 1
                result = self.apply_any_offset_to_cached_rom_result(best_edges, offset)
            else:
                rom = lat.edge_path_to_surf(best_edges)
                del lat
                if self.rom_cache_size < self.rom_max_cache_size:
                    self.rom_cache[(s, lcode, rom_format)] = rom
                    self.rom_cache_size += 1
                result = rom
        return result

    def romanize_string(self, s: str, lcode: str | None = None, rom_format: RomFormat = RomFormat.STR, **args) \
            -> str | List[Edge]:
        """Main entry point for romanizing a string. Recommended argument: lcode (language code).
        recursive only used for development.
        Method returns a string or a list of edges (with start and end offsets)."""
        lcode = lcode or args.get('lcode', None)
        # print('rom::', s, 'lcode:', lcode, 'print-lattice:', print_lattice_p)

        if args.get('decode_unicode'):
            s = self.decode_unicode_escapes(s)
        # with caching (for string format output only for now)
        if self.cache_p:
            rest, offset = s, 0
            result = '' if rom_format == RomFormat.STR else []
            while m3 := regex.match(r'(.*?)([.,; ]*[ 。་][.,; ]*)(.*)$', rest):
                pre, delimiter, rest = m3.group(1, 2, 3)
                result += self.romanize_string_core(pre, lcode, rom_format, offset, **args)
                offset += len(pre)
                result += self.romanize_string_core(delimiter, lcode, rom_format, offset, **args)
                offset += len(delimiter)
            result += self.romanize_string_core(rest, lcode, rom_format, offset, **args)
            return result
        else:
            return self.romanize_string_core(s, lcode, rom_format, 0, **args)


class Edge:
    """This class defines edges that span part of a sentence with a specific romanization.
    There might be multiple edges for a given span. The edges in turn are part of the
    romanization lattice."""
    def __init__(self, start: int, end: int, s: str, annotation: str = None):
        self.start = start
        self.end = end
        self.txt = s
        self.type = annotation

    def __str__(self):
        return f'[{self.start}-{self.end}] {self.txt} ({self.type})'

    def __repr__(self):
        return str(self)

    def json(self) -> str:  # start - end - text - annotation
        return json.dumps([self.start, self.end, self.txt, self.type])

    @staticmethod
    def json_str(rom_result: List[Edge] | str) -> str:
        if isinstance(rom_result, str):
            return rom_result
        else:
            result = '['
            for edge in rom_result:
                if isinstance(edge, Edge):
                    result += edge.json()
                else:
                    result += str(edge)
            result += ']'
            return result


class NumEdge(Edge):
    def __init__(self, start: int, end: int, s: str, uroman: Uroman | None, active: bool = False):
        """For NumEdge, the s argument is in original language (not yet romanized)."""
        # For speed, much of this processing should at some point be cached in data files.
        Edge.__init__(self, start, end, s)
        self.orig_txt, self.txt = s, s
        self.value, self.fraction, self.num_base, self.base_multiplier = None, None, None, None
        self.type, self.script, self.is_large_power, self.active = None, None, False, active
        self.n_decimals = None
        self.value_s = None     # precision for 3.14159265358979323846264338327950288419716939937510582097494
        if start+1 == end:
            char = s[0]
            if d := uroman.num_props.get(char):
                self.active = True
                self.value = d.get('value')
                fraction_list = d.get('fraction')
                self.fraction = Fraction(fraction_list[0], fraction_list[1]) if fraction_list else None
                self.num_base = d.get('base')
                self.base_multiplier = d.get('mult')
                self.type = d.get('type')
                self.script = d.get('script')
                self.is_large_power = d.get('is-large-power')
                self.update()

    def update(self,
               value: int | float | None = None,
               value_s: str | None = None,
               fraction: Fraction | None = None,
               n_decimals: int | None = None,
               num_base: int | None = None,
               base_multiplier: int | float | None = None,
               script: str | None = None,
               e_type: str | None = None,
               orig_txt: str | None = None) -> str:
        self.value = first_non_none(value, self.value)
        self.value_s = first_non_none(value_s, self.value_s)
        self.fraction = first_non_none(fraction, self.fraction)
        self.n_decimals = first_non_none(n_decimals, self.n_decimals)
        self.num_base = first_non_none(num_base, self.num_base)
        self.base_multiplier = first_non_none(base_multiplier, self.base_multiplier)
        self.script = first_non_none(script, self.script)
        self.type = first_non_none(e_type, self.type)
        self.orig_txt = first_non_none(orig_txt, self.orig_txt)
        if self.value_s is not None:
            value_s = self.value_s
        elif self.value is None:
            value_s = ''
        elif isinstance(self.value, float) and (self.n_decimals is not None):
            value_s = first_non_none(self.value_s, f'{self.value:0.{self.n_decimals}f}')
        else:
            value_s = str(self.value)
        fraction_s = '' if self.fraction is None else f'{self.fraction.numerator}/{self.fraction.denominator}'
        delimiter_s = ' ' if value_s and fraction_s else ''
        self.txt = (value_s + delimiter_s + fraction_s) or self.orig_txt
        return self.txt

    def __str__(self):
        if self.num_base is not None:
            if self.base_multiplier is not None:
                b_clause = f'{self.base_multiplier}*{self.num_base}'
            else:
                b_clause = str(self.num_base)
        else:
            b_clause = None
        return (('' if self.active else ' *')
                + f'[{self.start}-{self.end}] {self.orig_txt} R:{self.txt} T:{self.type}'
                + (' LP' if self.is_large_power else '')
                + (f' B:{b_clause}' if (b_clause is not None) else '')
                + (f' V:{self.value}' if ((self.value is not None) and (str(self.value) != self.txt)) else '')
                + (f' VS:{self.value_s}' if ((self.value_s is not None) and (self.value_s != self.txt)) else '')
                + (f' F:.{self.n_decimals}f' if self.n_decimals else f'')
                + (f' S:{self.script}' if self.script else ''))


class Lattice:
    """Lattice for a specific romanization instance. Has edges."""
    def __init__(self, s: str, uroman: Uroman, lcode: str = None):
        self.s = s
        self.lcode = lcode
        self.lattice = defaultdict(set)
        self.max_vertex = len(s)
        self.uroman = uroman
        self.props = {}
        self.simple_top_rom_cache = {}
        self.contains_script = defaultdict(bool)
        self.check_for_scripts()

    def check_for_scripts(self):
        for c in self.s:
            script_name = self.uroman.chr_script_name(c)
            self.contains_script[script_name] = True
            if regex.search(r'[\u2800-\u28FF]', self.s):
                self.contains_script['Braille'] = True

    def add_edge(self, edge: Edge):
        self.lattice[(edge.start, edge.end)].add(edge)
        self.lattice[(edge.start, 'right')].add(edge.end)
        self.lattice[(edge.end, 'left')].add(edge.start)

    def __str__(self):
        edges = []
        for start in range(self.max_vertex):
            for end in self.lattice[(start, 'right')]:
                for edge in self.lattice[(start, end)]:
                    edges.append(f'[{start}-{end}] {edge.txt} ({edge.type})')
        return ' '.join(edges)

    @staticmethod
    def char_is_braille(c: str) -> bool:
        return 0x2800 <= ord(c[0]) <= 0x28FF

    # Help Tibet
    def char_is_subjoined_letter(self, c: str) -> bool:
        return "SUBJOINED LETTER" in self.uroman.chr_name(c)

    def char_is_regular_letter(self, c: str) -> bool:
        char_name = self.uroman.chr_name(c)
        return ("LETTER" in char_name) and not ("SUBJOINED" in char_name)

    def char_is_letter(self, c: str) -> bool:
        return "LETTER" in self.uroman.chr_name(c)

    def char_is_vowel_sign(self, c: str) -> bool:
        return self.uroman.dict_bool[('is-vowel-sign', c)]

    def char_is_letter_or_vowel_sign(self, c: str) -> bool:
        return self.char_is_letter(c) or self.char_is_vowel_sign(c)

    def is_at_start_of_word(self, position: int) -> bool:
        # return not regex.match(r'(?:\pL|\pM)', self.s[position-1:position])
        first_char = self.s[position]
        first_char_is_braille = self.char_is_braille(first_char)
        end = position
        if (preceded_by_alpha := self.props.get(('preceded_by_alpha', end), None)) in (True, False):
            return not preceded_by_alpha
        if end:
            prev_orig_letter = self.s[end-1:end]
            if prev_orig_letter.isalpha():
                self.props[('preceded_by_alpha', position)] = True
                return False
        for start in self.lattice[(end, 'left')]:
            for edge in self.lattice[(start, end)]:
                prev_letter = None if edge.txt == '' else edge.txt[-1]
                if len(edge.txt) and (prev_letter.isalpha() or (first_char_is_braille and (prev_letter in ["'"]))):
                    self.props[('preceded_by_alpha', position)] = True
                    return False
        self.props[('preceded_by_alpha', position)] = False
        return True

    def is_at_end_of_word(self, position: int) -> bool:
        if (cached_followed_by_alpha := self.props.get(('followed_by_alpha', position), None)) in (True, False):
            return not cached_followed_by_alpha
        start = position
        if start < self.max_vertex:
            next_orig_letter = self.s[start:start+1]
            if next_orig_letter.isalpha():
                self.props[('followed_by_alpha', position)] = True
                return False
        while (start+1 < self.max_vertex) \
                and self.uroman.char_is_nonspacing_mark(self.s[start]) \
                and ('NUKTA' in self.uroman.chr_name(self.s[start])):
            start += 1
        for end in range(start + 1, self.max_vertex + 1):
            s = self.s[start:end]
            if not self.uroman.dict_bool[('s-prefix', s)]:
                break
            for rom_rule in self.uroman.rom_rules[s]:
                rom = rom_rule['t']
                if (not rom_rule['use-only-at-start-of-word']) and regex.search(r'\pL', rom):
                    self.props[('followed_by_alpha', position)] = True
                    return False
        self.props[('followed_by_alpha', position)] = False
        return True

    def is_at_end_of_syllable(self, position: int) -> Tuple[bool, str]:
        """At least initially for Thai"""
        prev_char = self.s[position-2] if position >= 2 else None
        # char = self.s[position-1] if position >= 1 else None
        next_char = self.s[position] if position < self.max_vertex else None
        if self.uroman.dict_str[('tone-mark', next_char)]:
            adj_position = position + 1
            next_char = self.s[adj_position] if adj_position < self.max_vertex else None
            # print('TONE-MARK', position, next_char)
        else:
            adj_position = position
        next_char2 = self.s[adj_position + 1] if adj_position + 1 < self.max_vertex else None
        if prev_char is None:
            return False, 'start-of-string'
        if not regex.search(r'(?:\pL|\pM)$', prev_char):  # start of token
            return False, 'start-of-token'
        if self.uroman.dict_str[('syllable-info', prev_char)] == 'written-pre-consonant-spoken-post-consonant':
            return False, 'pre-post-vowel-on-left'
        if self.uroman.dict_str[('syllable-info', next_char)] == 'written-pre-consonant-spoken-post-consonant':
            return True, 'pre-post-vowel-on-right'
        if adj_position >= self.max_vertex:  # end of string
            return True, 'end-of-string'
        # if not self.char_is_letter_or_vowel_sign(next_char):  # end of token
        if not regex.match(r'(?:\pL|\pM)', next_char):  # end of token
            return True, 'end-of-token'
        if position > 0:
            left_edge = self.best_left_neighbor_edge(position-1)
            if left_edge and regex.search(r'[bcdfghjklmnpqrstvxz]$', left_edge.txt):
                return False, 'consonant-to-the-left'
        next_char_rom = first_non_none(self.simple_top_romanization_candidate_for_span(adj_position,
                                                                                       adj_position + 2,
                                                                                       simple_search=True),
                                       self.simple_top_romanization_candidate_for_span(adj_position,
                                                                                       adj_position + 1,
                                                                                       simple_search=True),
                                       "?")
        if not regex.match(r"[aeiou]", next_char_rom.lower()):  # followed by consonant
            return True, f'not-followed-by-vowel {next_char_rom}'
        if (next_char == '\u0E2D') and (next_char2 is not None):  # THAI CHARACTER O ANG
            next_char2_rom = first_non_none(self.simple_top_romanization_candidate_for_span(adj_position+1,
                                                                                            adj_position+2,
                                                                                            simple_search=True),
                                            "?")
            if regex.match(r"[aeiou]", next_char2_rom.lower()):
                return True, 'o-ang-followed-by-vowel'  # In that context Thai char. "o ang" is considered a consonant
        return False, 'not-at-syllable-end-by-default'

    def romanization_by_first_rule(self, s) -> str | None:
        try:
            return self.uroman.rom_rules[s][0]['t']
        except IndexError:
            return None

    def expand_rom_with_special_chars(self, rom: str, start: int, end: int, **args) \
            -> Tuple[str, int, int, str | None]:
        """This method contains a number of special romanization heuristics that typically modify
        an existing or preliminary edge based on context."""
        orig_start = start
        uroman = self.uroman
        full_string = self.s
        annot = None
        if rom == '':
            return rom, start, end, None
        prev_char = (full_string[start-1] if start >= 1 else '')
        first_char = full_string[start]
        last_char = full_string[end-1]
        next_char = (full_string[end] if end < len(full_string) else '')
        # \u2820 is the Braille character indicating that the next letter is upper case
        if (prev_char == '\u2820') and regex.match(r'[a-z]', rom):
            return rom[0].upper() + rom[1:], start-1, end, 'rom exp'
        # noinspection SpellCheckingInspection   Normalize multi-upper case THessalonike -> Thessalonike,
        # noinspection SpellCheckingInspection   but don't change THESSALONIKE
        if start+1 == end and rom.isupper() and next_char.islower():
            ablation = args.get('ablation', '')     # VERBOSE
            if not ('nocap' in ablation):
                rom = rom.capitalize()
        # Japanese small tsu (and Gurmukhi addak) used as consonant doubler:
        if (prev_char and prev_char in 'っッ\u0A71') \
                and (uroman.chr_script_name(prev_char) == uroman.chr_script_name(prev_char)) \
                and (m_double_consonant := regex.match(r'(ch|[bcdfghjklmnpqrstwz])', rom)):
            # return m_double_consonant.group(1).replace('ch', 't') + rom, start-1, end, 'rom exp'
            # expansion might additional apply to the right
            if prev_char in 'っッ':  # for Japanese, per Hepburn, use tch
                rom = m_double_consonant.group(1).replace('ch', 't') + rom
            else:
                rom = m_double_consonant.group(1).replace('ch', 'c') + rom
            start = start-1
            first_char = full_string[start]
            prev_char = (full_string[start-1] if start >= 1 else '')
        # Thai
        if uroman.chr_script_name(first_char) == 'Thai':
            if (start+1 == end) and regex.match(r'[bcdfghjklmnpqrstvwxyz]+$', rom):
                if uroman.dict_str[('syllable-info', prev_char)] == 'written-pre-consonant-spoken-post-consonant':
                    for vowel_prefix_len in [1]:
                        if vowel_prefix_len <= start:
                            for vowel_suffix_len in [3, 2, 1]:
                                if end + vowel_suffix_len <= len(full_string):
                                    pattern = (full_string[start-vowel_prefix_len: start]
                                               + '–'
                                               + full_string[end:end+vowel_suffix_len])
                                    if uroman.rom_rules[pattern]:
                                        vowel_rom_rule = uroman.rom_rules[pattern][0]
                                        vowel_rom = vowel_rom_rule['t']
                                        # print(f" PATTERN {pattern} ({full_string[start:end]}/{rom}) {rom}{vowel_rom}")
                                        return rom + vowel_rom, start-vowel_prefix_len, end+vowel_suffix_len, 'rom exp'
            if (uroman.chr_script_name(prev_char) == 'Thai') \
                    and (uroman.dict_str[('syllable-info', prev_char)]
                         == 'written-pre-consonant-spoken-post-consonant') \
                    and regex.match(r'[bcdfghjklmnpqrstvwxyz]', rom) \
                    and (vowel_rom := self.romanization_by_first_rule(prev_char)):
                return rom + vowel_rom, start-1, end, 'rom exp'
            # THAI CHARACTER O ANG
            if (first_char == '\u0E2D') and (end - start == 1):
                prev_script = uroman.chr_script_name(prev_char)
                next_script = uroman.chr_script_name(next_char)
                prev_rom = self.find_rom_edge_path_backwards(0, start, 1, return_str=True)
                next_rom = self.romanization_by_first_rule(next_char)
                # if not recursive:
                #     lc = uroman.romanize_string(full_string[:start], lcode=self.lcode, recursive=True)
                #     rc = uroman.romanize_string(full_string[end:], lcode=self.lcode, recursive=True)
                #     print('PP', start, end, prev_script, next_script, prev_rom, next_rom, '  LC:', lc[-40:],
                #           '  RC:', rc[:40])
                # delete THAI CHARACTER O ANG unless it is surrounded on both sides by a Thai consonant
                if not ((prev_script == 'Thai') and (next_script == 'Thai')
                        and regex.match(r'[bcdfghjklmnpqrstvwxz]+$', prev_rom)
                        and regex.match(r'[bcdfghjklmnpqrstvwxz]+$', next_rom)):
                    # if not recursive:
                    #     print(f'* DELETE O ANG {first_char} {start}-{end}   LC: {lc[-40:]}  RC: {rc[:40]}')
                    return '', start, end, 'rom del'
        # Coptic: consonant + grace-accent = e + consonant
        if next_char and (next_char == "\u0300") and (uroman.chr_script_name(last_char) == "Coptic")\
                and (not self.simple_top_romanization_candidate_for_span(orig_start, end+1)):
            rom = 'e' + rom
            end = end+1
            last_char = full_string[end - 1]
            next_char = (full_string[end] if end < len(full_string) else '')
            annot = 'rom exp'
        # Japanese small y: ki + small ya = kya etc.
        y_rom = None
        if (next_char and next_char in 'ゃゅょャュョ') \
                and (uroman.chr_script_name(last_char) == uroman.chr_script_name(next_char)) \
                and regex.search(r'([bcdfghjklmnpqrstvwxyz]i$)', rom) \
                and (y_rom := self.romanization_by_first_rule(next_char)) \
                and (not self.simple_top_romanization_candidate_for_span(orig_start, end+1)) \
                and (not self.simple_top_romanization_candidate_for_span(start, end+1)):
            rom = rom[:-1] + y_rom
            end = end+1
            last_char = full_string[end - 1]
            next_char = (full_string[end] if end < len(full_string) else '')
            annot = 'rom exp'
        # Japanese vowel lengthener (U+30FC)
        last_rom_char = last_chr(rom)
        if (next_char == 'ー') \
                and (uroman.chr_script_name(last_char) in ('Hiragana', 'Katakana')) \
                and (last_rom_char in 'aeiou'):
            return rom + last_rom_char, start, end+1, 'rom exp'
        # Virama (in Indian languages)
        if self.uroman.dict_bool[('is-virama', next_char)]:
            return rom, start, end + 1, "rom exp"
        if rom.startswith(' ') and ((start == 0) or (prev_char == ' ')):
            rom = rom[1:]
        if rom.endswith(' ') and ((end == len(full_string)+1) or (next_char == ' ')):
            rom = rom[:-1]
        return rom, start, end, annot

    def prep_braille(self, **_args) -> None:
        if self.contains_script['Braille']:
            dots6 = '\u2820'  # characters in following word are upper case
            all_caps = False
            for i, c in enumerate(self.s):
                if (i >= 1) and (self.s[i-1] == dots6) and (c == dots6):
                    all_caps = True
                elif all_caps:
                    if c in '\u2800':  # Braille space
                        all_caps = False
                    else:
                        self.props[('is-upper', i)] = True

    def pick_tibetan_vowel_edge(self, **args) -> None:
        if not self.contains_script['Tibetan']:
            return None
        verbose = bool(args.get('verbose'))
        s = self.s
        uroman = self.uroman
        tibetan_syllable = []
        tibetan_letter_positions = []
        for start in range(self.max_vertex):
            c = s[start]
            if (uroman.chr_script_name(c) == 'Tibetan') and self.char_is_letter_or_vowel_sign(c):
                tibetan_letter_positions.append(start)
            else:
                if tibetan_letter_positions:
                    tibetan_syllable.append(tibetan_letter_positions)
                    tibetan_letter_positions = []
        if tibetan_letter_positions:
            tibetan_syllable.append(tibetan_letter_positions)
        for tibetan_letter_positions in tibetan_syllable:
            vowel_pos = None
            orig_txt = ''
            roms = []
            subjoined_letter_positions = []
            first_letter_position = tibetan_letter_positions[0]
            for i in tibetan_letter_positions:
                c = s[i]
                orig_txt += c
                rom = first_non_none(self.simple_top_romanization_candidate_for_span(i, i+1), "?")
                self.props[('edge-vowel', i)] = None
                if self.char_is_vowel_sign(c) or (rom and regex.match(r"[aeiou]+$", rom)):
                    vowel_pos = i
                    self.props[('edge-vowel', i)] = True
                    # delete any syllable initial ' before vowel
                    if roms == ["'"]:
                        self.props[('edge-delete', i-1)] = True
                elif self.char_is_subjoined_letter(c):
                    subjoined_letter_positions.append(i)
                    if i > first_letter_position:
                        if c == "\u0FB0":
                            vowel_pos = i-1
                            self.props[('edge-vowel', i-1)] = True
                        else:
                            self.props[('edge-vowel', i-1)] = False
                    rom = regex.sub(r'([bcdfghjklmnpqrstvwxyz].*)a$', r'\1', rom)
                elif c == "\u0F60":  # Tibetan letter -a, romanized as an apostrophe ("'")
                    self.props[('edge-vowel', i)] = False
                    if i > first_letter_position:
                        vowel_pos = i-1
                        self.props[('edge-vowel', i-1)] = True
                        if i == tibetan_letter_positions[-1]:
                            self.props[('edge-delete', i)] = True
                    if roms and not (roms[-1] in "aeiou"):
                        rom = "a'"
                    else:
                        rom = "'"
                else:
                    rom = regex.sub(r'([bcdfghjklmnpqrstvwxyz].*)a$', r'\1', rom)
                roms.append(rom)
            if vowel_pos is not None:
                for i in tibetan_letter_positions:
                    if self.props.get(('edge-vowel', i)) is None:
                        self.props[('edge-vowel', i)] = False
            else:
                best_cost, best_vowel_pos, best_pre, best_post = math.inf, None, None, None
                n_letters = len(tibetan_letter_positions)
                for i in tibetan_letter_positions:
                    rel_pos = i - first_letter_position
                    pre, post = ''.join(roms[:rel_pos+1]), ''.join(roms[rel_pos+1:])
                    if self.props.get(('edge-vowel', i)) is False:
                        cost = 20
                        if cost < best_cost:
                            best_cost, best_vowel_pos, best_pre, best_post = cost, i, pre, post
                    elif n_letters == 1:
                        cost = 0
                        if cost < best_cost:
                            best_cost, best_vowel_pos, best_pre, best_post = cost, i, pre, post
                    elif n_letters == 2:
                        cost = 0 if i == 0 else 0.1
                        if cost < best_cost:
                            best_cost, best_vowel_pos, best_pre, best_post = cost, i, pre, post
                    else:
                        good_suffix = regex.match(r"(?:|[bcdfghjklmnpqrstvwxz]|bh|bs|ch|cs|dd|ddh|"
                                                  r"dh|dz|dzh|gh|gr|gs|kh|khs|kss|n|nn|nt|ms|ng|ngs|ns|ph|"
                                                  r"rm|sh|ss|th|ts|tsh|tt|tth|zh|zhs)'?$", post)
                        # noinspection SpellCheckingInspection
                        good_prefix = regex.match(r"'?(?:.|bd|br|brg|brgy|bs|bsh|bst|bt|bts|by|bz|bzh|"
                                                  r"ch|db|dby|dk|dm|dp|dpy|dr|"
                                                  r"gl|gn|gr|gs|gt|gy|gzh|kh|khr|khy|kr|ky|ld|lh|lt|mkh|mny|mth|mtsh|"
                                                  r"ny|ph|phr|phy|rgy|rk|el|rn|rny|rt|rts|"
                                                  r"sk|skr|sky|sl|sm|sn|sny|sp|spy|sr|st|th|ts|tsh)$", pre)
                        subjoined_suffix = all([x in subjoined_letter_positions
                                                for x in tibetan_letter_positions[rel_pos+2:]])
                        # print('GOOD', good_suffix, good_prefix, subjoined_suffix, f'{pre}a{post}',
                        #       subjoined_letter_positions, tibetan_letter_positions[rel_pos+2:])
                        if good_suffix and good_prefix:
                            cost = len(pre) * 0.1
                        elif good_suffix:
                            cost = len(pre)
                        elif subjoined_suffix and good_prefix:
                            cost = len(pre) * 0.3
                        elif subjoined_suffix:
                            cost = len(pre) * 0.5
                        else:
                            cost = math.inf
                    if cost < best_cost:
                        best_cost, best_vowel_pos, best_pre, best_post = cost, i, pre, post
                if best_vowel_pos is not None:
                    for i in tibetan_letter_positions:
                        if self.props.get(('edge-vowel', i)) is None:
                            value = (i == best_vowel_pos)
                            self.props[('edge-vowel', i)] = value
                if verbose:
                    best_cost = best_cost if isinstance(best_cost, int) else round(best_cost, 2)
                    sys.stderr.write(f'Tib. best cost: "{best_pre}a{best_post}"  o:{orig_txt}  c:{round(best_cost, 2)}'
                                     f'   p:{best_vowel_pos} {tibetan_letter_positions}\n')

    def add_default_abugida_vowel(self, rom: str, start: int, end: int, annotation: str = '') -> str:
        """Adds an abugida vowel (e.g. "a") where needed. Important for many languages in South Asia."""
        uroman = self.uroman
        s = self.s
        # noinspection PyBroadException
        try:
            first_s_char = s[start]
            last_s_char = s[end-1]
            script_name = uroman.chr_script_name(first_s_char)
            script = self.uroman.scripts[script_name.lower()]
            if not (abugida_default_vowels := script['abugida-default-vowels']):
                return rom
            key = (script, rom)
            if key in uroman.abugida_cache:
                base_rom, base_rom_plus_vowel, mod_rom = uroman.abugida_cache[key]
                rom = mod_rom
            else:
                vowels_regex1 = '|'.join(abugida_default_vowels)   # e.g. 'a' or 'a|o'
                vowels_regex2 = '|'.join(map(lambda x: x + '+', abugida_default_vowels))   # e.g. 'a+' or 'a+|o+'
                # noinspection SpellCheckingInspection
                if m := regex.match(fr'([cfghkmnqrstxy]?y)({vowels_regex2})-?$', rom):
                    base_rom = m.group(1)
                    base_rom_plus_vowel = base_rom + m.group(2)
                elif m := regex.match(fr'([bcdfghjklmnpqrstvwxyz]+)({vowels_regex1})-?$', rom):
                    base_rom = m.group(1)
                    base_rom_plus_vowel = base_rom + m.group(2)
                    if rom.endswith('-') and (start+1 == end) and rom[0].isalpha():
                        rom = rom[:-1]
                else:
                    base_rom = rom
                    base_rom_plus_vowel = base_rom + abugida_default_vowels[0]
                if (not regex.match(r"[bcdfghjklmnpqrstvwxyz]+$", base_rom)
                        and (not ((script_name == 'Tibetan') and (base_rom == "'")))):
                    base_rom, base_rom_plus_vowel = None, None
                uroman.abugida_cache[key] = (base_rom, base_rom_plus_vowel, rom)
            if base_rom is None:
                return rom
            if 'tail' in annotation:
                return rom
            prev_s_char = s[start-1] if start >= 1 else ''
            next_s_char = s[end] if len(s) > end else ''
            next2_s_char = s[end+1] if len(s) > end+1 else ''
            if script_name == 'Tibetan':
                if self.props.get(('edge-delete', start)):
                    return ''
                elif self.props.get(('edge-vowel', start)):
                    return base_rom_plus_vowel
                else:
                    return base_rom
            if (next_s_char and ((base_rom in "bcdfghklmnpqrstvwz") or (base_rom in ["ng"]))
                    and (next_s_char in "យ")):  # Khmer yo
                return base_rom
            if self.uroman.dict_bool[('is-vowel-sign', next_s_char)]:
                return base_rom
            if self.uroman.dict_bool[('is-medial-consonant-sign', next_s_char)]:
                return base_rom
            if self.char_is_subjoined_letter(next_s_char):
                return base_rom
            if self.uroman.char_is_nonspacing_mark(next_s_char) \
                    and self.uroman.dict_bool[('is-vowel-sign', next2_s_char)]:
                return base_rom
            if self.uroman.dict_bool[('is-virama', next_s_char)]:
                return base_rom
            if self.uroman.char_is_nonspacing_mark(next_s_char) \
                    and self.uroman.dict_bool[('is-virama', next2_s_char)]:
                return base_rom
            if self.uroman.dict_bool[('is-virama', prev_s_char)]:
                return base_rom_plus_vowel
            if self.is_at_start_of_word(start) and not regex.search('r[aeiou]', rom):
                return base_rom_plus_vowel
            # delete many final schwas from most Devanagari languages (except: Sanskrit)
            if self.is_at_end_of_word(end):
                if (script_name in ("Devanagari",)) and (self.lcode not in ('san',)):  # Sanskrit
                    return rom
                elif self.lcode in ('asm', 'ben', 'guj', 'kas', 'pan'):
                    return rom
                else:
                    return base_rom_plus_vowel
            if uroman.chr_script_name(prev_s_char) != script_name:
                return base_rom_plus_vowel
            if 'VOCALIC' in self.uroman.chr_name(last_s_char):
                return base_rom
            if uroman.chr_script_name(next_s_char) == script_name:
                return base_rom_plus_vowel
        except Exception:
            return rom
        else:
            pass
            # print('ABUGIDA', rom, start, script_name, script, abugida_default_vowels, prev_s_char, next_s_char)
        return rom

    def cand_is_valid(self, rom_rule: RomRule, start: int, end: int, rom: str) -> bool:
        if rom is None:
            return False
        if rom_rule['dont-use-at-start-of-word'] and self.is_at_start_of_word(start):
            return False
        if rom_rule['use-only-at-start-of-word'] and not self.is_at_start_of_word(start):
            return False
        if rom_rule['dont-use-at-end-of-word'] and self.is_at_end_of_word(end):
            return False
        if rom_rule['use-only-at-end-of-word'] and not self.is_at_end_of_word(end):
            return False
        if rom_rule['use-only-for-whole-word'] \
                and not (self.is_at_start_of_word(start) and self.is_at_end_of_word(end)):
            return False
        if (lcodes := rom_rule['lcodes']) and (self.lcode not in lcodes):
            return False
        return True

    # @profile
    def simple_sorted_romanization_candidates_for_span(self, start, end) -> List[str]:
        s = self.s[start:end]
        if not self.uroman.dict_bool[('s-prefix', s)]:
            return []
        rom_rule_candidates = []
        for rom_rule in self.uroman.rom_rules[s]:
            rom = rom_rule['t']
            if self.cand_is_valid(rom_rule, start, end, rom):
                rom_rule_candidates.append((rom_rule['n-restr'] or 0, rom_rule['t']))
        rom_rule_candidates.sort(reverse=True)
        return [x[1] for x in rom_rule_candidates]

    def simple_top_romanization_candidate_for_span(self, start, end, simple_search: bool = False) -> str | None:
        if (start < 0) or (end > self.max_vertex):
            return None
        span_range = (start, end)
        if (cached_result := self.simple_top_rom_cache.get(span_range)) is not None:
            return cached_result
        best_cand, best_n_restr, best_rom_rule = None, None, None
        for rom_rule in self.uroman.rom_rules[self.s[start:end]]:
            if self.cand_is_valid(rom_rule, start, end, rom_rule['t']):
                n_restr = rom_rule['n-restr'] or 0
                if best_n_restr is None or (n_restr > best_n_restr):
                    best_cand, best_n_restr, best_rom_rule = rom_rule['t'], n_restr, rom_rule
        if simple_search:
            return best_cand
        if best_rom_rule:
            t_at_end_of_syllable = best_rom_rule['t-at-end-of-syllable']
            # noinspection GrazieInspection
            if t_at_end_of_syllable is not None:
                is_at_end_of_syllable, rationale = self.is_at_end_of_syllable(end)
                if is_at_end_of_syllable:
                    best_cand = t_at_end_of_syllable
                # print(f"   SIMPLE {start}-{end} {best_cand} ({best_rom_rule['t']},{t_at_end_of_syllable}) "
                #       f"END:{is_at_end_of_syllable} ({rationale})")
        self.simple_top_rom_cache[span_range] = best_cand
        # if (best_rom_rule is not None) and ('cancel' in (prov := best_rom_rule['prov'])):
        #     sys.stderr.write(f'   Cancel {self.s} ({start}-{end}) {prov} {self.s[start:end]}\n')
        return best_cand

    def decomp_rom(self, char_position: int) -> str | None:
        """Input: decomposable character such as ﻼ or ½
        Output: la or 1/2"""
        full_string = self.s
        char = full_string[char_position]
        rom = None
        if ud_decomp_s := ud.decomposition(char):
            format_comps = []
            other_comps = []
            decomp_s = ''
            # name = self.uroman.chr_name(char)
            for ud_decomp_elem in ud_decomp_s.split():
                if ud_decomp_elem.startswith("<"):
                    format_comps.append(ud_decomp_elem)
                else:
                    try:
                        norm_char = chr(int(ud_decomp_elem, 16))
                    except ValueError:
                        other_comps.append(ud_decomp_elem)
                    else:
                        decomp_s += norm_char
            if (format_comps and (format_comps[0] not in ('<super>', '<sub>', '<noBreak>', '<compat>'))
                    and (not other_comps) and decomp_s):
                rom = self.uroman.romanize_string(decomp_s, self.lcode)
            # make sure to add a space for 23½ -> 23 1/2
            if rom and ud.numeric(char, None):
                rom = rom.replace('⁄', '/')
                if char_position >= 1 and ud.numeric(full_string[char_position-1], None):
                    rom = ' ' + rom
                if (char_position+1 < len(full_string)) and ud.numeric(full_string[char_position+1], None):
                    rom += ' '
        return rom

    def add_romanization(self, **args):
        """Adds a romanization edge to the romanization lattice."""
        for start in range(self.max_vertex):
            for end in range(start+1, self.max_vertex+1):
                if not self.uroman.dict_bool[('s-prefix', self.s[start:end])]:
                    break
                if (rom := self.simple_top_romanization_candidate_for_span(start, end)) is not None:
                    if self.contains_script['Braille'] and (start+1 == end):
                        if self.props.get(('is-upper', start)):
                            rom = rom.upper()
                    edge_annotation = 'rom'
                    if regex.match(r'\+(m|ng|n|h|r)', rom):
                        rom, edge_annotation = rom[1:], 'rom tail'
                    new_rom = self.add_default_abugida_vowel(rom, start, end, annotation=edge_annotation)
                    if new_rom.startswith(rom):
                        suffix = new_rom[len(rom):]
                        if suffix and regex.match(r'[aeiou]+$', suffix):
                            edge_annotation += f' c:{rom} s:{suffix}'
                    rom = new_rom
                    # orig_rom, orig_start, orig_end = rom, start, end
                    rom, start2, end2, exp_edge_annotation \
                        = self.expand_rom_with_special_chars(rom, start, end, annotation=edge_annotation,
                                                             recursive=args.get('recursive', False), **args)
                    edge_annotation = exp_edge_annotation or edge_annotation
                    # if (orig_rom, orig_start, orig_end) != (rom, start, end):
                    #     print(f"EXP {s} {orig_rom} {orig_start}-{orig_end} -> {rom} {start}-{end}")
                    # if rom != rom_orig: print('** Add ABUGIDA', rom, start, end, rom2)
                    self.add_edge(Edge(start2, end2, rom, edge_annotation))
            if start < len(self.s):
                char = self.s[start]
                cp = ord(char)
                # Korean Hangul characters
                if 0xAC00 <= cp <= 0xD7A3:
                    if rom := self.uroman.unicode_hangul_romanization(char):
                        self.add_edge(Edge(start, start+1, rom, 'rom'))
                # character decomposition
                if rom_decomp := self.decomp_rom(start):
                    self.add_edge(Edge(start, start + 1, rom_decomp, 'rom decomp'))

    @staticmethod
    def update_edge_list(edges, new_edge, old_edges) -> List[NumEdge]:
        new_edge_not_yet_added = True
        result = []
        for edge in edges:
            if edge in old_edges:
                edge.active = False
                if new_edge_not_yet_added:
                    result.append(new_edge)
                    new_edge_not_yet_added = False
            else:
                result.append(edge)
        if new_edge_not_yet_added:
            result.append(new_edge)
        return result

    @staticmethod
    def edge_is_digit(edge: Edge | None) -> bool:
        return (isinstance(edge, NumEdge)
                and (edge.value is not None)
                and isinstance(edge.value, int)
                and (edge.type == 'digit')
                and (0 <= edge.value <= 9)
                and (edge.end - edge.start == 1))

    @staticmethod
    def is_gap_null_edge(edge: Edge) -> bool:
        return isinstance(edge, NumEdge) and (edge.orig_txt in ('零', '〇'))

    @staticmethod
    def braille_digit(char: str) -> str | None:
        position = '\u281A\u2801\u2803\u2809\u2819\u2811\u280B\u281B\u2813\u280A'.find(char)  # Braille 0-9
        return str(position) if position >= 0 else None

    def add_braille_number(self, start: int, end: int, txt: str, **_args) -> None:
        new_edge = NumEdge(start, end, txt, self.uroman)
        new_edge.type = 'number'
        self.add_edge(new_edge)

    def add_braille_numbers(self, **_args):
        if self.contains_script['Braille']:
            s = self.s
            num_s, start = '', None
            for i in range(len(s)):
                char = s[i]
                if char == '\u283C':  # number mark
                    if start is None:
                        start = i
                elif (start is not None) and (digit_s := self.braille_digit(char)):
                    num_s += digit_s
                elif (start is not None) and (char == '\u2832'):  # period
                    num_s += '.'
                elif (start is not None) and (char == '\u2802'):  # comma
                    num_s += ','
                elif isinstance(start, int) and (num_s != ''):
                    self.add_braille_number(start, i, num_s)
                    num_s, start = '', None
            if (start is not None) and (num_s != ''):
                self.add_braille_number(start, len(s), num_s)

    # noinspection PyUnboundLocalVariable
    def add_numbers(self, uroman, **args):
        """Adds a numerical romanization edge to the romanization lattice, currently just for digits."""
        verbose = bool(args.get('verbose'))
        s = self.s
        num_edges = []
        for start in range(len(s)):
            char = s[start]
            if uroman.num_props[char]:
                new_edge = NumEdge(start, start + 1, char, uroman)
                num_edges.append(new_edge)
                if verbose:
                    print('NumEdge', new_edge)
                self.add_edge(new_edge)
        # D1 sequence of digits 1234
        for edge in num_edges:
            if self.edge_is_digit(edge) and edge.active:  # and (edge.value != 0):
                n_decimal_points = 0
                n_decimals = None
                new_value_s = str(edge.value)
                sub_edges = [edge]
                prev_edge = edge
                while True:
                    right_edge = self.best_right_neighbor_edge(prev_edge.end)
                    if self.edge_is_digit(right_edge):
                        sub_edges.append(right_edge)
                        new_value_s += str(right_edge.value)
                        if n_decimals is not None:
                            n_decimals += 1
                        prev_edge = right_edge
                    elif ((prev_edge.end < len(s)) and (s[prev_edge.end] == '.') and (n_decimal_points == 0)
                            and (right_edge2 := self.best_right_neighbor_edge(prev_edge.end + 1))
                            and self.edge_is_digit(right_edge2)):
                        if right_edge is None:
                            right_edge = Edge(prev_edge.end, prev_edge.end+1, s[prev_edge.end],
                                              'decimal period')
                            self.add_edge(right_edge)
                        sub_edges.append(right_edge)
                        sub_edges.append(right_edge2)
                        new_value_s += '.' + str(right_edge2.value)
                        n_decimal_points += 1
                        n_decimals = 1
                        prev_edge = right_edge2
                    else:
                        break
                if len(sub_edges) >= 2:
                    new_value = float(new_value_s) if '.' in new_value_s else int(new_value_s)
                    new_edge = NumEdge(sub_edges[0].start, sub_edges[-1].end, str(new_value), uroman, active=True)
                    new_edge.update(value=new_value, value_s=new_value_s, n_decimals=n_decimals, num_base=1, 
                                    e_type='D1', script=sub_edges[-1].script)
                    self.add_edge(new_edge)
                    num_edges = self.update_edge_list(num_edges, new_edge, sub_edges)
                    if verbose:
                        print(new_edge.type, new_edge)
        # G1 combine (*) "single digits" 2*100=200, 3*10= 30
        for edge in num_edges:
            if (isinstance(edge, NumEdge) and edge.active and (edge.num_base == 1)
                    and isinstance(edge.value, int) and (edge.value >= 1)):
                right_edge = self.best_right_neighbor_edge(edge.end, skip_num_edge=False)
                if (right_edge
                        and isinstance(right_edge, NumEdge)
                        and right_edge.active
                        and isinstance(right_edge.value, int)
                        and (right_edge.num_base > 1)
                        and (not right_edge.is_large_power)):
                    new_value = edge.value * right_edge.value
                    new_edge = NumEdge(edge.start, right_edge.end, str(new_value), uroman, active=True)
                    new_edge.update(value=new_value, num_base=right_edge.num_base, e_type='G1',
                                    orig_txt=edge.orig_txt + right_edge.orig_txt,
                                    script=right_edge.script)
                    self.add_edge(new_edge)
                    num_edges = self.update_edge_list(num_edges, new_edge, [edge, right_edge])
                    if verbose:
                        print(new_edge.type, new_edge)
        # G2 combine (+) G1 "single digits" 200+30+4=234 (within larger blocks of 1000, 1000000)
        for edge in num_edges:
            if isinstance(edge, NumEdge) and edge.active and isinstance(edge.value, int) and not edge.is_large_power:
                sub_edges = [edge]
                prev_edge = edge
                prev_non_edge = edge  # None if (edge.orig_txt in '零') else prev_edge
                right_edge, right_edge2 = None, None
                while (prev_edge
                       and (right_edge := self.best_right_neighbor_edge(prev_edge.end, skip_num_edge=False))
                       and isinstance(right_edge, NumEdge)
                       and right_edge.active
                       and isinstance(right_edge.value, int)
                       and (not right_edge.is_large_power)
                       and (self.is_gap_null_edge(prev_non_edge)
                            or ((prev_non_edge.num_base > right_edge.value)
                                and (prev_non_edge.num_base > right_edge.num_base)))):
                    sub_edges.append(right_edge)
                    prev_edge = right_edge
                    if not self.is_gap_null_edge(right_edge):
                        prev_non_edge = right_edge
                if len(sub_edges) >= 2:
                    new_value = sum([e.value for e in sub_edges])
                    new_edge = NumEdge(sub_edges[0].start, sub_edges[-1].end, str(new_value), uroman, active=True)

                    new_edge.update(value=new_value, num_base=sub_edges[-1].num_base, e_type='G2',
                                    orig_txt=''.join([e.orig_txt for e in sub_edges]),
                                    script=sub_edges[-1].script)
                    self.add_edge(new_edge)
                    num_edges = self.update_edge_list(num_edges, new_edge, sub_edges)
                    new_edge.type = 'G2'
                    if verbose:
                        print(new_edge.type, new_edge)
        # G3 combine (*) G2 blocks with large powers, e.g. 234*1000 = 234000
        for edge in num_edges:
            if (isinstance(edge, NumEdge) and edge.active and (not edge.is_large_power)
                    and (isinstance(edge.value, int) or isinstance(edge.value, float))):
                right_edge = self.best_right_neighbor_edge(edge.end, skip_num_edge=False)
                if (right_edge
                        and isinstance(right_edge, NumEdge)
                        and right_edge.active
                        and isinstance(right_edge.value, int)
                        and (right_edge.num_base > 1)
                        and right_edge.is_large_power):
                    new_value = round(edge.value * right_edge.value, 5)
                    if isinstance(new_value, float) and new_value.is_integer():
                        new_value = int(new_value)
                    new_edge = NumEdge(edge.start, right_edge.end, str(new_value), uroman, active=True)
                    new_edge.update(value=new_value, num_base=right_edge.num_base, e_type='G3',
                                    orig_txt=edge.orig_txt + right_edge.orig_txt,
                                    script=right_edge.script)
                    self.add_edge(new_edge)
                    num_edges = self.update_edge_list(num_edges, new_edge, [edge, right_edge])
                    if verbose:
                        print(new_edge.type, new_edge)
        # G4 combine (+) G3 blocks 234000+567=234567
        for edge in num_edges:
            if isinstance(edge, NumEdge) and edge.active and isinstance(edge.value, int):
                sub_edges = [edge]
                while ((prev_edge := sub_edges[-1])
                       and (right_edge := self.best_right_neighbor_edge(prev_edge.end, skip_num_edge=False))
                       and isinstance(right_edge, NumEdge)
                       and right_edge.active
                       and isinstance(right_edge.value, int)
                       and (prev_edge.num_base > right_edge.value)
                       and (prev_edge.num_base > right_edge.num_base)):
                    if ((prev_edge.script == 'CJK')
                            and (prev_edge.num_base >= 1000)
                            and ('tag' not in prev_edge.type)
                            and regex.match('10+$', str(prev_edge.num_base))
                            and (1 <= right_edge.value <= 9)
                            and (right_edge.start + 1 == right_edge.end)):
                        new_num_base = prev_edge.num_base // 10
                        new_value = new_num_base * right_edge.value
                        # print('DIGIT TAG', prev_edge, right_edge, new_value)
                        right_edge.value = new_value
                        right_edge.num_base = new_num_base
                        right_edge.type = 'G4tag'
                    sub_edges.append(right_edge)
                if len(sub_edges) >= 2:
                    new_value = sum([e.value for e in sub_edges])
                    new_edge = NumEdge(sub_edges[0].start, sub_edges[-1].end, str(new_value), uroman, active=True)
                    new_edge.update(value=new_value, num_base=sub_edges[-1].num_base, e_type='G4',
                                    orig_txt=''.join([e.orig_txt for e in sub_edges]),
                                    script=sub_edges[-1].script)
                    self.add_edge(new_edge)
                    num_edges = self.update_edge_list(num_edges, new_edge, sub_edges)
                    if verbose:
                        print(new_edge.type, new_edge)

        # G5 (Chinese) fractions, percentages
        for edge in num_edges:
            if edge.value is None:
                continue
            if not isinstance(edge.value, int):
                continue
            for fraction_connector in self.uroman.fraction_connectors:
                fraction_connector_end = edge.end+len(fraction_connector)
                if self.s[edge.end:fraction_connector_end] != fraction_connector:
                    continue
                right_edge = self.best_right_neighbor_edge(fraction_connector_end)
                if right_edge.value is None:
                    continue
                if edge.value == 100:
                    if (isinstance(right_edge.value, int) or isinstance(right_edge.value, float)) and (edge.value >= 0):
                        new_edge = Edge(edge.start, right_edge.end, f'{right_edge.value}%', 'percentage')
                        self.add_edge(new_edge)
                        num_edges = self.update_edge_list(num_edges, new_edge, [edge, right_edge])
                else:
                    if isinstance(right_edge.value, int) and (edge.value > 0):
                        new_edge = NumEdge(edge.start, right_edge.end, f'{right_edge.value}/{edge.value}',
                                           self.uroman, True)
                        new_edge.fraction = Fraction(right_edge.value, edge.value)
                        new_edge.type = 'fraction'
                        self.add_edge(new_edge)
                        num_edges = self.update_edge_list(num_edges, new_edge, [edge, right_edge])

        # G6 plus/minus signs
        for edge in num_edges:
            _left_edge = self.best_left_neighbor_edge(edge.start)
            for minus_sign in self.uroman.minus_signs:
                if self.s[edge.start-len(minus_sign):edge.start] == minus_sign:
                    new_edge = Edge(edge.start-len(minus_sign), edge.end, f'-{edge.txt}', f'{edge.type} -')
                    self.add_edge(new_edge)
            for plus_sign in self.uroman.plus_signs:
                if self.s[edge.start-len(plus_sign):edge.start] == plus_sign:
                    new_edge = Edge(edge.start-len(plus_sign), edge.end, f'+{edge.txt}', f'{edge.type} +')
                    self.add_edge(new_edge)

        # F1
        for edge in num_edges:
            # cushion fractions with spaces as needed: e.g. 23½ -> 23 1/2 or 十一五 -> 11 5
            if isinstance(edge, NumEdge) and regex.match(r'\d', edge.txt):
                left_edge = self.best_left_neighbor_edge(edge.start)
                if left_edge and regex.search(r'\d$', left_edge.txt):
                    if edge.fraction:
                        sep = ' '
                    else:
                        sep = '·'
                    edge.txt = sep + edge.txt

        # exceptions: mostly some single-digit number characters
        for edge in num_edges:
            if (isinstance(edge, NumEdge) and edge.active and (edge.value is not None)
                    and (((edge.value > 1000) and (edge.start + 1 == edge.end))
                         or (edge.orig_txt in '兩參参伍陆陸什')
                         or (edge.orig_txt in ('京兆', )))):
                edge.active = False
        if verbose:  # or (num_edges and any([e.type in ['G1', 'G2', 'G3', 'G4'] for e in num_edges])):
            if num_edges:
                print('actives:')
            for num_edge in num_edges:
                print(num_edge)
        for start in range(len(s)):
            start_char = s[start]
            if (best_edge := self.best_edge_in_span(start, start+1)) and isinstance(best_edge, NumEdge):
                continue
            if (num := ud_numeric(start_char)) is not None:
                name = self.uroman.chr_name(start_char)
                if ("DIGIT" in name) and isinstance(num, int) and (0 <= num <= 9):
                    # if start_char not in '0123456789': print('DIGIT', s[start], num, name)
                    self.add_edge(Edge(start, start + 1, str(num), 'num'))
                else:
                    uroman.stats[('*NUM', start_char, num)] += 1

    def add_rom_fall_back_singles(self, **_args):
        """For characters in the original string not covered by romanizations and numbers,
        add a fallback edge based on type, romanization of single char, or original char."""
        for start in range(self.max_vertex):
            end = start+1
            orig_char = self.s[start]
            if not self.lattice[(start, end)]:
                rom, edge_annotation = orig_char, 'orig'
                if self.uroman.char_is_nonspacing_mark(rom):
                    rom, edge_annotation = '', 'Mn'
                elif self.uroman.char_is_format_char(rom):  # e.g. zero-width non-joiner, zero-width joiner
                    rom, edge_annotation = '', 'Cf'
                elif ud.category(orig_char) == 'Co':
                    rom, edge_annotation = '', 'Co'
                elif rom == ' ':
                    edge_annotation = 'orig'
                # elif self.uroman.char_is_space_separator(rom):
                #     rom, edge_annotation = ' ', 'Zs'
                elif (rom2 := self.simple_top_romanization_candidate_for_span(start, end)) is not None:
                    rom = rom2
                    if regex.match(r'\+(m|ng|n|h|r)', rom):
                        rom = rom[1:]
                    edge_annotation = 'rom single'
                # else the original values still hold: rom, edge_annotation = orig_char, 'orig'
                self.add_edge(Edge(start, end, rom, edge_annotation))

    @staticmethod
    def add_new_edge(old_edges: List[Edge], start: int, end: int, new_rom: str, new_type: str, position: int | None,
                     old_edge_dict: dict)\
            -> None:
        if (start, end, new_rom) not in old_edge_dict:
            new_edge = Edge(start, end, new_rom, new_type)
            if position is None:
                old_edges.append(new_edge)
            else:
                old_edges.insert(position + 1, new_edge)
            old_edge_dict[(start, end, new_rom)] = new_edge
            # print(f'  ALT {start}-{end} {new_rom}')

    def add_alternatives(self, old_edges: List[Edge]) -> None:
        old_edge_dict = {}
        for old_edge in old_edges:
            old_edge_dict[(old_edge.start, old_edge.end, old_edge.txt)] = old_edge
        for position, old_edge in enumerate(old_edges):
            if old_edge.type.startswith('rom-alt'):
                continue   # not old
            start, end = old_edge.start, old_edge.end
            orig_s = self.s[start:end]
            old_rom = old_edge.txt
            if m := regex.search(r'\bc:([a-z]+)\s+s:([a-z]+)\b', old_edge.type):
                old_rom_core, old_rom_suffix = m.group(1, 2)
            else:
                old_rom_core, old_rom_suffix = None, None
                # print(f'    CORE:{old_rom_core} SUFFIX:{old_rom_suffix}')
            # self.lattice[(start, end)]:
            for rom_rule in self.uroman.rom_rules[orig_s]:
                rom_t = rom_rule['t']
                if self.cand_is_valid(rom_rule, start, end, rom_t):
                    rom_alts = rom_rule['t-alts']
                    rom_end_of_syllable = rom_rule['t-at-end-of-syllable']
                    if (rom_t in [old_rom, old_rom_core]) and rom_alts:
                        for rom_alt in rom_alts:
                            if old_rom_suffix and (rom_t == old_rom_core):
                                rom_alt += old_rom_suffix
                            self.add_new_edge(old_edges, start, end, rom_alt, 'rom-alt', position,
                                              old_edge_dict)
                    if (rom_t == old_rom) and rom_end_of_syllable:
                        self.add_new_edge(old_edges, start, end, rom_t, 'rom-alt2', position, old_edge_dict)
                    if rom_end_of_syllable == old_rom:
                        self.add_new_edge(old_edges, start, end, rom_t, 'rom-alt3', position, old_edge_dict)

    def all_edges(self, start: int, end: int) -> List[Edge]:
        result = []
        for start2 in range(start, end):
            for end2 in sorted(list(self.lattice[(start2, 'right')]), reverse=True):
                if end2 <= end:
                    result.extend(self.lattice[(start2, end2)])
                else:
                    break
        return result

    def best_edge_in_span(self, start: int, end: int, skip_num_edge: bool = False) -> Edge | None:
        edges = self.lattice[(start, end)]
        # if len(edges) >= 2: print('Multi edge', start2, end2, self.s[start2:end2], edges)
        decomp_edge, other_edge, rom_edge = None, None, None
        for edge in edges:
            if isinstance(edge, NumEdge):
                if skip_num_edge:
                    continue
                if edge.active:
                    return edge
            if edge.type.startswith('rom decomp'):
                if decomp_edge is None:
                    decomp_edge = edge  # plan C
            elif regex.match(r'(?:rom|num)', edge.type):
                if rom_edge is None:
                    rom_edge = edge  # plan B
            elif other_edge is None:
                other_edge = edge  # plan D
        return rom_edge or decomp_edge or other_edge

    def best_right_neighbor_edge(self, start: int, skip_num_edge: bool = False) -> Edge | None:
        for end in sorted(list(self.lattice[(start, 'right')]), reverse=True):
            if best_edge := self.best_edge_in_span(start, end, skip_num_edge=skip_num_edge):
                return best_edge
        return None

    def best_left_neighbor_edge(self, end: int, skip_num_edge: bool = False) -> Edge | None:
        for start in sorted(list(self.lattice[(end, 'left')])):
            if best_edge := self.best_edge_in_span(start, end, skip_num_edge=skip_num_edge):
                return best_edge
        return None

    def best_rom_edge_path(self, start: int, end: int, skip_num_edge: bool = False) -> List[Edge]:
        """Finds the best romanization edge path through the romanization lattice, including
        non-romanized pieces such as ASCII and non-ASCII punctuation."""
        result = []
        start2 = start
        while start2 < end:
            if best_edge := self.best_right_neighbor_edge(start2, skip_num_edge=skip_num_edge):
                result.append(best_edge)
                start2 = best_edge.end
            else:  # should not happen
                start2 += 1
        return result

    def find_rom_edge_path_backwards(self, start: int, end: int, min_char: int | None = None,
                                     return_str: bool = False, skip_num_edge: bool = False) -> List[Edge] | str:
        """Finds a partial best path on the left from a start position to provide left contexts for
        romanization rules. Can return a string or a list of edges. Is typically used for a short context,
        as specified by min_char."""
        result_edges = []
        rom = ''
        end2 = end
        while start < end2:
            old_end2 = end2
            if new_edge := self.best_left_neighbor_edge(end2, skip_num_edge=skip_num_edge):
                result_edges = [new_edge] + result_edges
                rom = new_edge.txt + rom
                end2 = new_edge.start
            if min_char and len(rom) >= min_char:
                break
            if old_end2 >= end2:
                end2 -= 1
        if return_str:
            return rom
        else:
            return result_edges

    @staticmethod
    def edge_path_to_surf(edges) -> str:
        result = ''
        for edge in edges:
            result += edge.txt
        return result


# @timer
def main():
    """This function provides a user interface, either using argparse for a command line interface,
    or providing direct function calls.
    First, a uroman object will have to created, loading uroman data (directory must be provided,
    listed as default). This only needs to be done once.
    After that you can romanize from file to file, or just romanize a string."""

    # Compute data_dir based on the location of this executable script.
    src_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(src_dir)
    data_dir = os.path.join(root_dir, "data")
    # print(src_dir, root_dir, data)

    parser = argparse.ArgumentParser()
    parser.add_argument('direct_input', nargs='*', type=str)
    parser.add_argument('--data_dir', type=Path, default=data_dir, help='uroman resource dir')
    parser.add_argument('-i', '--input_filename', type=str, help='default: sys.stdin')
    parser.add_argument('-o', '--output_filename', type=str, help='default: sys.stdout')
    parser.add_argument('-l', '--lcode', type=str, default=None,
                        help='ISO 639-3 language code, e.g. eng')
    # parser.add_argument('-f', '--rom_format', type=RomFormat, default=RomFormat.STR, help:'alt: RomFormat.EDGES')
    parser.add_argument('-f', '--rom_format', type=RomFormat, default=RomFormat.STR,
                        choices=list(RomFormat), help="Output format of romanization. 'edges' provides offsets")
    # The remaining arguments are mostly for development and test
    parser.add_argument('--max_lines', type=int, default=None, help='limit uroman to first n lines')
    parser.add_argument('--load_log', action='count', default=0, help='report load stats (boolean)')
    parser.add_argument('--test', action='count', default=0, help='perform/display a few tests')
    parser.add_argument('-d', '--decode_unicode', action='count', default=0,
                        help='decodes Unicode escape notation, e.g. \\u03B4 to δ')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--rebuild_ud_props', action='count', default=0,
                        help='rebuild UnicodeDataProps files (for development mode only)')
    parser.add_argument('--rebuild_num_props', action='count', default=0,
                        help='rebuild NumProps file (for development mode only)')
    parser.add_argument('-c', '--cache_size', type=int, default=DEFAULT_ROM_MAX_CACHE_SIZE,
                        help='for speed')
    parser.add_argument('--silent', action='count', default=0, help='suppress ... progress')
    parser.add_argument('-a', '--ablation', type=str, default='', help='for development mode: nocap')
    parser.add_argument('--stats', action='count', default=0, help='for development mode: numbers')
    parser.add_argument('--ignore_args', action='count', default=0, help='for usage illustration only')
    parser.add_argument(PROFILE_FLAG, type=argparse.FileType('w', encoding='utf-8', errors='ignore'),
                        default=None, metavar='PROFILE-FILENAME', help='(optional output for performance analysis)')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__} last modified: {last_mod_date}')
    args = parser.parse_args()
    # copy selected (minor) args from argparse.Namespace to dict
    args_dict = {'rom_format': args.rom_format, 'load_log': args.load_log, 'test': args.test, 'stats': args.stats,
                 'cache_size': args.cache_size, 'max_lines': args.max_lines, 'verbose': args.verbose,
                 'rebuild_ud_props': args.rebuild_ud_props, 'rebuild_num_props': args.rebuild_num_props,
                 'ablation': args.ablation, 'silent': args.silent, 'decode_unicode': args.decode_unicode}
    pr = None
    if args.profile:
        gc.enable()
        gc.set_debug(gc.DEBUG_STATS)
        gc.set_debug(gc.DEBUG_LEAK)
        pr = cProfile.Profile()
        pr.enable()
    # noinspection SpellCheckingInspection
    '''Sample calls:
    uroman.py --help
    uroman.py -i ../test/multi-script.txt -o ../test/multi-script-out2.txt
    uroman.py  < ../test/multi-script.txt  > ../test/multi-script-out2.txt
    uroman.py Игорь
    uroman.py Игорь --lcode ukr
    uroman.py ألاسكا 서울 Καλιφόρνια
    uroman.py ちょっとまってください -f edges
    uroman.py "महात्मा गांधी" -f lattice
    uroman.py สวัสดี --load_log
    uroman.py --test
    uroman.py --ignore_args
    uroman.py Բարեւ -o ../test/tmp-out.txt -f edges
    # In double input cases such as in the line below,
    # the input-file's romanization is sent to stdout, while the direct-input romanization is sent to stderr
    uroman.py ⴰⵣⵓⵍ -i ../test/multi-script.txt > ../test/multi-script-out2.txt
        '''

    if args.ignore_args:
        # minimal calls
        uroman = Uroman(args.data_dir)
        s, s2, s3, s4 = 'Игорь', 'ちょっとまってください', 'ka‍n‍ne', 'महात्मा गांधी  '
        print(s, uroman.romanize_string(s))
        print(s, uroman.romanize_string(s, lcode='ukr'))
        print(s2, Edge.json_str(uroman.romanize_string(s2, rom_format=RomFormat.EDGES)))
        print(s3, Edge.json_str(uroman.romanize_string(s3, rom_format=RomFormat.EDGES)))
        print(s4, Edge.json_str(uroman.romanize_string(s4, rom_format=RomFormat.LATTICE)))
        # Note that ../test/multi-script.txt has several lines starting with ::lcode eng etc.
        # This allows users to select specific language codes to specific lines, overwriting the overall --lcodes
        uroman.romanize_file(input_filename='../test/multi-script.txt',
                             output_filename='../test/multi-script-out3.txt')
    else:
        # build a Uroman object (once for many applications and different scripts and languages)
        # uroman = Uroman(args.data_dir, load_log=args.load_log, rebuild_ud_props=args.rebuild_ud_props,
        #                 rebuild_num_props=args.rebuild_num_props)
        uroman = Uroman(args.data_dir, **args_dict)
        romanize_file_p = (args.input_filename or args.output_filename
                           or not (args.direct_input or args.test or args.ignore_args
                                   or args.rebuild_ud_props or args.rebuild_num_props))
        # Romanize any positional arguments, interpreted as strings to be romanized.
        for s in args.direct_input:
            result = uroman.romanize_string(s.rstrip('\n'), lcode=args.lcode, **args_dict)
            result_json = Edge.json_str(result)
            if romanize_file_p:
                # input from both file/stdin (to file/stdout) and direct-input (to stderr)
                if args.input_filename:
                    sys.stderr.write(result_json + '\n')
                # input from direct-input (but not from file/stdin) to stdout
                # else pass
            # no file/stdin or file/stdout, so we write romanization of direct-input to stdout
            else:
                print(result_json)
        # If provided, apply romanization to an entire file.
        if romanize_file_p:
            uroman.romanize_file(args.input_filename, args.output_filename, lcode=args.lcode,
                                 direct_input=args.direct_input, **args_dict)
        if args.test:
            uroman.test_output_of_selected_scripts_and_rom_rules()
            uroman.test_romanization()
        if uroman.stats and args.stats:
            stats100 = {k: uroman.stats[k] for k in list(dict(uroman.stats))[:100]}
            sys.stderr.write(f'Stats: {stats100} ...\n')
    if args.profile:
        if pr:
            pr.disable()
            ps = pstats.Stats(pr, stream=args.profile).sort_stats(pstats.SortKey.TIME)
            ps.print_stats()
        print(gc.get_stats())


if __name__ == "__main__":
    main()
