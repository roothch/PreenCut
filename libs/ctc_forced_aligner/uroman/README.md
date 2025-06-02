# uroman

*uroman* is a *universal romanizer*. It converts text in any script to the standard Latin alphabet.<br>
&nbsp;&nbsp;&nbsp;&nbsp;Example (Greek): Νεπάλ → Nepal<br>
&nbsp;&nbsp;&nbsp;&nbsp;Example (Hindi):&nbsp; नेपाल → nepaal<br>
&nbsp;&nbsp;&nbsp;&nbsp;Example (Urdu):&nbsp; نیپال → nypal<br>
&nbsp;&nbsp;&nbsp;&nbsp;Example (Chinese): 三万一 → 31000

* *uroman* enables the application of string-similarity metrics to texts from different scripts without the need and complexity of an intermediate phonetic representation.
* *uroman* converts digital numbers in various scripts to Western Arabic numerals.
* *uroman* uses m-to-n character mappings, context, and a user-provided language code (optional), i.e. *uroman* does not just replace characters one by one.

New Python version: 1.3.1 (released in June 24, 2024)<br>
Last Perl version: 1.2.8 (released April 23, 2021)<br>
Author: Ulf Hermjakob, USC Information Sciences Institute  

## (New) Python version

#### Installation
```bash
python -m pip install uroman
```

### Command Line Interface (CLI)
#### Examples
<sup>Note: Directories _text_ and _test_ are under _uroman_'s root directory on GitHub.</sup>
```bash
uroman.py "Игорь Стравинский"
uroman.py Игорь -l ukr
uroman.py Ντέιβις Καπ -l ell
uroman.py "\u03C0\u03B9" -d
uroman.py -l hin -i text/hin.txt
uroman.py -l fas -i text/fas.txt -o text/fas-rom.jsonl -f edges
uroman.py < test/multi-script.txt > test/multi-script.uroman.txt
uroman.py -h
```

#### *uroman.py* &nbsp; Argument Structure Highlights 
<table>
  <tr><td><i>Direct inputs (zero&nbsp;or&nbsp;more)</i></td><td>such as ‘Игорь Стравинский’ and ‘Ντέιβις’ above.</td></tr>
  <tr><td>-l<br>--lcode</td><td>language code according to <a href="https://en.wikipedia.org/wiki/List_of_ISO_639-3_codes" target="_LCODE">ISO-639-3</a>, e.g. <i>-l ukr</i> for Ukrainian, <i>-l hin</i> for Hindi, <i>-l fas</i> for Persian</td></tr>
  <tr><td>-i<br>--input_filename</td><td>alternative:&nbsp;<i>stdin</i><br>Note: If both <i>direct inputs</i> and <i>input_filename</i> are given, the romanization results for <i>direct inputs</i> will be written to <i>stderr</i>.</td></tr>
  <tr><td width="200">-o<br><nobr>--output_filename</nobr></td><td>alternative: <i>stdout</i></td></tr>
  <tr><td>-f<br>--rom_format</td><td>Output format choices:
        <ul>
           <li> -f str &nbsp;&nbsp;&nbsp;&nbsp;&nbsp (best string, default, output format: string)
           <li> -f edges (best edges, includes offset information, output format: JSONL)
           <li> -f alts &nbsp;&nbsp;&nbsp;&nbsp; (lattice including alternative edges, output format: JSONL)
           <li> -f lattice (lattice including alternative and superseded edges, output format: JSONL)
        </ul></td></tr>
  <tr><td>-d<br>--decode_unicode</td><td>Decode Unicode escape sequences such as ‘\u03C0\u03B9’ to ‘πι’ which in turn will be romanized to ‘pi’. This is useful for input formats such as JSON.</td></tr>
  <tr><td>-h<br>--help</td><td>Use this option to see the full argument structure with all options.</td></tr>
</table>

### Using _uroman_ inside Python
#### Examples

```bash
import uroman

uroman = Uroman()
print(uroman.romanize_string('Игорь Стравинский'))
print(uroman.romanize_string('Игорь', lcode='ukr'))
uroman.romanize_file(input_filename='test/multi-script.txt',
                     output_filename='test/multi-script.uroman.jsonl',
                     rom_format=RomFormat.LATTICE)
```

#### Methods
__`uroman = Uroman(data_dir)`__

This constructor method loads data needed for the romanization of different languages.
This constructor call might take about a second (real time) to load all of the romanization data, but it is necessary only once for multiple subsequent romanization calls.
<table>
  <tr><td>data_dir</td><td>data directory (optional, default: standard uroman data directory)</td></tr>
</table>

<hr>

__`uroman.romanize_string(s, lcode, rom_format)`__

This method takes a string <i>s</i> and returns its romanization in the format according to <i>rom_format</i>: a string (default), or a list of edges.
<table>
  <tr><td>s</td><td>string to be romanized, e.g. "ایران"</td></tr>
  <tr><td>lcode</td><td>language code, optional, a 3-letter code such as 'eng' for English (ISO-639-3)</td></tr>
  <tr><td>rom_format</td><td>Output format choices:
        <ul>
           <li> RomFormat.STR &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(best string, default, output format: string)
           <li> RomFormat.EDGES &nbsp;(best edges, includes offset information, output format: JSONL)
           <li> RomFormat.ALTS &nbsp;&nbsp;&nbsp;&nbsp;(lattice including alternative edges, output format: JSONL)
           <li> RomFormat.LATTICE (lattice including alternative and superseded edges, output format: JSONL)
        </ul>
</table>

<hr>

__`uroman.romanize_file(input_filename, output_filename, lcode)`__

This method romanizes a file <i>input_filename</i> to <i>output_filename</i>.
<table>
  <tr><td>input_filename</td><td>default: stdin&nbsp;(for input_filename value of <i>None</i>)</td></tr>
  <tr><td width="200">output_filename</td><td>default: stdout&nbsp;(for output_filename value of <i>None</i>)</td></tr>
  <tr><td>lcode</td><td>language code (optional), a 3-letter code such as 'eng' for English (ISO-639-3)</td></tr>
</table>

## (Old) Perl Version

### Usage
```bash
$ uroman.pl [-l <lang-code>] [--chart] [--no-cache] < STDIN
       where the optional <lang-code> is a 3-letter languages code, e.g. ara, bel, bul, deu, ell, eng, fas,
            grc, ell, eng, heb, kaz, kir, lav, lit, mkd, mkd2, oss, pnt, pus, rus, srp, srp2, tur, uig, ukr, yid.
       --chart specifies chart output (in JSON format) to represent alternative romanizations.
       --no-cache disables caching.
```
### Examples
<sup>Note: Directories _text_ and _test_ are under _uroman_'s root directory on GitHub.</sup>
```bash
uroman.pl < text/zho.txt
uroman.pl -l tur < text/tur.txt
uroman.pl -l heb --chart < text/heb.txt
uroman.pl < test/multi-script.txt > test/multi-script.uroman-perl.txt
```

Identifying the input as Arabic, Belarusian, Bulgarian, English, German,
Ancient Greek, Modern Greek, Pontic Greek, Hebrew, Kazakh, Kyrgyz, Latvian,
Lithuanian, North Macedonian, Ossetian, Persian, Russian, Serbian, Turkish, 
Ukrainian, Uyghur or Yiddish 
will improve romanization for those languages as some letters in those 
languages have different sound values from other languages using the same script 
(Arabic vs. Persian, Russian vs. Ukrainian, Hebrew vs. Yiddish).
No effect for other languages in this version.

### Bibliography
Ulf Hermjakob, Jonathan May, and Kevin Knight. 2018. Out-of-the-box universal romanization tool uroman. In Proceedings of the 56th Annual Meeting of Association for Computational Linguistics, Demo Track. ACL-2018 Best Demo Paper Award. [Paper in ACL Anthology](https://www.aclweb.org/anthology/P18-4003) | [Poster](https://www.isi.edu/~ulf/papers/poster-uroman-acl2018.pdf) | [BibTex](https://www.aclweb.org/anthology/P18-4003.bib)

### Change History

Changes in version 1.3.0
 * Added Python version.
 * Initial dedicated support for Coptic (Egypt); significantly improved support for Thai; improved support for Khmer, Tibetan and several Indian languages incl. better final schwa deletion.
 * Chinese fractions and percentages.
 * Various small improvements.

Changes in version 1.2.8
 * Updated to Unicode 13.0 (2021), which supports several new scripts (10% larger UnicodeData.txt).
 * Improved support for Georgian.
 * Preserve various symbols (as opposed to mapping to the symbols' names).
 * Various small improvements.

Changes in version 1.2.7
 * Improved support for Pashto.

Changes in version 1.2.6
 * Improved support for Ukrainian, Russian and Ogham (ancient Irish script).
 * Added support for English Braille.
 * Added alternative Romanization for North Macedonian and Serbian (mkd2/srp2)
   reflecting a casual style that many native speakers of those languages use
   when writing text in Latin script, e.g. non-accented single letters (e.g. "s")
   rather than phonetically motivated combinations of letters (e.g. "sh").
 * When a line starts with "::lcode xyz ", the new uroman version will switch to
   that language for that line. This is used for the new reference test file.
 * Various small improvements.

Changes in version 1.2.5
 * Improved support for Armenian and eight languages using Cyrillic scripts.
   -- For Serbian and Macedonian, which are often written in both Cyrillic
      and Latin scripts, uroman will map both official versions to the same
      romanized text, e.g. both "Ниш" and "Niš" will be mapped to "Nish" (which
      properly reflects the pronunciation of the city's name).
      For both Serbian and Macedonian, casual writers often use a simplified
      Latin form without diacritics, e.g. "s" to represent not only Cyrillic "с"
      and Latin "s", but also "ш" or "š", even if this conflates "s" and "sh" and
      other such pairs. The casual romanization can be simulated by using
      alternative uroman language codes "srp2" and "mkd2", which romanize
      both "Ниш" and "Niš" to "Nis" to reflect the casual Latin spelling.
 * Various small improvements.

Changes in version 1.2.4
  * Bug-fix that generated two emtpy lines for each empty line in cache mode.

Changes in version 1.2
 * Run-time improvement based on (1) token-based caching and (2) shortcut 
   romanization (identity) of ASCII strings for default 1-best (non-chart) 
   output. Speed-up by a factor of 10 for Bengali and Uyghur on medium and 
   large size texts.
 * Incremental improvements for Farsi, Amharic, Russian, Hebrew and related
   languages.
 * Richer lattice structure (more alternatives) for "Romanization" of English
   to support better matching to romanizations of other languages.
   Changes output only when --chart option is specified. No change in output for
   default 1-best output, which for ASCII characters is always the input string.

Changes in version 1.1 (major upgrade)
 * Offers chart output (in JSON format) to represent alternative romanizations.
   * Location of first character is defined to be "line: 1, start:0, end:0".
 * Incremental improvements of Hebrew and Greek romanization; Chinese numbers.
 * Improved web-interface (now) at https://uhermjakob.github.io/uroman.html
   * Shows corresponding original and romanization text in red
     when hovering over a text segment.
   * Shows alternative romanizations when hovering over romanized text
     marked by dotted underline.
   * Added right-to-left script detection and improved display for right-to-left
     script text (as determined line by line).
   * On-page support for some scripts that are often not pre-installed on users'
     computers (Burmese, Egyptian, Klingon).

Changes in version 1.0 (major upgrade)
 * Upgraded principal internal data structure from string to lattice.
 * Improvements mostly in vowelization of South and Southeast Asian languages.
 * Vocalic 'r' more consistently treated as vowel (no additional vowel added).
 * Repetition signs (Japanese/Chinese/Thai/Khmer/Lao) are mapped to superscript 2.
 * Japanese Katakana middle dots now mapped to ASCII space.
 * Tibetan intersyllabic mark now mapped to middle dot (U+00B7).
 * Some corrections regarding analysis of Chinese numbers.
 * Many more foreign diacritics and punctuation marks dropped or mapped to ASCII.
 * Zero-width characters dropped, except line/sentence-initial byte order marks.
 * Spaces normalized to ASCII space.
 * Fixed bug that in some cases mapped signs (such as dagger or bullet) to their verbal descriptions.
 * Tested against previous version of uroman with a new uroman visual diff tool.
 * Almost an order of magnitude faster.

Changes in version 0.7 (minor upgrade)
 * Added script uroman-quick.pl for Arabic script languages, incl. Uyghur.
   Much faster, pre-caching mapping of Arabic to Latin characters, simple greedy processing.
   Will not convert material from non-Arabic blocks such as any (somewhat unusual) Cyrillic
   or Chinese characters in Uyghur texts.

Changes in version 0.6 (minor upgrade)
 * Added support for two letter characters used in Uzbek:
   (1) character "ʻ" ("modifier letter turned comma", which modifies preceding "g" and "u" letters)
   (2) character "ʼ" ("modifier letter apostrophe", which Uzbek uses to mark a glottal stop).
   Both are now mapped to "'" (plain ASCII apostrophe).
 * Added support for Uyghur vowel characters such as "ې" (Arabic e) and "ۆ" (Arabic oe)
   even when they are not preceded by "ئ" (yeh with hamza above).
 * Added support for Arabic semicolon "؛", Arabic ligature forms for phrases such as "ﷺ"
   ("sallallahou alayhe wasallam" = "prayer of God be upon him and his family and peace")
 * Added robustness for Arabic letter presentation forms (initial/medial/final/isolated).
   However, it is strongly recommended to normalize any presentation form Arabic letters
   to their non-presentation form before calling uroman.
 * Added force flush directive ($|=1;).

Changes in version 0.5 (minor upgrade)
 * Improvements for Uyghur (make sure to use language option: -l uig)

Changes in version 0.4 (minor upgrade)
 * Improvements for Thai (special cases for vowel/consonant reordering, e.g. for "sara o"; dropped some aspiration 'h's)
 * Minor change for Arabic (added "alef+fathatan" = "an")

New features in version 0.3
 * Covers Mandarin (Chinese)
 * Improved romanization for numerous languages
 * Preserves capitalization (e.g. from Latin, Cyrillic, Greek scripts)
 * Maps from native digits to Western numbers
 * Faster for South Asian languages

### Other features
 * Web interface (old Perl): https://uhermjakob.github.io/uroman.html
 * Vowelization is provided when locally computable, e.g. for many South Asian languages and Tibetan.

### Limitations
 * The current version of uroman has a few limitations, some of which we plan to address in future versions.
   For Japanese, *uroman* currently romanizes hiragana and katakana as expected, but kanji are interpreted as Chinese characters and romanized as such. 
   For Egyptian hieroglyphs, only single-sound phonetic characters and numbers are currently romanized. 
   For Linear B, only phonetic syllabic characters are romanized. 
   For some other extinct scripts such as cuneiform, no romanization is provided.
 * A romanizer is not a full transliterator. For example, this version of
   uroman does not vowelize text that lacks explicit vowelization such as
   normal text in Arabic and Hebrew (without diacritics/points).

### Acknowledgments
Earlier versions of this tool were based upon work supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via contract # FA8650-17-C-9116, and by research sponsored by Air Force Research Laboratory (AFRL) under agreement number FA8750-19-1-1000. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, Air Force Laboratory, DARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.
