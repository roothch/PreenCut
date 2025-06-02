uroman version 1.2.8
Release date: April 23, 2021
Author: Ulf Hermjakob, USC Information Sciences Institute

uroman is a universal romanizer. It converts text in any script to the Latin alphabet.

Usage: uroman.pl [-l <lang-code>] [--chart] [--no-cache] < STDIN
       where the optional <lang-code> is a 3-letter languages code, e.g. ara, bel, bul, deu, ell, eng, fas,
            grc, ell, eng, heb, kaz, kir, lav, lit, mkd, mkd2, oss, pnt, pus, rus, srp, srp2, tur, uig, ukr, yid.
       --chart specifies chart output (in JSON format) to represent alternative romanizations.
       --no-cache disables caching.
Examples: bin/uroman.pl < text/zho.txt
          bin/uroman.pl -l tur < text/tur.txt
	  bin/uroman.pl -l heb --chart < text/heb.txt
	  bin/uroman.pl < test/multi-script.txt > test/multi-script.uroman.txt

Identifying the input as Arabic, Belarusian, Bulgarian, English, Farsi, German, 
Ancient Greek, Modern Greek, Pontic Greek, Hebrew, Kazakh, Kyrgyz, Latvian, 
Lithuanian, North Macedonian, Russian, Serbian, Turkish, Ukrainian, Uyghur or Yiddish 
will improve romanization for those languages as some letters in those languages 
have different sound values from other languages using the same script.
No effect for other languages in this version.

Bibliography: Ulf Hermjakob, Jonathan May, and Kevin Knight. 2018. Out-of-the-box universal romanization tool uroman. In Proceedings of the 56th Annual Meeting of Association for Computational Linguistics, Demo Track. [Best Demo Paper Award]

Changes in version 1.2.8
 * Improved support for Georgian.
 * Updated UnicodeData.txt to version 13 (2021) with several new scripts (10% larger).
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
 * Added support for Tifinagh (a script used for Berber languages).
 * Bug-fix that generated two emtpy lines for each empty line in cache mode.
Changes in version 1.2.3
 * Exclude emojis, dingbats, many other pictographs from being romanized (e.g. to "face")
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
   -- Location of first character is defined to be "line: 1, start:0, end:0".
 * Incremental improvements of Hebrew and Greek romanization; Chinese numbers.
 * Improved web-interface at http://www.isi.edu/~ulf/uroman.html
   -- Shows corresponding original and romanization text in red
      when hovering over a text segment.
   -- Shows alternative romanizations when hovering over romanized text
      marked by dotted underline.
   -- Added right-to-left script detection and improved display for right-to-left
      script text (as determined line by line).
   -- On-page support for some scripts that are often not pre-installed on users'
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

Other features
 * Web interface: http://www.isi.edu/~ulf/uroman.html
 * Vowelization is provided when locally computable, e.g. for many South Asian
   languages and Tibetan.

Limitations
 * This version of uroman assumes all CJK ideographs to be Mandarin (Chinese).
   This means that Japanese kanji are incorrectly romanized; however, Japanese
   hiragana and katakana are properly romanized.
 * A romanizer is not a full transliterator. For example, this version of
   uroman does not vowelize text that lacks explicit vowelization such as
   normal text in Arabic and Hebrew (without diacritics/points).

