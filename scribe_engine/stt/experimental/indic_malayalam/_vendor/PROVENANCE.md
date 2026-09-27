# Vendored file provenance

## processor.py

- Source: https://github.com/VarunGumma/IndicTransToolkit
- Commit: `a95da3a008` (v1.0.2 era, Aug 2024, package name `IndicTransTokenizer`)
- Path at that commit: `IndicTransTokenizer/processor.py`
- License of upstream repo: MIT
- Reason for vendoring: the current toolkit (v1.1.1, 2025) compiles a Cython
  extension (`processor.pyx`) and publishes no Windows wheels; building from
  source requires MSVC C++ build tools, which the demo/dev machine lacks.
  The v1.0.2-era processor is pure Python and functionally equivalent for
  inference preprocessing (normalization, numeral transliteration, script
  conversion, detokenization).
- Local modifications: NONE (vendored verbatim; a small `.py` header comment
  was not added to avoid any drift — this file documents provenance instead).

The processor requires: `sacremoses`, `indic-nlp-library-itt` (imported as
`indicnlp`), `tqdm` — all MIT/pure-Python, pinned in the experiment README.
