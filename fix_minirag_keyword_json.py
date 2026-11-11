#!/usr/bin/env python3
"""Apply a minimal MiniRAG HF keyword-output normalization fix in-place.

Run from the project root:
    python fix_minirag_keyword_json.py

Backs up the target before changing it; only edits the Hugging Face portable
completion branch, never MiniRAG's retrieval or experiment evaluation code.
"""
from pathlib import Path
import ast
import shutil
import sys

TARGET = Path('run_experiment_mps_memory_fixed.py')
MARKER = '# MiniRAG HF keyword JSON compatibility (validated local fix)'
ANCHOR = '            keyword_extraction = kwargs.pop("keyword_extraction", None)'
END = '        return portable_complete, model_name'

INSERT = '''            # MiniRAG HF keyword JSON compatibility (validated local fix)
            # Official MiniRAG expects a dict; small instruction LLMs may emit
            # an array of dicts instead. Normalize structure only, not content.
            if ("answer_type_keywords" in str(prompt)
                    and "entities_from_query" in str(prompt)):
                import json as _mini_json
                try:
                    _mini_parsed = _mini_json.loads(response.strip())
                except _mini_json.JSONDecodeError:
                    # Retain the existing MiniRAG parser's handling of other
                    # output shapes; this fix targets valid JSON arrays only.
                    _mini_parsed = None
                if isinstance(_mini_parsed, list):
                    _mini_fields = ("answer_type_keywords", "entities_from_query")
                    if not _mini_parsed or not all(isinstance(_item, dict) for _item in _mini_parsed):
                        raise ValueError("MiniRAG keyword output is a JSON list, but not a nonempty list of objects")
                    _mini_combined = {field: [] for field in _mini_fields}
                    _mini_found = set()
                    for _item in _mini_parsed:
                        for _field in _mini_fields:
                            if _field not in _item:
                                continue
                            _mini_found.add(_field)
                            _values = _item[_field]
                            if isinstance(_values, str):
                                _values = [_values] if _values.strip() else []
                            if not isinstance(_values, list) or not all(isinstance(v, str) for v in _values):
                                raise ValueError(f"MiniRAG keyword field {_field} has invalid type")
                            _mini_combined[_field].extend(_values)
                    if not _mini_found:
                        raise ValueError("MiniRAG keyword JSON list has no recognized keyword fields")
                    # No guessed keywords; preserve original extracted strings.
                    response = _mini_json.dumps(
                        {field: list(dict.fromkeys(_mini_combined[field])) for field in _mini_fields},
                        ensure_ascii=False,
                    )
                    print("[MiniRAG HF FORMAT] normalized keyword JSON array to object", flush=True)
'''

def main():
    if not TARGET.is_file():
        sys.exit(f"Not found: {TARGET}. Run this script in the project root.")
    source = TARGET.read_text(encoding='utf-8')
    if MARKER in source:
        print('Already patched. No changes made.')
        return
    if source.count(END) != 1:
        sys.exit('Could not uniquely locate portable_complete branch. No changes made.')
    before, after = source.split(END, 1)
    if before.count(ANCHOR) != 1:
        sys.exit('Could not uniquely locate keyword_extraction assignment. No changes made.')
    patched = before.replace(ANCHOR, INSERT + ANCHOR, 1) + END + after
    try:
        ast.parse(patched, filename=str(TARGET))
    except SyntaxError as ex:
        sys.exit(f'Patch failed syntax validation ({ex}). No changes made.')
    backup = TARGET.with_suffix(TARGET.suffix + '.before_keyword_fix.bak')
    if backup.exists():
        sys.exit(f'Backup already exists: {backup}. Move it first; no changes made.')
    shutil.copy2(TARGET, backup)
    TARGET.write_text(patched, encoding='utf-8')
    print(f'Patched {TARGET}')
    print(f'Backup  {backup}')
    print('Re-run the same smoke command to verify.')

if __name__ == '__main__':
    main()
