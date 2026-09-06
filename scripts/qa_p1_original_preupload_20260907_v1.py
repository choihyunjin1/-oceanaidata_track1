"""Read-only independent preupload checks; writes only a new aggregate receipt."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(root, keys_file):
    terminal = read(root / 'terminal.json')
    contract = read(root / 'contract.json')
    tree = read(root / '03_model/tree/training-result.json')
    ms_root = root / '03_model/mstcn'
    ms = read(ms_root / 'training-result.json')
    msqa = read(ms_root / 'fresh-process-replay.json')
    treeqa = read(root / '03_model/tree/qa.json')
    answer = root / '05_answer/P1_submission.csv'
    receipt = read(root / '06_docs/P1_submission.csv.json')
    checks = {
        'terminal_complete': terminal['status'] == 'TRAIN_TO_ANSWER_COMPLETE',
        'seven_fits': terminal['fits'] == tree['fits'] + ms['completed_fits'] == 7,
        'within_six_hours': 0 < terminal['runtime_seconds'] <= 21600,
        'tree_qa': treeqa['status'] == 'PASS' and all(treeqa['checks'].values()),
        'ms_qa': msqa['status'] == 'PASS' and all(msqa['checks'].values()),
        'ms_qa_receipt_pin': msqa['training_result_sha256'] == sha(ms_root / 'training-result.json'),
        'train_only': tree['official_rows'] == ms['official_access_rows'] == 0,
        'old_input_reads': tree['old_model_cache_prediction_reads'] == ms['old_models_cache_answer_reads'] == 0,
        'answer_receipt_pin': sha(answer) == receipt['sha256'],
        'historical_answer_exact': sha(answer) == contract['historical_answer_sha256'],
    }
    for relative, expected in read(root / 'source-manifest.json')['files'].items():
        path = (root / relative).resolve()
        checks['source:' + relative] = path.is_relative_to(root) and sha(path) == expected
    for relative, expected in ms['files_sha256'].items():
        path = (ms_root / relative).resolve()
        checks['ms:' + relative] = path.is_relative_to(ms_root) and sha(path) == expected
    for name in ['O', 'B']:
        checks[name + '_model_pin'] = sha(root / f'03_model/tree/{name}.joblib') == tree[name + '_sha256']
    keys = ['station', 'year', 'layer', 'time']
    frame = pd.read_csv(answer)
    reference = pd.read_csv(keys_file, usecols=keys)
    checks.update({
        'columns': frame.columns.tolist() == keys + ['label'],
        'rows': len(frame) == len(reference) == 169011,
        'keys_order': frame[keys].equals(reference[keys]),
        'duplicate_keys_zero': not frame.duplicated(keys).any(),
        'missing_zero': not frame.isna().any().any(),
        'binary_finite': bool(np.isfinite(frame.label).all() and frame.label.isin([0, 1]).all()),
    })
    return {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
            'check_count': len(checks), 'answer': str(answer), 'sha256': sha(answer),
            'rows': len(frame), 'positive': int(frame.label.sum()), 'fits': 7,
            'runtime_seconds': terminal['runtime_seconds'], 'new_fits': 0,
            'hidden_truth_reads': 0, 'scope': 'Hash/receipt/schema QA, not organizer acceptance or new OOF quality'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--keys-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root.resolve(), args.keys_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}))
    if result['status'] != 'PASS':
        raise SystemExit(1)
