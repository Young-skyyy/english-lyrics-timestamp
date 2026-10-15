"""Run an actual audio file through the API without starting a web server."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--lyrics', type=Path, required=True, help='UTF-8 lyrics without a song-title header')
    parser.add_argument('--mode', choices=('fast', 'rap', 'quality'), default='rap')
    parser.add_argument('--output-dir', type=Path, default=Path('validation-output'))
    parser.add_argument('--duration-ms', type=int, default=0, help='Optional known audio duration for bounds checking')
    args = parser.parse_args()

    from fastapi.testclient import TestClient
    from api import app

    started = time.perf_counter()
    with TestClient(app) as client, args.audio.open('rb') as source:
        response = client.post('/api/align',
            files={'audio': (args.audio.name, source, 'application/octet-stream')},
            data={'lyrics': args.lyrics.read_text(encoding='utf-8-sig'),
                  'language': 'en', 'mode': args.mode, 'duration_ms': str(args.duration_ms)})
    response.raise_for_status()
    body = response.json()
    timed = [line for line in body['lines'] if line['start_ms'] is not None]
    summary = {
        'mode': args.mode, 'algorithm_version': body['algorithm_version'],
        'elapsed_s': round(time.perf_counter() - started, 2),
        'lyric_lines': len(body['lines']), 'timed_lines': len(timed),
        'untimed_lines': len(body['lines']) - len(timed), 'coverage_score': body['coverage_score'],
        'methods': dict(Counter(line['match_method'] for line in body['lines'])),
        'non_monotonic_lines': [right['line_index'] for left, right in zip(timed, timed[1:])
                                if right['start_ms'] < left['start_ms']],
        'invalid_ranges': [line['line_index'] for line in timed
                           if line['end_ms'] is None or not 0 <= line['start_ms'] <= line['end_ms']],
        'out_of_bounds_lines': [line['line_index'] for line in timed
                                if args.duration_ms and line['end_ms'] is not None
                                and line['end_ms'] > args.duration_ms],
        'note': 'Coverage and confidence are diagnostics, not measurements against human timing labels.',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{args.audio.stem}.{args.mode}'
    for suffix, value in (('.json', json.dumps(body, ensure_ascii=False, indent=2)),
                          ('.lrc', body['lrc']),
                          ('.summary.json', json.dumps(summary, ensure_ascii=False, indent=2))):
        (args.output_dir / f'{stem}{suffix}').write_text(value, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 1 if any(summary[key] for key in ('non_monotonic_lines', 'invalid_ranges', 'out_of_bounds_lines')) else 0


if __name__ == '__main__':
    raise SystemExit(main())
