from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

from lyric_tool.models import AsrSegment, AsrWord


@unittest.skipUnless(importlib.util.find_spec('fastapi') and importlib.util.find_spec('httpx'),
                     'Install fastapi, python-multipart and httpx for API integration tests')
class ApiPrecisionTests(unittest.TestCase):
    def test_rap_request_keeps_acoustic_time_when_lrc_is_wrong(self):
        import api
        from fastapi.testclient import TestClient

        segments = [
            AsrSegment('silver stars', 1000, 2000, .98,
                       [AsrWord('silver', 1123, 1400, .98), AsrWord('stars', 1500, 1900, .98)]),
            AsrSegment('we walk home', 3000, 4500, .98,
                       [AsrWord('we', 3123, 3300, .98), AsrWord('walk', 3500, 3800, .98),
                        AsrWord('home', 4000, 4400, .98)]),
        ]
        with patch('api.run_transcription', return_value=segments):
            response = TestClient(api.app).post('/api/align',
                files={'audio': ('test.wav', b'fixture', 'audio/wav')},
                data={'lyrics': 'silver stars\nwe walk home', 'mode': 'rap', 'language': 'en',
                      'imported_lrc': '[00:00.62]silver stars\n[00:02.62]we walk home'})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual([line['start_ms'] for line in body['lines']], [1123, 3123])
        self.assertEqual(body['global_offset_ms'], 503)
        self.assertEqual(body['issue_count'], len(body['issues']))
        self.assertEqual(body['lrc'], '[00:01.12]silver stars\n[00:03.12]we walk home')


if __name__ == '__main__':
    unittest.main()
