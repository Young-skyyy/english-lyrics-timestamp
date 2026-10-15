from io import BytesIO
import importlib.util
import unittest
import zipfile

from lyric_tool.lyric_import import MAX_LYRIC_BYTES, import_lyrics_file


def zipped(entries):
    data = BytesIO()
    with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, text in entries.items():
            archive.writestr(name, text)
    return data.getvalue()


class LyricImportTests(unittest.TestCase):
    def test_note_zip_ignores_internal_note_and_removes_explicit_heading(self):
        data = zipped({'id1_note.txt': 'Song - Artist (Full Explicit Lyrics)\n[Intro]\nsilver stars\nwe walk home',
                       '.1.note': b'internal note data'})
        result = import_lyrics_file('note.zip', data)
        self.assertEqual(result['line_count'], 2)
        self.assertEqual(result['text'], '[Intro]\nsilver stars\nwe walk home')

    def test_encoding_detection(self):
        for encoding in ('utf-8-sig', 'utf-16', 'gb18030'):
            with self.subTest(encoding=encoding):
                self.assertEqual(import_lyrics_file('lyrics.txt', '星光\n回家'.encode(encoding))['line_count'], 2)

    def test_lrc_inside_txt_keeps_original_timestamps(self):
        text = '[00:01.123]silver stars\n[00:03.456]we walk home'
        result = import_lyrics_file('lyrics.txt', text.encode())
        self.assertEqual(result['format'], 'lrc')
        self.assertEqual(result['text'], text)

    def test_srt_and_vtt_become_timed_lyrics(self):
        for name, text in (
            ('song.srt', '1\n00:00:01,123 --> 00:00:03,000\nsilver\nstars\n\n2\n00:00:03,456 --> 00:00:05,000\nwe walk home'),
            ('song.vtt', 'WEBVTT\n\ncue1\n00:01.123 --> 00:03.000 align:start\n<b>silver stars</b>\n\n00:03.456 --> 00:05.000\nwe walk home'),
        ):
            with self.subTest(name=name):
                result = import_lyrics_file(name, text.encode())
                self.assertEqual(result['text'], '[00:01.123]silver stars\n[00:03.456]we walk home')

    def test_ambiguous_zip_does_not_silently_choose_a_song(self):
        with self.assertRaisesRegex(ValueError, '多个'):
            import_lyrics_file('songs.zip', zipped({'a.txt': 'one', 'b.lrc': '[00:01]two'}))

    def test_zip_size_and_invalid_format(self):
        with self.assertRaisesRegex(ValueError, '2 MB'):
            import_lyrics_file('big.zip', zipped({'lyrics.txt': 'x' * (MAX_LYRIC_BYTES + 1)}))
        for name, data in (('a.zip', b'not zip'), ('a.exe', b'not lyrics'), ('a.txt', b'')):
            with self.subTest(name=name), self.assertRaises(ValueError):
                import_lyrics_file(name, data)

    @unittest.skipUnless(importlib.util.find_spec('fastapi') and importlib.util.find_spec('httpx'), 'API test dependencies required')
    def test_import_endpoint_accepts_zip_without_loading_speech_model(self):
        from fastapi.testclient import TestClient
        from api import app
        response = TestClient(app).post('/api/lyrics/import',
            files={'file': ('note.zip', zipped({'a.txt': 'silver stars\nwe walk home'}), 'application/zip')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['line_count'], 2)
        bad = TestClient(app).post('/api/lyrics/import', files={'file': ('bad.zip', b'not zip')})
        self.assertEqual(bad.status_code, 400)


if __name__ == '__main__':
    unittest.main()
