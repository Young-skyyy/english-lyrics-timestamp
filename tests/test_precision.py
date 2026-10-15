from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lyric_tool import AlignmentPipeline
from lyric_tool.acoustic_aligner import WhisperXAcousticAligner
from lyric_tool.exporters import format_lrc
from lyric_tool.isolated_transcriber import _worker_main
from lyric_tool.matcher import asr_words, align_lyric_lines_to_word_windows
from lyric_tool.models import AsrSegment, AsrWord
from lyric_tool.parser import parse_lyrics
from lyric_tool.transcriber import FasterWhisperTranscriber


def measured(text, start_ms):
    tokens = text.split()
    words = [AsrWord(token, start_ms + i * 400, start_ms + i * 400 + 250, .98)
             for i, token in enumerate(tokens)]
    return AsrSegment(text, start_ms - 100, words[-1].end_ms + 100, .98, words)


class PrecisionTests(unittest.TestCase):
    def test_missing_lines_do_not_cross_next_measured_time(self):
        result = AlignmentPipeline().run('silver stars\nzzzzzzzz zzzzzzzz\nwe walk home',
            [measured('silver stars', 1000), measured('we walk home', 2000)])
        self.assertIsNone(result.lines[1].start_ms)
        self.assertIsNone(result.lines[1].end_ms)
        self.assertAlmostEqual(result.coverage_score, 2 / 3, places=4)
        self.assertEqual(result.lines[2].start_ms, 2000)
        self.assertTrue(any(issue.issue_type == 'missing_match' for issue in result.issues))

    def test_missing_outro_is_not_extrapolated_past_audio(self):
        result = AlignmentPipeline().run('silver stars\nzzzzzzzz zzzzzzzz\nqqqqqqqq qqqqqqqq',
            [measured('silver stars', 1000)])
        self.assertEqual([line.start_ms for line in result.lines], [1000, None, None])
        self.assertEqual(format_lrc(result), '[00:01.00]silver stars')

    def test_imported_provisional_times_are_not_acoustic_coverage(self):
        result = AlignmentPipeline().run('silver stars\nzzzzzzzz zzzzzzzz\nwe walk home',
            [measured('silver stars', 1000), measured('we walk home', 3000)],
            '[00:01.00]silver stars\n[00:02.00]zzzzzzzz zzzzzzzz\n[00:03.00]we walk home')
        self.assertEqual(result.lines[1].match_method, 'lrc_provisional')
        self.assertAlmostEqual(result.coverage_score, 2 / 3, places=4)

    def test_acoustic_word_uncertainty_limits_timing_confidence(self):
        segment = measured('silver stars', 1000)
        for word in segment.words:
            word.confidence = .3
        result = AlignmentPipeline().run('silver stars', [segment])
        self.assertLessEqual(result.lines[0].timing_confidence, .3)
        self.assertEqual(result.lines[0].review_status, 'needs_review')

    def test_single_line_can_span_multiple_segments(self):
        result = AlignmentPipeline().run('silver stars are shining bright',
            [AsrSegment('silver stars are', 1000, 2000, .98),
             AsrSegment('shining bright', 2100, 3500, .98)])
        self.assertEqual(len(result.lines), 1)
        self.assertEqual((result.lines[0].start_ms, result.lines[0].end_ms), (1000, 3500))

    def test_single_line_uses_matching_words_within_segment(self):
        result = AlignmentPipeline().run('silver stars', [measured('hello silver stars goodbye', 1000)])
        self.assertEqual((result.lines[0].start_ms, result.lines[0].end_ms), (1400, 2050))

    def test_ctc_preserves_overlapping_search_windows(self):
        windows = WhisperXAcousticAligner._clean_windows([
            {'text': 'silver', 'start': 1., 'end': 3.},
            {'text': 'stars', 'start': 2., 'end': 4.},
        ])
        self.assertEqual(windows[1]['start'], 2.)

    def test_ctc_retries_when_first_alignment_has_no_words(self):
        whisperx = Mock()
        whisperx.load_align_model.return_value = (object(), {})
        whisperx.align.side_effect = [
            {'segments': []},
            {'segments': [{'text': 'hello', 'start': 1.1, 'end': 1.4,
                           'words': [{'word': 'hello', 'start': 1.123, 'end': 1.4, 'score': .98}]}]},
        ]
        aligner = WhisperXAcousticAligner(whisperx)
        result = aligner.align([0] * 32000, [{'text': 'hello', 'start': 1., 'end': 1.5}])
        self.assertEqual(whisperx.align.call_count, 2)
        self.assertEqual(result[0].words[0].start_ms, 1123)

    def test_faster_whisper_native_word_objects_in_both_conversion_paths(self):
        native = SimpleNamespace(
            text=' hello world', start=1., end=2., avg_logprob=-.01,
            words=[SimpleNamespace(word=' hello', start=1.123, end=1.456, probability=.98),
                   SimpleNamespace(word=' world', start=1.6, end=1.99, probability=.97)],
        )
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        transcriber.model = Mock()
        transcriber.model.transcribe.return_value = (iter([native]), SimpleNamespace())
        for results in (transcriber.transcribe('test.wav'), transcriber._convert_segments([native])):
            self.assertEqual([(w.text, w.start_ms, w.end_ms) for w in results[0].words],
                             [('hello', 1123, 1456), ('world', 1600, 1990)])

    def test_no_fabricated_word_times(self):
        self.assertEqual(asr_words([AsrSegment('hello world', 1000, 9000, .98)]), [])

    def test_lyric_spanning_segments_keeps_full_end(self):
        result = AlignmentPipeline().run(
            'silver stars are shining bright\nwe walk home',
            [AsrSegment('silver stars are', 1000, 2000, .98),
             AsrSegment('shining bright', 2100, 3500, .98),
             AsrSegment('we walk home', 4000, 5000, .98)],
        )
        self.assertEqual((result.lines[0].start_ms, result.lines[0].end_ms), (1000, 3500))
        self.assertEqual(result.lines[0].match_method, 'asr_segment_window')
        self.assertLessEqual(result.lines[0].timing_confidence, .5)

    def test_lines_sharing_segment_keep_independent_word_times(self):
        segment = measured('hello world silver stars', 1123)
        result = AlignmentPipeline().run('hello world\nsilver stars', [segment])
        self.assertEqual([(line.start_ms, line.end_ms) for line in result.lines],
                         [(1123, 1773), (1923, 2573)])
        self.assertTrue(all(line.sequence_score == 1 for line in result.lines))

    def test_long_spoken_intro_does_not_hide_first_line(self):
        lyrics = parse_lyrics('silver stars shining\nwe walk home')
        intro = measured('welcome everyone this performance starts today please enjoy the music before we begin this evening', 1000)
        observed = [intro, measured('silver stars shining we walk home', 20000)]
        windows = align_lyric_lines_to_word_windows(lyrics, observed)
        self.assertEqual(windows[0][0][0], len(intro.words))
        self.assertEqual(windows[1][0][0], len(intro.words) + 3)

    def test_repeated_chorus_keeps_both_occurrences(self):
        lyrics = 'hold me close\nthrough the night\nhold me close\nuntil daylight'
        segments = [measured('hold me closer', 1000), measured('through the night', 3000),
                    measured('hold me close', 5000), measured('until daylight', 7000)]
        result = AlignmentPipeline().run(lyrics, segments)
        self.assertEqual([line.start_ms for line in result.lines], [1000, 3000, 5000, 7000])

    def test_contractions_and_hyphens_match_without_confidence_loss(self):
        lyrics = parse_lyrics("don't stop\nwell-known stars")
        windows = align_lyric_lines_to_word_windows(lyrics, [measured("don't stop well-known stars", 1000)])
        self.assertEqual([windows[i][1] for i in range(2)], [1., 1.])

    def test_lrc_agreement_cannot_upgrade_coarse_timing(self):
        result = AlignmentPipeline().run('silver stars\nwe walk home',
            [AsrSegment('silver stars', 1000, 2000, .99),
             AsrSegment('we walk home', 3000, 4000, .99)],
            '[00:01.00]silver stars\n[00:03.00]we walk home')
        self.assertTrue(all(line.timing_confidence <= .5 for line in result.lines))

    def test_imported_lrc_is_diagnostic(self):
        segments = [measured('silver stars', 1123), measured('we walk home', 3123)]
        result = AlignmentPipeline().run('silver stars\nwe walk home', segments,
                                       '[00:00.62]silver stars\n[00:02.62]we walk home')
        self.assertEqual([line.start_ms for line in result.lines], [1123, 3123])
        self.assertEqual(result.global_offset_ms, 503)

    def test_segment_fallback_cannot_cross_future_word_anchor(self):
        segments = [AsrSegment('first gentle breeze', 1000, 2000, .98),
                    measured('silver stars', 3000),
                    AsrSegment('first gentle breeze', 5000, 6000, .98),
                    measured('we walk home', 7000)]
        result = AlignmentPipeline().run('first gentle breeze\nsilver stars\nwe walk home', segments)
        self.assertEqual([line.start_ms for line in result.lines], [1000, 3000, 7000])

    def test_lrc_rounds_across_minute_boundary(self):
        result = AlignmentPipeline().run('silver stars\nwe walk home',
                    [measured('silver stars', 59999), measured('we walk home', 120001)])
        self.assertEqual(format_lrc(result).splitlines(),
                         ['[01:00.00]silver stars', '[02:00.00]we walk home'])

    def test_worker_uses_ctc_only_in_precision_modes(self):
        for mode in ('fast', 'rap', 'quality'):
            with self.subTest(mode=mode):
                transcriber = Mock()
                rough = [measured('hello world', 1000)]
                transcriber.transcribe.return_value = rough
                acoustic = Mock()
                acoustic.align.return_value = [measured('hello world', 1123)]
                connection = Mock()
                connection.recv.side_effect = [('transcribe', ('input.wav', 'en')), ('close', None)]
                whisperx = Mock()
                with patch('lyric_tool.transcriber.FasterWhisperTranscriber', return_value=transcriber), \
                     patch('lyric_tool.acoustic_aligner.WhisperXAcousticAligner', return_value=acoustic), \
                     patch('lyric_tool.whisperx_transcriber.WhisperXTranscriber._load_whisperx', return_value=whisperx):
                    _worker_main(mode, connection)
                result = next(call.args[0][1] for call in connection.send.call_args_list if call.args[0][0] == 'result')
                self.assertEqual(result[0].words[0].start_ms, 1000 if mode == 'fast' else 1123)
                self.assertEqual(acoustic.align.call_count, 0 if mode == 'fast' else 1)
                if mode != 'fast':
                    self.assertEqual(acoustic.align.call_args.args[1],
                                     [{'text': 'hello world', 'start': .9, 'end': 1.75}])


if __name__ == '__main__':
    unittest.main()
