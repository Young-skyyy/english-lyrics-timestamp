from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
import re
import zipfile

from .parser import parse_lyrics, parse_lrc

MAX_LYRIC_BYTES = 2 * 1024 * 1024
TEXT_EXTENSIONS = {'.txt', '.lrc', '.srt', '.vtt'}
_CUE = re.compile(
    r'^(?:(\d+):)?(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*'
)
_TITLE = re.compile(r'^.+\s[-–—]\s.+\((?:full\s+)?(?:explicit\s+)?lyrics\)$', re.I)


def _decode(data: bytes) -> str:
    encodings = ('utf-16',) if data.startswith((b'\xff\xfe', b'\xfe\xff')) else ('utf-8-sig', 'gb18030')
    for encoding in encodings:
        try:
            text = data.decode(encoding)
            if '\x00' not in text:
                return text.replace('\r\n', '\n').replace('\r', '\n')
        except UnicodeDecodeError:
            continue
    raise ValueError('无法识别歌词编码，请另存为 UTF-8 文本后重试。')


def _subtitle_to_lrc(text: str) -> str:
    rows: list[str] = []
    for block in re.split(r'\n\s*\n', text.strip()):
        lines = block.strip().splitlines()
        for index, line in enumerate(lines):
            cue = _CUE.match(line.strip())
            if cue is None:
                continue
            hours, minutes, seconds, fraction = cue.groups()
            total_ms = ((int(hours or 0) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(fraction)
            words = ' '.join(item.strip() for item in lines[index + 1:] if item.strip())
            words = re.sub(r'<[^>]*>', '', words).strip()
            if words:
                mins, rem = divmod(total_ms, 60_000)
                secs, millis = divmod(rem, 1000)
                rows.append(f'[{mins:02d}:{secs:02d}.{millis:03d}]{words}')
            break
    if not rows:
        raise ValueError('字幕文件中没有找到可读取的歌词时间段。')
    return '\n'.join(rows)


def import_lyrics_file(filename: str, data: bytes) -> dict[str, object]:
    if len(data) > MAX_LYRIC_BYTES:
        raise ValueError('歌词文件不能超过 2 MB。')
    extension = PurePosixPath(filename).suffix.lower()
    inner_filename: str | None = None
    if extension == '.zip':
        try:
            with zipfile.ZipFile(BytesIO(data)) as archive:
                candidates = [entry for entry in archive.infolist()
                              if not entry.is_dir()
                              and not any(part.startswith(('.', '__MACOSX')) for part in PurePosixPath(entry.filename).parts)
                              and PurePosixPath(entry.filename).suffix.lower() in TEXT_EXTENSIONS]
                if not candidates:
                    raise ValueError('压缩包中没有 TXT、LRC、SRT 或 VTT 歌词文件。')
                if len(candidates) > 1:
                    raise ValueError('压缩包中有多个歌词文件，请单独拖入需要使用的那一份。')
                entry = candidates[0]
                if entry.file_size > MAX_LYRIC_BYTES:
                    raise ValueError('解压后的歌词不能超过 2 MB。')
                with archive.open(entry) as source:
                    data = source.read(MAX_LYRIC_BYTES + 1)
                if len(data) > MAX_LYRIC_BYTES:
                    raise ValueError('解压后的歌词不能超过 2 MB。')
                inner_filename = entry.filename
                extension = PurePosixPath(inner_filename).suffix.lower()
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
            raise ValueError('无法读取压缩包，请确认文件完整且没有加密。') from exc
    if extension not in TEXT_EXTENSIONS:
        raise ValueError('请拖入 TXT、LRC、SRT、VTT 或包含歌词文本的 ZIP 文件。')

    text = _decode(data).strip()
    if extension in {'.srt', '.vtt'} or any(_CUE.match(line.strip()) for line in text.splitlines()):
        text = _subtitle_to_lrc(text)
    # Remove an explicit lyrics-document heading, without guessing whether an
    # arbitrary first line is a song title or part of the lyrics.
    lines = text.splitlines()
    if lines and _TITLE.fullmatch(lines[0].strip()):
        text = '\n'.join(lines[1:]).strip()
    is_lrc = bool(parse_lrc(text))
    count = len(parse_lrc(text)) if is_lrc else len(parse_lyrics(text))
    if not count:
        raise ValueError('文件中没有找到歌词内容。')
    return {'text': text, 'format': 'lrc' if is_lrc else 'text', 'line_count': count,
            'filename': filename, 'inner_filename': inner_filename}
