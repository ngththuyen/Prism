import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydub import AudioSegment as PydubAudioSegment
from pydantic import BaseModel, Field


class AudioSegment(BaseModel):
    """Single audio segment from TTS synthesis"""
    text: str
    start_time: float  # seconds
    end_time: float    # seconds
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[float] = None  # MB


class AudioResult(BaseModel):
    """Result of audio synthesis"""
    success: bool
    audio_path: Optional[str] = None
    audio_segments: List[AudioSegment] = Field(default_factory=list)
    total_duration: Optional[float] = None
    file_size_mb: Optional[float] = None
    error_message: Optional[str] = None

    # Metadata
    generation_time: Optional[float] = None
    voice_settings: Optional[Dict[str, Any]] = None
    model_used: str = "unknown"


class BaseTTSSynthesizer(ABC):
    """Abstract base class for TTS providers"""

    def __init__(
        self,
        api_key: str,
        output_dir: Path,
        max_retries: int = 3,
        timeout: int = 120,
        backoff_factor: float = 2.0
    ):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create cache directory for audio segments
        self.cache_dir = Path("cache") / "audio"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_key(self, text: str, voice_settings: Dict[str, Any]) -> str:
        """Generate a cache key from text and voice settings"""
        settings_str = '_'.join(f"{k}={v}" for k, v in sorted(voice_settings.items()))
        return f"{hash(text)}_{hash(settings_str)}"

    def _get_from_cache(self, cache_key: str) -> Optional[AudioResult]:
        """Try to get synthesized audio from cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                result = AudioResult.parse_file(cache_file)
                # Verify audio file still exists
                if result.audio_path and Path(result.audio_path).exists():
                    self.logger.info(f"Cache hit for key: {cache_key}")
                    return result
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {str(e)}")
        return None

    def _save_to_cache(self, cache_key: str, result: AudioResult):
        """Save synthesized audio result to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            cache_file.write_text(result.json(), encoding='utf-8')
            self.logger.info(f"Saved to cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {str(e)}")

    def synthesize_with_retry(self, text: str, voice_settings: Dict[str, Any]) -> AudioResult:
        """Synthesize audio with retry logic and caching"""
        cache_key = self._get_cache_key(text, voice_settings)
        
        # Try cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result

        # Attempt synthesis with retries
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = self.synthesize(text, voice_settings)
                if result.success:
                    result.generation_time = time.time() - start_time
                    self._save_to_cache(cache_key, result)
                    return result
            except Exception as e:
                last_error = str(e)
                wait_time = self.backoff_factor ** attempt
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)

        # All retries failed
        return AudioResult(
            success=False,
            error_message=f"All retries failed. Last error: {last_error}",
            generation_time=time.time() - start_time
        )
        self._setup_directories()

    @abstractmethod
    def execute(self, script_path: str, target_duration: Optional[float] = None) -> AudioResult:
        """Main synthesis method - must be implemented by providers"""
        pass

    def _setup_directories(self):
        """Create output directories"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "segments").mkdir(parents=True, exist_ok=True)

    def _parse_srt_file(self, script_file: Path) -> List[Dict[str, Any]]:
        """Parse SRT file and extract subtitles with timing"""

        subtitles = []

        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()

        blocks = content.strip().split('\n\n')

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            try:
                # Parse sequence number
                sequence = int(lines[0].strip())

                # Parse timestamps
                timestamp_line = lines[1].strip()
                if '-->' not in timestamp_line:
                    continue

                start_time_str, end_time_str = [t.strip() for t in timestamp_line.split('-->')]

                # Convert timestamp to seconds
                start_time = self._timestamp_to_seconds(start_time_str)
                end_time = self._timestamp_to_seconds(end_time_str)

                # Parse text (may be multiple lines)
                text = ' '.join(line.strip() for line in lines[2:] if line.strip())

                subtitles.append({
                    'sequence': sequence,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text,
                    'duration': end_time - start_time
                })

            except Exception as e:
                self.logger.warning(f"Error parsing subtitle block: {e}")
                continue

        return subtitles

    def _normalize_timestamp(self, timestamp: str) -> str:
        """Normalize malformed timestamps to proper HH:MM:SS,mmm format"""
        
        timestamp = timestamp.strip()
        timestamp = timestamp.replace('.', ',').replace(':', ',')
        parts = timestamp.split(',')
        
        if len(parts) == 3:
            minutes, seconds, milliseconds = parts
            hours = "00"
        elif len(parts) == 4:
            hours, minutes, seconds, milliseconds = parts
        else:
            raise ValueError(f"Cannot parse timestamp: {timestamp}")
        
        hours = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)
        milliseconds = int(milliseconds)
        
        if minutes >= 60:
            hours += minutes // 60
            minutes = minutes % 60
        
        if seconds >= 60:
            minutes += seconds // 60
            seconds = seconds % 60
        
        if milliseconds >= 1000:
            seconds += milliseconds // 1000
            milliseconds = milliseconds % 1000
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert SRT timestamp (HH:MM:SS,mmm) to seconds"""

        normalized = self._normalize_timestamp(timestamp)
        normalized = normalized.replace(',', ':')
        parts = normalized.split(':')

        if len(parts) != 4:
            raise ValueError(f"Invalid timestamp format: {timestamp}")

        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        milliseconds = int(parts[3])

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""

        hours = int(seconds // 3600)
        remaining = seconds % 3600
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        milliseconds = int((remaining % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _concatenate_audio_segments(self, audio_segments: List[AudioSegment], script_stem: str) -> Path:
        """Concatenate audio segments with proper timing using pydub"""

        if not audio_segments:
            raise ValueError("No audio segments to concatenate")

        self.logger.info(f"Concatenating {len(audio_segments)} audio segments with timing")

        audio_segments.sort(key=lambda s: s.start_time)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{script_stem}_audio_{timestamp}.mp3"
        output_path = self.output_dir / output_filename

        combined = PydubAudioSegment.silent(duration=0)
        current_time = 0.0

        for i, segment in enumerate(audio_segments):
            if not segment.audio_path:
                continue

            gap = segment.start_time - current_time

            if gap > 0.05:
                silence_duration_ms = int(gap * 1000)
                combined += PydubAudioSegment.silent(duration=silence_duration_ms)
                self.logger.debug(f"Added {gap:.2f}s silence before segment {i+1}")

            audio_clip = PydubAudioSegment.from_mp3(segment.audio_path)
            combined += audio_clip
            current_time = segment.start_time + (segment.duration or 0)

        combined.export(output_path, format="mp3", bitrate="192k")
        self.logger.info(f"Successfully concatenated audio with timing: {output_filename}")
        return output_path

    def _add_silence_padding(self, audio_path: Path, padding_duration: float) -> Path:
        """Add silence padding to match target duration"""

        self.logger.info(f"Adding {padding_duration:.2f}s silence padding")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        padded_filename = f"{audio_path.stem}_padded_{timestamp}.mp3"
        padded_path = self.output_dir / padded_filename

        audio = PydubAudioSegment.from_mp3(audio_path)
        silence = PydubAudioSegment.silent(duration=int(padding_duration * 1000))
        padded_audio = audio + silence
        padded_audio.export(padded_path, format="mp3", bitrate="192k")

        self.logger.info(f"Added silence padding: {padded_filename}")
        return padded_path

    def _get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """Get audio duration using pydub"""

        try:
            audio = PydubAudioSegment.from_mp3(audio_path)
            return len(audio) / 1000.0
        except Exception as e:
            self.logger.warning(f"Could not get audio duration for {audio_path}: {e}")
            return None

    def cleanup_temp_files(self):
        """Clean up temporary audio segment files"""

        segments_dir = self.output_dir / "segments"
        if segments_dir.exists():
            for file_path in segments_dir.glob("*.mp3"):
                try:
                    file_path.unlink()
                    self.logger.debug(f"Cleaned up temporary file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {file_path}: {e}")

    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get statistics about audio synthesis performance"""
        return {
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "output_dir": str(self.output_dir)
        }