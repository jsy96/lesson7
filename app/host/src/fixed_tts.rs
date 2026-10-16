use std::fs;
use std::io::{self, Seek, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use thiserror::Error;
use zeroize::Zeroizing;

use crate::audio::{self, MAX_TTS_SECONDS, TTS_SAMPLE_RATE};
use crate::dashscope::{TtsAudio, TtsError, TtsReceipt};
use crate::paths::{AppPaths, secure_directory};
use crate::summary::MAX_SPOKEN_TEXT_BYTES;
use crate::summary_orchestrator::SummarySynthesizer;

pub const FIXED_TTS_MODEL: &str = "fixed-local-v1";
pub const FIXED_TTS_VOICE: &str = "macos-say";
pub const FIXED_TTS_PHRASE: &str = "任务已完成，详细内容请在 Mac 上查看。";
pub const LOCAL_SAY_TTS_MODEL: &str = "local-say-dynamic-v1";
pub const LOCAL_SAY_TTS_VOICE: &str = "macos-say-tingting";
const FIXED_TTS_FILE_NAME: &str = "task-complete.wav";
const LOCAL_SAY_EXECUTABLE: &str = "/usr/bin/say";
const LOCAL_SAY_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Error)]
pub enum FixedTtsError {
    #[error("fixed TTS audio could not be read: {0}")]
    Io(#[from] io::Error),
    #[error("fixed TTS audio is missing")]
    Missing,
    #[error("fixed TTS audio must be a private regular file owned by this process")]
    InsecureFile,
    #[error("fixed TTS audio must be canonical 48 kHz mono PCM16 WAV")]
    InvalidAudio,
    #[error("macOS say failed to generate fixed TTS audio")]
    GenerationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedTtsReceipt {
    pub bytes: u64,
    pub sha256: String,
    pub duration_ms: u64,
    pub destination: PathBuf,
}

pub struct FixedTtsLibrary {
    pcm: Zeroizing<Vec<u8>>,
    samples: u64,
}

impl FixedTtsLibrary {
    pub fn open(paths: &AppPaths) -> Result<Self, FixedTtsError> {
        let path = fixed_tts_path(&paths.root);
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                FixedTtsError::Missing
            } else {
                error.into()
            }
        })?;
        let permission_bits = metadata.permissions().mode() & 0o777;
        if !metadata.is_file()
            || metadata.uid() != unsafe { libc::geteuid() }
            || (permission_bits & !0o600) != 0
        {
            return Err(FixedTtsError::InsecureFile);
        }
        let wav = Zeroizing::new(fs::read(&path)?);
        let pcm = audio::decode_wav(&wav).map_err(|_| FixedTtsError::InvalidAudio)?;
        let samples = (pcm.len() / 2) as u64;
        if samples == 0 {
            return Err(FixedTtsError::InvalidAudio);
        }
        Ok(Self { pcm, samples })
    }

    pub fn audio(&self) -> TtsAudio {
        TtsAudio::new(
            Zeroizing::new(self.pcm.as_slice().to_vec()),
            TtsReceipt {
                model: FIXED_TTS_MODEL,
                voice: FIXED_TTS_VOICE.to_owned(),
                sample_rate: TTS_SAMPLE_RATE,
                samples: self.samples,
                characters: None,
                transport: "local-file",
                attempts: 1,
            },
        )
    }
}

pub struct FixedAudioSummarySynthesizer<'a> {
    library: &'a FixedTtsLibrary,
}

impl<'a> FixedAudioSummarySynthesizer<'a> {
    pub const fn new(library: &'a FixedTtsLibrary) -> Self {
        Self { library }
    }
}

impl SummarySynthesizer for FixedAudioSummarySynthesizer<'_> {
    fn synthesize(
        &self,
        _text: &str,
        _voice: &str,
        _instructions: &str,
    ) -> Result<TtsAudio, TtsError> {
        Ok(self.library.audio())
    }
}

pub struct LocalSaySummarySynthesizer<'a> {
    directory: PathBuf,
    executable: PathBuf,
    timeout: Duration,
    library: &'a FixedTtsLibrary,
}

impl<'a> LocalSaySummarySynthesizer<'a> {
    pub fn new(paths: &AppPaths, library: &'a FixedTtsLibrary) -> Self {
        Self {
            directory: local_say_tts_directory(&paths.root),
            executable: PathBuf::from(LOCAL_SAY_EXECUTABLE),
            timeout: LOCAL_SAY_TIMEOUT,
            library,
        }
    }

    fn generate(&self, text: &str) -> Result<TtsAudio, TtsError> {
        if text.trim().is_empty()
            || text.len() > MAX_SPOKEN_TEXT_BYTES
            || text.chars().any(char::is_control)
        {
            return Err(TtsError::InvalidRequest);
        }

        secure_directory(&self.directory).map_err(|_| TtsError::Unavailable)?;
        let temporary = tempfile::Builder::new()
            .prefix(".spoken-text-")
            .suffix(".wav")
            .tempfile_in(&self.directory)
            .map_err(|_| TtsError::Unavailable)?;
        temporary
            .as_file()
            .set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|_| TtsError::Unavailable)?;

        let mut command = Command::new(&self.executable);
        command
            .arg("-v")
            .arg("Tingting")
            .arg("--data-format=LEI16@48000")
            .arg("-o")
            .arg(temporary.path())
            .arg("-f")
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let mut child = command.spawn().map_err(|_| TtsError::Unavailable)?;
        let mut stdin = child.stdin.take().ok_or(TtsError::Protocol)?;
        let payload = Zeroizing::new(text.as_bytes().to_vec());
        let writer = thread::spawn(move || stdin.write_all(&payload));

        let deadline = Instant::now() + self.timeout;
        let status = loop {
            if let Some(status) = child.try_wait().map_err(|_| TtsError::Unavailable)? {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                return Err(TtsError::Unavailable);
            }
            thread::sleep(Duration::from_millis(10));
        };
        writer
            .join()
            .map_err(|_| TtsError::Protocol)?
            .map_err(|_| TtsError::Unavailable)?;
        if !status.success() {
            return Err(TtsError::Unavailable);
        }

        temporary
            .as_file()
            .set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|_| TtsError::Unavailable)?;
        let generated = Zeroizing::new(fs::read(temporary.path()).map_err(|_| TtsError::Protocol)?);
        let pcm = Zeroizing::new(extract_say_pcm(&generated).ok_or(TtsError::Protocol)?);
        let samples = (pcm.len() / 2) as u64;
        if samples == 0 || samples > u64::from(TTS_SAMPLE_RATE) * u64::from(MAX_TTS_SECONDS) {
            return Err(TtsError::AudioLimit);
        }
        audio::encode_wav(&pcm).map_err(|_| TtsError::Protocol)?;
        Ok(TtsAudio::new(
            pcm,
            TtsReceipt {
                model: LOCAL_SAY_TTS_MODEL,
                voice: LOCAL_SAY_TTS_VOICE.to_owned(),
                sample_rate: TTS_SAMPLE_RATE,
                samples,
                characters: Some(text.chars().count() as u64),
                transport: "local-process",
                attempts: 1,
            },
        ))
    }
}

impl SummarySynthesizer for LocalSaySummarySynthesizer<'_> {
    fn synthesize(
        &self,
        text: &str,
        _voice: &str,
        _instructions: &str,
    ) -> Result<TtsAudio, TtsError> {
        match self.generate(text) {
            Ok(audio) => Ok(audio),
            Err(TtsError::InvalidRequest) => Err(TtsError::InvalidRequest),
            Err(error) => {
                let reason = match error {
                    TtsError::Unavailable => "generation_failed",
                    TtsError::Protocol => "invalid_output",
                    TtsError::AudioLimit => "audio_limit",
                    _ => "generation_failed",
                };
                eprintln!("local_tts_fallback reason={reason}");
                Ok(self.library.audio())
            }
        }
    }
}

pub fn fixed_tts_directory(root: &Path) -> PathBuf {
    root.join("tts").join("fixed")
}

pub fn fixed_tts_path(root: &Path) -> PathBuf {
    fixed_tts_directory(root).join(FIXED_TTS_FILE_NAME)
}

pub fn fixed_tts_is_configured(paths: &AppPaths) -> bool {
    fixed_tts_path(&paths.root).is_file()
}

pub fn local_say_tts_directory(root: &Path) -> PathBuf {
    root.join("tts").join("local-say")
}

pub fn local_say_is_configured() -> bool {
    Path::new(LOCAL_SAY_EXECUTABLE).is_file()
}

fn extract_say_pcm(wav: &[u8]) -> Option<Vec<u8>> {
    let read_u32 = |offset: usize| -> Option<u32> {
        wav.get(offset..offset.checked_add(4)?)?
            .try_into()
            .ok()
            .map(u32::from_le_bytes)
    };
    let read_u16 = |offset: usize| -> Option<u16> {
        wav.get(offset..offset.checked_add(2)?)?
            .try_into()
            .ok()
            .map(u16::from_le_bytes)
    };
    if wav.len() < 12 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return None;
    }
    let mut offset = 12;
    let mut format_valid = false;
    let mut data = None;
    while offset + 8 <= wav.len() {
        let identifier = &wav[offset..offset + 4];
        let size = read_u32(offset + 4)? as usize;
        let payload = offset.checked_add(8)?;
        let end = payload.checked_add(size)?;
        if end > wav.len() {
            return None;
        }
        if identifier == b"fmt " {
            format_valid = size >= 16
                && read_u16(payload)? == 1
                && read_u16(payload + 2)? == 1
                && read_u32(payload + 4)? == TTS_SAMPLE_RATE
                && read_u16(payload + 12)? == 2
                && read_u16(payload + 14)? == 16;
        } else if identifier == b"data" && data.is_none() {
            data = Some(wav[payload..end].to_vec());
        }
        offset = end.checked_add(size & 1)?;
    }
    let data = data?;
    if !format_valid || data.is_empty() || data.len() % 2 != 0 {
        return None;
    }
    Some(data)
}

#[cfg(target_os = "macos")]
pub fn generate_fixed_tts(
    paths: &AppPaths,
    voice: Option<&str>,
) -> Result<FixedTtsReceipt, FixedTtsError> {
    let directory = fixed_tts_directory(&paths.root);
    secure_directory(&directory)?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".task-complete-")
        .suffix(".wav")
        .tempfile_in(&directory)?;
    let status = Command::new("/usr/bin/say")
        .arg("-v")
        .arg(voice.unwrap_or("Tingting"))
        .arg("--data-format=LEI16@48000")
        .arg("-o")
        .arg(temporary.path())
        .arg(FIXED_TTS_PHRASE)
        .status()?;
    if !status.success() {
        return Err(FixedTtsError::GenerationFailed);
    }
    temporary
        .as_file()
        .set_permissions(fs::Permissions::from_mode(0o600))?;
    temporary.as_file().sync_all()?;
    let generated = fs::read(temporary.path())?;
    let pcm = extract_say_pcm(&generated).ok_or(FixedTtsError::InvalidAudio)?;
    let canonical = audio::encode_wav(&pcm).map_err(|_| FixedTtsError::InvalidAudio)?;
    temporary.as_file().set_len(0)?;
    temporary.as_file().seek(std::io::SeekFrom::Start(0))?;
    temporary.write_all(&canonical)?;
    temporary.as_file().sync_all()?;
    let destination = fixed_tts_path(&paths.root);
    let bytes = canonical.len() as u64;
    let sha256 = format!("{:x}", Sha256::digest(&canonical));
    let duration_ms = (canonical.len() as u64 - 44) * 1_000 / 2 / u64::from(TTS_SAMPLE_RATE);
    temporary
        .persist(&destination)
        .map_err(|error| FixedTtsError::Io(error.error))?;
    fs::set_permissions(&destination, fs::Permissions::from_mode(0o600))?;
    Ok(FixedTtsReceipt {
        bytes,
        sha256,
        duration_ms,
        destination,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_wav(path: &Path, sample_rate: u32, mode: u32) {
        let pcm = [0_i16, 1, -1, 2]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect::<Vec<_>>();
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
        wav.extend_from_slice(&pcm);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, wav).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
    }

    fn install_executable(path: &Path, contents: &str) {
        fs::write(path, contents).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }

    fn write_pcm_wav(path: &Path, samples: &[i16]) {
        let pcm = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(path, audio::encode_wav(&pcm).unwrap()).unwrap();
    }

    fn dynamic_synthesizer<'a>(
        paths: &'a AppPaths,
        library: &'a FixedTtsLibrary,
        executable: &'a Path,
        timeout: Duration,
    ) -> LocalSaySummarySynthesizer<'a> {
        let mut synthesizer = LocalSaySummarySynthesizer::new(paths, library);
        synthesizer.executable = executable.to_path_buf();
        synthesizer.timeout = timeout;
        synthesizer
    }

    fn install_successful_fake_say(path: &Path, record: &Path) {
        assert!(!record.to_string_lossy().contains('\''));
        install_executable(
            path,
            &format!(
                r#"#!/bin/sh
set -eu
record='{}'
output=
previous=
for argument in "$@"; do
    if [ "$previous" = "-o" ]; then output=$argument; fi
    previous=$argument
done
input=$(cat)
printf '%s\n' "$@" > "$record/argv"
printf '%s' "$input" > "$record/stdin"
case "$input" in
    *dynamic-first*) cp "$record/first.wav" "$output" ;;
    *dynamic-second*) cp "$record/second.wav" "$output" ;;
    *) exit 9 ;;
esac
"#,
                record.display()
            ),
        );
    }

    fn install_output_fake_say(path: &Path, output_contents: &str) {
        install_executable(
            path,
            &format!(
                r#"#!/bin/sh
set -eu
output=
previous=
for argument in "$@"; do
    if [ "$previous" = "-o" ]; then output=$argument; fi
    previous=$argument
done
printf '%s' '{output_contents}' > "$output"
"#
            ),
        );
    }

    #[test]
    fn say_wav_with_extra_chunks_is_reduced_to_canonical_pcm() {
        let pcm = [0_i16, 1, -1, 2]
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&0_u32.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"JUNK");
        wav.extend_from_slice(&28_u32.to_le_bytes());
        wav.extend_from_slice(&[0; 28]);
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&TTS_SAMPLE_RATE.to_le_bytes());
        wav.extend_from_slice(&(TTS_SAMPLE_RATE * 2).to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"FLLR");
        wav.extend_from_slice(&3_u32.to_le_bytes());
        wav.extend_from_slice(&[7, 8, 9, 0]);
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
        wav.extend_from_slice(&pcm);
        let riff_size = (wav.len() - 8) as u32;
        wav[4..8].copy_from_slice(&riff_size.to_le_bytes());
        assert_eq!(extract_say_pcm(&wav).unwrap(), pcm);
        let mut wrong_rate = wav.clone();
        wrong_rate[60..64].copy_from_slice(&44_100_u32.to_le_bytes());
        assert!(extract_say_pcm(&wrong_rate).is_none());
    }

    #[test]
    fn library_publishes_fixed_provenance() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_root(temporary.path().to_path_buf());
        write_wav(&fixed_tts_path(&paths.root), TTS_SAMPLE_RATE, 0o600);
        let library = FixedTtsLibrary::open(&paths).unwrap();
        let audio = library.audio();
        assert_eq!(audio.receipt().model, FIXED_TTS_MODEL);
        assert_eq!(audio.receipt().voice, FIXED_TTS_VOICE);
        assert_eq!(audio.receipt().sample_rate, TTS_SAMPLE_RATE);
        assert_eq!(audio.receipt().samples, 4);
        let synthesizer = FixedAudioSummarySynthesizer::new(&library);
        let synthesized = synthesizer
            .synthesize("ignored", "ignored", "ignored")
            .unwrap();
        assert_eq!(synthesized.receipt(), audio.receipt());
        assert_eq!(synthesized.pcm(), audio.pcm());
    }

    #[test]
    fn local_say_reads_text_from_stdin_and_publishes_dynamic_provenance() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_root(temporary.path().to_path_buf());
        write_wav(&fixed_tts_path(&paths.root), TTS_SAMPLE_RATE, 0o600);
        let library = FixedTtsLibrary::open(&paths).unwrap();
        let record = temporary.path().join("record");
        fs::create_dir(&record).unwrap();
        fs::set_permissions(&record, fs::Permissions::from_mode(0o700)).unwrap();
        write_pcm_wav(&record.join("first.wav"), &[1, 2, 3]);
        write_pcm_wav(&record.join("second.wav"), &[4, 5, 6]);
        let executable = temporary.path().join("fake-say");
        install_successful_fake_say(&executable, &record);
        let synthesizer = dynamic_synthesizer(&paths, &library, &executable, LOCAL_SAY_TIMEOUT);

        let first_text = "dynamic-first summary";
        let second_text = "dynamic-second summary";
        let first = synthesizer
            .synthesize(first_text, "ignored", "ignored")
            .unwrap();
        let second = synthesizer
            .synthesize(second_text, "ignored", "ignored")
            .unwrap();

        assert_ne!(first.pcm(), second.pcm());
        let first_pcm = [1_i16, 2, 3]
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let second_pcm = [4_i16, 5, 6]
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(first.pcm(), first_pcm);
        assert_eq!(second.pcm(), second_pcm);
        assert_eq!(first.receipt().model, LOCAL_SAY_TTS_MODEL);
        assert_eq!(first.receipt().voice, LOCAL_SAY_TTS_VOICE);
        assert_eq!(first.receipt().sample_rate, TTS_SAMPLE_RATE);
        assert_eq!(first.receipt().samples, 3);
        assert_eq!(
            first.receipt().characters,
            Some(first_text.chars().count() as u64)
        );
        assert_eq!(first.receipt().transport, "local-process");
        assert_eq!(
            second.receipt().characters,
            Some(second_text.chars().count() as u64)
        );

        let argv = fs::read_to_string(record.join("argv")).unwrap();
        assert!(!argv.contains(first_text));
        assert!(!argv.contains(second_text));
        assert!(argv.contains("-f"));
        assert!(argv.contains("-"));
        assert_eq!(
            fs::read(record.join("stdin")).unwrap(),
            second_text.as_bytes()
        );
        let directory_mode = fs::metadata(local_say_tts_directory(&paths.root))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(directory_mode, 0o700);
    }

    #[test]
    fn local_say_process_failure_falls_back_to_fixed_audio() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_root(temporary.path().to_path_buf());
        write_wav(&fixed_tts_path(&paths.root), TTS_SAMPLE_RATE, 0o600);
        let library = FixedTtsLibrary::open(&paths).unwrap();
        let executable = temporary.path().join("fail-say");
        install_executable(&executable, "#!/bin/sh\nexit 42\n");
        let synthesizer = dynamic_synthesizer(&paths, &library, &executable, LOCAL_SAY_TIMEOUT);

        let synthesized = synthesizer
            .synthesize("valid summary", "ignored", "ignored")
            .unwrap();
        let fallback = library.audio();
        assert_eq!(synthesized.receipt(), fallback.receipt());
        assert_eq!(synthesized.pcm(), fallback.pcm());
    }

    #[test]
    fn local_say_invalid_output_falls_back_to_fixed_audio() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_root(temporary.path().to_path_buf());
        write_wav(&fixed_tts_path(&paths.root), TTS_SAMPLE_RATE, 0o600);
        let library = FixedTtsLibrary::open(&paths).unwrap();
        let executable = temporary.path().join("invalid-say");
        install_output_fake_say(&executable, "not-a-riff-wav");
        let synthesizer = dynamic_synthesizer(&paths, &library, &executable, LOCAL_SAY_TIMEOUT);

        let synthesized = synthesizer
            .synthesize("valid summary", "ignored", "ignored")
            .unwrap();
        assert_eq!(synthesized.receipt().model, FIXED_TTS_MODEL);
        assert_eq!(synthesized.receipt().voice, FIXED_TTS_VOICE);
    }

    #[test]
    fn local_say_timeout_falls_back_without_waiting_forever() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_root(temporary.path().to_path_buf());
        write_wav(&fixed_tts_path(&paths.root), TTS_SAMPLE_RATE, 0o600);
        let library = FixedTtsLibrary::open(&paths).unwrap();
        let executable = temporary.path().join("slow-say");
        install_executable(&executable, "#!/bin/sh\nexec /bin/sleep 60\n");
        let synthesizer =
            dynamic_synthesizer(&paths, &library, &executable, Duration::from_millis(40));

        let started = Instant::now();
        let synthesized = synthesizer
            .synthesize("valid summary", "ignored", "ignored")
            .unwrap();
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(synthesized.receipt().model, FIXED_TTS_MODEL);
    }

    #[test]
    #[ignore = "requires the real macOS Tingting voice"]
    fn real_local_say_generates_distinct_audio() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_root(temporary.path().to_path_buf());
        write_wav(&fixed_tts_path(&paths.root), TTS_SAMPLE_RATE, 0o600);
        let library = FixedTtsLibrary::open(&paths).unwrap();
        let synthesizer = LocalSaySummarySynthesizer::new(&paths, &library);

        let first = synthesizer
            .synthesize("这是第一句本地语音检查。", "ignored", "ignored")
            .unwrap();
        let second = synthesizer
            .synthesize("这是第二句完全不同的结果。", "ignored", "ignored")
            .unwrap();

        assert_eq!(first.receipt().model, LOCAL_SAY_TTS_MODEL);
        assert_eq!(first.receipt().voice, LOCAL_SAY_TTS_VOICE);
        assert_ne!(first.pcm(), second.pcm());
        assert!(first.receipt().samples > 0);
        assert!(second.receipt().samples > 0);
    }

    #[test]
    fn local_say_rejects_invalid_text_before_spawning() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_root(temporary.path().to_path_buf());
        write_wav(&fixed_tts_path(&paths.root), TTS_SAMPLE_RATE, 0o600);
        let library = FixedTtsLibrary::open(&paths).unwrap();
        let executable = temporary.path().join("must-not-run");
        install_executable(&executable, "#!/bin/sh\nexit 42\n");
        let synthesizer = dynamic_synthesizer(&paths, &library, &executable, LOCAL_SAY_TIMEOUT);

        assert!(matches!(
            synthesizer.synthesize("bad\ncontrol", "ignored", "ignored"),
            Err(TtsError::InvalidRequest)
        ));
        assert!(!executable.with_extension("ran").exists());
    }

    #[test]
    fn library_rejects_wrong_rate_and_loose_permissions() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_root(temporary.path().to_path_buf());
        write_wav(&fixed_tts_path(&paths.root), 44_100, 0o600);
        assert!(matches!(
            FixedTtsLibrary::open(&paths),
            Err(FixedTtsError::InvalidAudio)
        ));
        write_wav(&fixed_tts_path(&paths.root), TTS_SAMPLE_RATE, 0o644);
        assert!(matches!(
            FixedTtsLibrary::open(&paths),
            Err(FixedTtsError::InsecureFile)
        ));
    }

    #[test]
    fn missing_library_is_reported_as_missing() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_root(temporary.path().to_path_buf());
        assert!(matches!(
            FixedTtsLibrary::open(&paths),
            Err(FixedTtsError::Missing)
        ));
        assert!(!fixed_tts_is_configured(&paths));
    }
}
