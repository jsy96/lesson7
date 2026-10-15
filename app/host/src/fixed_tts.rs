use std::fs;
use std::io::{self, Seek, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};
use thiserror::Error;
use zeroize::Zeroizing;

use crate::audio::{self, TTS_SAMPLE_RATE};
use crate::dashscope::{TtsAudio, TtsError, TtsReceipt};
use crate::paths::{AppPaths, secure_directory};
use crate::summary_orchestrator::SummarySynthesizer;

pub const FIXED_TTS_MODEL: &str = "fixed-local-v1";
pub const FIXED_TTS_VOICE: &str = "macos-say";
pub const FIXED_TTS_PHRASE: &str = "任务已完成，详细内容请在 Mac 上查看。";
const FIXED_TTS_FILE_NAME: &str = "task-complete.wav";

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

pub fn fixed_tts_directory(root: &Path) -> PathBuf {
    root.join("tts").join("fixed")
}

pub fn fixed_tts_path(root: &Path) -> PathBuf {
    fixed_tts_directory(root).join(FIXED_TTS_FILE_NAME)
}

pub fn fixed_tts_is_configured(paths: &AppPaths) -> bool {
    fixed_tts_path(&paths.root).is_file()
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
