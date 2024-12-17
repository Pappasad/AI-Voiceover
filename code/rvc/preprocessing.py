import librosa
import os
import numpy as np
import soundfile as sf
import sys
from pydub import AudioSegment
from pydub.silence import split_on_silence
from moviepy.video.io.VideoFileClip import VideoFileClip



__VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}
__AUDIO_EXT = {'.mp3', '.ogg', '.flac', '.aac', '.m4a', '.wma'}

def preprocessMain(input_dir=os.path.join('code', 'rvc', 'data'), output_dir=os.path.join('code', 'rvc', 'processed'), sample_rate=22050, num_fft=1024, hop_size=256, num_mels=80):
    print("Preprocessing...")
    if not os.path.exists(input_dir):
        print("<<<ERROR>>> code -> rvc -> preprocessing.py: input dir cant be found at:", input_dir)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    audio_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.wav'):
            audio_files.append(file)
        elif file[file.rfind('.'):] in __VIDEO_EXT or file[file.rfind('.'):] in __AUDIO_EXT:
            audio_files.append(convert2Wav(os.path.join(input_dir, file)))

    for audio_file in audio_files:
        input_path = os.path.join(input_dir, audio_file)
        clips_dir = preprocessAudio(input_path)

    for audio_file in os.listdir(clips_dir):
        input_path = os.path.join(clips_dir, audio_file)

        audio_waveform, sample_rate = librosa.load(input_path, sr=sample_rate)

        mel_spectrogram = librosa.feature.melspectrogram(y=audio_waveform, sr=sample_rate, n_fft=num_fft, hop_length=hop_size, n_mels=num_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        output_path = os.path.join(output_dir, audio_file.replace('.wav', '.npy'))

        np.save(output_path, mel_spectrogram_db)

        print(f"\tPreprocessed {audio_file}")

def preprocessAudio(path: str):
    print("\tPreprocessing :", path)
    audio, sr = librosa.load(path)
    print("\tDenoising...")
    denoised = eliminateNoise(audio)
    print("\tRemoving Silence...")
    denoised = librosa2Pydub(denoised, sr)
    no_silence = removeSilence(denoised)
    print("\tSplitting Audio...")
    clips = splitAudio(no_silence)

    output_dir = os.path.join('code', 'rvc', 'clips')
    os.makedirs(output_dir, exist_ok=True)

    print("\tExporting clips...")
    idx = 0
    for clip in clips:
        while any(f'{idx}' in file for file in os.listdir(output_dir)):
            idx += 1
        output_path = os.path.join(output_dir, f'{idx}.wav')
        clip.export(output_path, format='wav')

    return output_dir

def eliminateNoise(audio, noise_reduction=0.02):
    stft = librosa.stft(audio)
    mag, phase = librosa.magphase(stft)

    noise_thres = np.median(mag, axis=1) * noise_reduction

    filtered = np.where(mag > noise_thres[:, None], mag, 0)

    filtered = filtered * phase
    denoised = librosa.istft(filtered)

    return denoised

def removeSilence(audio, min_sil=500, threshold=-40):
    
    chunks = split_on_silence(
        audio,
        min_silence_len=min_sil,
        silence_thresh=threshold
    )

    combined = AudioSegment.empty()
    for chunk in chunks:
        combined += chunk

    return combined

def splitAudio(audio, min_length=5, max_length=10):
    clips = []
    duration = len(audio)

    start_time = 0
    while start_time < duration:
        clip_length = np.random.randint(min_length*1000, max_length*1000)

        end_time = min(start_time + clip_length, duration)

        clip = audio[start_time:end_time]
        clips.append(clip)

        start_time = end_time

    return clips

def librosa2Pydub(audio, sr):
    audio = (audio * 32767).astype(np.int16)
    audio = audio.tobytes()
    audio = AudioSegment(
        data=audio,
        sample_width=2,
        frame_rate=sr,
        channels=1
    )
    return audio

def convert2Wav(input_path: str):

    name, ext = os.path.splitext(input_path)

    output_path = input_path.replace(ext, '.wav')

    try:
        if ext.lower() in __VIDEO_EXT:
            print(f"Converting {ext} to .wav ...")
            video = VideoFileClip(input_path)
            video.audio.write_audiofile(output_path)
            video.close()
        elif ext.lower in __AUDIO_EXT:
            print(f"Converting {ext} to .wav ...")
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format='wav')
        elif ext.lower == '.wav':
            print("Already .wav")
            return input_path.split('\\')[-1]
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        os.remove(input_path)
        return output_path.split('\\')[-1]

    except Exception as e:
        print(f"<WARNING> code -> rvc -> preprocessing.py -> convert2Wav: {e}")
        return






