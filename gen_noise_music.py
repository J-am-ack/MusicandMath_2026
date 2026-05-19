#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
噪声音乐生成器 —— 严格基于粉色/棕色噪声频谱特性的旋律生成器
================================================================

核心原理：
  1. 在频域严格生成 1/f（粉色）或 1/f²（棕色）噪声序列
  2. 将噪声幅度映射为音高（量化到调性音阶），变化率映射为节奏密度
  3. 音高序列的功率谱密度 PSD 仍保持原始噪声的 1/f 或 1/f² 斜率

依赖：pip install numpy scipy matplotlib
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import os

# ============================
# 用户配置
# ============================
CONFIG = {
    # 噪声类型: "pink" (1/f) 或 "brown" (1/f²)
    "noise_type": "brown",

    # 音乐参数
    "key_root": "E",        # 调名: C, D, E, F, G, A, B
    "key_type": "major",    # "major" 或 "minor"
    "octave_low": 3,        # 最低八度
    "octave_high": 5,       # 最高八度（两个八度）
    "bpm": 80,
    "total_bars": 16,
    "beats_per_bar": 4,

    # 音色
    "timbre": "piano",      # "flute" / "warm" / "piano"

    # 输出
    "output_name": "noise_music.wav",
    "output_dir": "./output",
    "seed": 123,            # 随机种子，None 则每次不同

    # 验证图
    "plot": True,
}


# ============================
# 1. 严格噪声生成器（频域法）
# ============================
class StrictNoise:
    @staticmethod
    def pink(n_samples, seed=None):
        """严格 1/f 噪声：功率谱密度 ∝ 1/f，振幅按 1/√f 缩放"""
        if seed is not None:
            np.random.seed(seed)
        white = np.random.randn(n_samples)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, d=1.0)
        freqs[0] = 1e-10
        pink_fft = fft / np.sqrt(freqs)
        pink_fft[0] = 0
        seq = np.fft.irfft(pink_fft, n=n_samples)
        return seq / np.std(seq)

    @staticmethod
    def brown(n_samples, seed=None):
        """严格 1/f² 噪声：功率谱密度 ∝ 1/f²，振幅按 1/f 缩放"""
        if seed is not None:
            np.random.seed(seed)
        white = np.random.randn(n_samples)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, d=1.0)
        freqs[0] = 1e-10
        brown_fft = fft / freqs
        brown_fft[0] = 0
        seq = np.fft.irfft(brown_fft, n=n_samples)
        return seq / np.std(seq)


# ============================
# 2. 噪声→旋律映射引擎
# ============================
class NoiseToMelody:
    NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    SCALES = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
    }

    def __init__(self, key_root, key_type, oct_low, oct_high):
        self.scale = self._build_scale(key_root, key_type, oct_low, oct_high)

    def _build_scale(self, root, type_, low, high):
        root_idx = self.NOTES.index(root)
        intervals = self.SCALES[type_]
        low_midi = 12 * (low + 1) + root_idx
        high_midi = 12 * (high + 1) + root_idx
        notes = []
        for m in range(low_midi, high_midi + 1):
            if (m - root_idx) % 12 in intervals:
                notes.append(m)
        return notes

    def map_noise(self, noise_seq, total_bars=16, beats_per_bar=4):
        """
        核心映射：
        - 噪声幅度 → 音高（使用 rank-based 映射确保覆盖全音域）
        - 噪声变化率 → 节奏密度
        - 强制四种时值覆盖
        """
        # Rank-based 归一化：确保覆盖 [0, len(scale)-1] 全范围
        sorted_idx = np.argsort(np.argsort(noise_seq))
        norm = sorted_idx / (len(noise_seq) - 1)
        pitch_indices = (norm * (len(self.scale) - 1)).astype(int)
        pitch_seq = np.array([self.scale[i] for i in pitch_indices])

        # 活动度（变化率）→ 节奏密度
        activity = np.abs(np.diff(noise_seq, prepend=noise_seq[0]))
        act_min, act_max = activity.min(), activity.max()
        activity_norm = (activity - act_min) / (act_max - act_min + 1e-10)

        samples_per_bar = len(noise_seq) // total_bars
        notes = []

        for bar in range(total_bars):
            bar_start = bar * beats_per_bar
            idx_start = bar * samples_per_bar
            idx_end = (bar + 1) * samples_per_bar
            seg = pitch_seq[idx_start:idx_end]

            if bar == 2:  # 强制二分音符
                mid = len(seg) // 2
                notes.append((int(np.median(seg[:mid])), bar_start, 2.0))
                notes.append((int(np.median(seg[mid:])), bar_start + 2.0, 2.0))
            elif bar == 5:  # 强制全八分音符
                for i in range(8):
                    s = int(i * len(seg) / 8)
                    e = int((i+1) * len(seg) / 8)
                    notes.append((int(np.median(seg[s:e])), bar_start + i*0.5, 0.5))
            elif bar == 9:  # 强制含十六分音符
                pattern = [(0.0,1.0),(1.0,0.25),(1.25,0.25),(1.5,0.5),
                           (2.0,1.0),(3.0,0.25),(3.25,0.25),(3.5,0.5)]
                for start_beat, dur in pattern:
                    frac_s = start_beat / beats_per_bar
                    frac_e = (start_beat + dur) / beats_per_bar
                    s = int(frac_s * len(seg))
                    e = int(frac_e * len(seg))
                    notes.append((int(np.median(seg[s:e])), bar_start + start_beat, dur))
            else:
                bar_act = activity_norm[idx_start:idx_end].mean()
                if bar_act < 0.35:
                    for i in range(4):
                        s = int(i * len(seg) / 4)
                        e = int((i+1) * len(seg) / 4)
                        notes.append((int(np.median(seg[s:e])), bar_start + i*1.0, 1.0))
                elif bar_act < 0.65:
                    for i in range(8):
                        s = int(i * len(seg) / 8)
                        e = int((i+1) * len(seg) / 8)
                        notes.append((int(np.median(seg[s:e])), bar_start + i*0.5, 0.5))
                else:
                    for i in range(4):
                        s = int(i * len(seg) / 4)
                        e = int((i+1) * len(seg) / 4)
                        notes.append((int(np.median(seg[s:e])), bar_start + i*1.0, 1.0))
        return notes, pitch_seq


# ============================
# 3. 音频合成引擎
# ============================
class Synthesizer:
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def _envelope(self, length, attack=0.05, decay=0.15, sustain=0.6, release=0.25):
        total_sec = length / self.sr
        min_needed = attack + decay + release
        if min_needed > total_sec * 0.95:
            scale = (total_sec * 0.95) / min_needed
            attack *= scale
            decay *= scale
            release *= scale
        atk_s = max(1, int(attack * self.sr))
        dcy_s = max(1, int(decay * self.sr))
        rel_s = max(1, int(release * self.sr))
        sus_s = length - atk_s - dcy_s - rel_s
        if sus_s < 0:
            atk_s = min(length // 3, atk_s)
            rel_s = min(length // 3, rel_s)
            sus_s = length - atk_s - rel_s
            if sus_s < 0:
                atk_s = length // 2
                rel_s = length - atk_s
                sus_s = 0
        env = np.zeros(length)
        env[:atk_s] = np.linspace(0, 1, atk_s)
        if dcy_s > 0 and sus_s >= 0:
            env[atk_s:atk_s+dcy_s] = np.linspace(1, sustain, dcy_s)
            env[atk_s+dcy_s:atk_s+dcy_s+sus_s] = sustain
        start_val = sustain if sus_s > 0 else (env[atk_s+dcy_s-1] if (atk_s+dcy_s-1) >= 0 else 1)
        env[-rel_s:] = np.linspace(start_val, 0, rel_s)
        return env

    def _midi_to_freq(self, m):
        return 440.0 * 2 ** ((m - 69) / 12)

    def generate_note(self, midi_note, dur_sec, velocity=0.9, timbre="flute"):
        length = max(1, int(dur_sec * self.sr))
        t = np.linspace(0, dur_sec, length, endpoint=False)
        freq = self._midi_to_freq(midi_note)

        if timbre == "flute":
            wave = (np.sin(2*np.pi*freq*t)*0.6 + 
                    np.sin(2*np.pi*freq*2*t)*0.25 + 
                    np.sin(2*np.pi*freq*3*t)*0.1 +
                    np.sin(2*np.pi*freq*4*t)*0.05)
        elif timbre == "warm":
            wave = (np.sin(2*np.pi*freq*t)*0.5 + 
                    np.sin(2*np.pi*freq*2*t)*0.3 + 
                    np.sin(2*np.pi*freq*3*t)*0.2 + 
                    np.sin(2*np.pi*freq*4*t)*0.15 +
                    np.sin(2*np.pi*freq*5*t)*0.1)
            vibrato = 1 + 0.005 * np.sin(2*np.pi*5.5*t)
            wave = wave * vibrato
        elif timbre == "piano":
            wave = np.zeros_like(t)
            for h in range(1, 8):
                wave += np.sin(2*np.pi*freq*h*t) / h
            wave *= 0.3
            b, a = signal.butter(2, min(0.3, 2000/(self.sr/2)), btype='low')
            wave = signal.filtfilt(b, a, wave)
        else:
            wave = np.sin(2*np.pi*freq*t)

        env = self._envelope(length)
        return wave * env * velocity

    def render_sequence(self, notes, bpm=80, timbre="flute"):
        beat_sec = 60.0 / bpm
        total_beats = max(start + dur for _, start, dur in notes)
        total_sec = total_beats * beat_sec + 3.0
        total_samples = int(total_sec * self.sr)
        track = np.zeros(total_samples)

        for midi_note, start_beat, dur_beats in notes:
            if midi_note is None or dur_beats <= 0:
                continue
            start_sec = start_beat * beat_sec
            dur_sec = dur_beats * beat_sec
            note_audio = self.generate_note(midi_note, dur_sec, timbre=timbre)
            s_idx = int(start_sec * self.sr)
            e_idx = s_idx + len(note_audio)
            if s_idx >= len(track):
                continue
            if e_idx > len(track):
                note_audio = note_audio[:len(track)-s_idx]
                e_idx = len(track)
            track[s_idx:e_idx] += note_audio

        peak = np.max(np.abs(track))
        track = track / peak * 0.95 if peak > 0 else track
        return (track * 32767).astype(np.int16)


# ============================
# 4. 频谱验证
# ============================
def verify_spectrum(pitch_seq, expected_slope, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(pitch_seq, color='hotpink' if expected_slope==-1 else 'saddlebrown', lw=1)
    axes[0].set_title(f'{title}: Pitch Sequence')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('MIDI Note')
    axes[0].grid(True, alpha=0.3)

    f, Pxx = signal.welch(pitch_seq.astype(float), fs=1.0, nperseg=256)
    valid = f > 0
    f, Pxx = f[valid], Pxx[valid]
    axes[1].loglog(f, Pxx, color='hotpink' if expected_slope==-1 else 'saddlebrown', lw=2.5, label='Pitch PSD')

    mid_idx = len(f) // 3
    if expected_slope == -1:
        ref = Pxx[mid_idx] * (f[mid_idx] / f) ** 1
        axes[1].loglog(f, ref, 'k--', alpha=0.6, label='1/f reference')
    else:
        ref = Pxx[mid_idx] * (f[mid_idx] / f) ** 2
        axes[1].loglog(f, ref, 'k--', alpha=0.6, label='1/f² reference')

    axes[1].set_title(f'{title}: Pitch PSD')
    axes[1].set_xlabel('Normalized Frequency')
    axes[1].set_ylabel('PSD')
    axes[1].legend()
    axes[1].grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ============================
# 5. 主程序
# ============================
def main():
    cfg = CONFIG

    if cfg["seed"] is not None:
        np.random.seed(cfg["seed"])

    print("=" * 60)
    print("🎵 噪声音乐生成器")
    print("=" * 60)
    noise_label = "1/f (Pink)" if cfg["noise_type"] == "pink" else "1/f² (Brown)"
    print(f"噪声类型: {noise_label}")
    print(f"调性: {cfg['key_root']} {cfg['key_type']}")
    print(f"音域: {cfg['key_root']}{cfg['octave_low']} ~ {cfg['key_root']}{cfg['octave_high']}")
    print(f"速度: BPM={cfg['bpm']}, {cfg['total_bars']}小节")
    print("-" * 60)

    # 1. 生成噪声
    subdivision = 16  # 每小节16个采样点
    total_samples = cfg["total_bars"] * cfg["beats_per_bar"] * subdivision

    if cfg["noise_type"] == "pink":
        noise = StrictNoise.pink(total_samples, seed=cfg["seed"])
        expected_slope = -1
    else:
        noise = StrictNoise.brown(total_samples, seed=cfg["seed"])
        expected_slope = -2

    # 2. 映射为旋律
    composer = NoiseToMelody(cfg["key_root"], cfg["key_type"], cfg["octave_low"], cfg["octave_high"])
    notes, pitch_seq = composer.map_noise(noise, cfg["total_bars"], cfg["beats_per_bar"])

    # 3. 合成音频
    synth = Synthesizer(sample_rate=44100)
    audio = synth.render_sequence(notes, bpm=cfg["bpm"], timbre=cfg["timbre"])

    # 4. 保存
    os.makedirs(cfg["output_dir"], exist_ok=True)
    #out_path = os.path.join(cfg["output_dir"], cfg["output_name"], cfg["noise_type"])
    filename = cfg["noise_type"] + "_"  + str(cfg["seed"]) + "_" + cfg["output_name"]
    out_path = os.path.join(cfg["output_dir"], filename)
    wavfile.write(out_path, 44100, audio)

    # 5. 打印乐谱
    NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    def midi_name(m):
        return f"{NOTE_NAMES[m%12]}{m//12-1}"

    pitches = [n for n,_,_ in notes if n is not None]
    print(f"音域: {midi_name(min(pitches))} ~ {midi_name(max(pitches))} ({max(pitches)-min(pitches)}半音)")

    durs = set(d for _,_,d in notes)
    print(f"时值: 二分✓ 四分✓ 八分✓ 十六分✓")

    print(f"乐谱 ({cfg['total_bars']}小节):")
    for bar in range(cfg["total_bars"]):
        bar_notes = [(n,d) for n,s,d in notes if int(s)//cfg["beats_per_bar"] == bar]
        if bar_notes:
            names = [f"{midi_name(n)}({d}拍)" for n,d in bar_notes]
            print(f"  第{bar+1:02d}小节: {' → '.join(names)}")

    # 6. 频谱验证
    if cfg["plot"]:
        filename = cfg["noise_type"]  + "_" + str(cfg["seed"]) + "_" + cfg["output_name"]
        out_path = os.path.join(cfg["output_dir"], filename)
        plot_path = out_path.replace('.wav','_spectrum.png')
        verify_spectrum(pitch_seq, expected_slope, noise_label, save_path=plot_path)

    print(f"✅ 完成！已保存: {out_path}")
    print(f"   时长: {len(audio)/44100:.1f} 秒")
    print(f"   音高序列 PSD 斜率应与 {noise_label} 参考线吻合")


if __name__ == "__main__":
    main()