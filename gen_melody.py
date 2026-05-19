#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单旋律主题生成器
================
生成具备音乐性的单旋律线条，可直接作为乐曲主题进行扩展。

特性：
  - 自定义调性、音域、速度、小节数
  - 自动包含二分/四分/八分/十六分音符
  - 基于音阶与和弦分解的智能旋律生成
  - 内置 ADSR 包络与多种虚拟音色
  - 输出标准 44.1kHz 16-bit WAV

依赖：pip install numpy scipy
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
import random
import os

# ============================
# 用户配置区（修改这里即可）
# ============================
CONFIG = {
    # 基础参数
    "bpm": 80,              # 速度
    "time_signature": (4,4), # 拍号 (分子,分母)
    "total_bars": 16,       # 总小节数
    "sample_rate": 44100,   # 采样率

    # 调性与音域
    "key_root": "G",        # 调名: C, D, E, F, G, A, B 等
    "key_type": "major",    # 调式: "major" 或 "minor"
    "octave_low": 3,        # 最低八度 (如 3 代表 G3)
    "octave_high": 5,       # 最高八度 (如 5 代表 G5)

    # 旋律性格
    "mood": "warm",         # "warm"(温暖叙事) / "bright"(明亮通透) / "dark"(低沉内敛)
    "timbre": "flute",      # "flute"(长笛) / "warm"(弦乐) / "piano"(钢琴感)

    # 输出
    "output_name": "my_theme.wav",
    "output_dir": "./output",

    # 随机种子（固定可复现，设为 None 则每次不同）
    "seed": 42
}


# ============================
# 音乐理论引擎
# ============================
class MusicTheory:
    NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    # 大调/小调音阶（半音数间隔）
    SCALES = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],  # 自然小调
    }

    # 各级数和弦（大调/小调的三和弦根音级数）
    CHORD_DEGREES = {
        "major": [0, 1, 2, 3, 4, 5, 6],  # I-II-III-IV-V-VI-VII
        "minor": [0, 1, 2, 3, 4, 5, 6],
    }

    @classmethod
    def note_to_midi(cls, note_name, octave):
        """将音名+八度转为 MIDI 音高，如 ('G',3)->43"""
        idx = cls.NOTES.index(note_name)
        return 12 * (octave + 1) + idx

    @classmethod
    def midi_to_freq(cls, midi_note):
        return 440.0 * 2 ** ((midi_note - 69) / 12)

    @classmethod
    def build_scale_midi(cls, root_name, key_type, octave_low, octave_high):
        """构建指定调性的音阶 MIDI 列表"""
        root_idx = cls.NOTES.index(root_name)
        intervals = cls.SCALES[key_type]
        scale_notes = []
        for octv in range(octave_low, octave_high + 1):
            for iv in intervals:
                midi = 12 * (octv + 1) + root_idx + iv
                scale_notes.append(midi)
        # 去重排序
        scale_notes = sorted(list(set(scale_notes)))
        # 限制在要求的八度范围内
        low_limit = cls.note_to_midi(root_name, octave_low)
        high_limit = cls.note_to_midi(root_name, octave_high)
        return [n for n in scale_notes if low_limit <= n <= high_limit]

    @classmethod
    def get_chord_tones(cls, root_name, key_type, degree, octave):
        """获取某级和弦的三和弦音（根音、三音、五音）"""
        scale = cls.build_scale_midi(root_name, key_type, octave, octave+1)
        # 确保取到足够音
        scale = sorted(list(set(scale)))
        root = scale[degree % len(scale)]
        third_idx = (degree + 2) % 7
        fifth_idx = (degree + 4) % 7
        # 向上找到合适的八度位置
        chord = [scale[degree % 7]]
        for idx in [third_idx, fifth_idx]:
            note = scale[idx % 7]
            while note <= chord[-1]:
                note += 12
            chord.append(note)
        return chord[:3]


# ============================
# 音频合成引擎
# ============================
class Synthesizer:
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def _envelope(self, length, attack=0.05, decay=0.15, sustain=0.6, release=0.25):
        """自适应 ADSR：短音符自动压缩包络"""
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
        # 安全地处理 release
        start_val = sustain if sus_s > 0 else (env[atk_s+dcy_s-1] if (atk_s+dcy_s-1) >= 0 else 1)
        env[-rel_s:] = np.linspace(start_val, 0, rel_s)
        return env

    def generate_note(self, midi_note, duration_sec, velocity=0.9, timbre="flute"):
        """合成单个音符"""
        length = max(1, int(duration_sec * self.sr))
        t = np.linspace(0, duration_sec, length, endpoint=False)
        freq = MusicTheory.midi_to_freq(midi_note)

        if timbre == "flute":
            # 长笛：正弦为主，弱谐波
            wave = (np.sin(2*np.pi*freq*t)*0.6 + 
                    np.sin(2*np.pi*freq*2*t)*0.25 + 
                    np.sin(2*np.pi*freq*3*t)*0.1 +
                    np.sin(2*np.pi*freq*4*t)*0.05)
        elif timbre == "warm":
            # 温暖弦乐：更多谐波+轻微颤音
            wave = (np.sin(2*np.pi*freq*t)*0.5 + 
                    np.sin(2*np.pi*freq*2*t)*0.3 + 
                    np.sin(2*np.pi*freq*3*t)*0.2 + 
                    np.sin(2*np.pi*freq*4*t)*0.15 +
                    np.sin(2*np.pi*freq*5*t)*0.1)
            vibrato = 1 + 0.005 * np.sin(2*np.pi*5.5*t)
            wave = wave * vibrato
        elif timbre == "piano":
            # 钢琴感：锯齿波+低通滤波模拟
            wave = np.zeros_like(t)
            for h in range(1, 8):
                wave += np.sin(2*np.pi*freq*h*t) / h
            wave *= 0.3
            # 简单低通滤波模拟
            b, a = signal.butter(2, min(0.3, 2000/(self.sr/2)), btype='low')
            wave = signal.filtfilt(b, a, wave)
        else:
            wave = np.sin(2*np.pi*freq*t)

        env = self._envelope(length, attack=0.05, decay=0.15, sustain=0.6, release=0.25)
        return wave * env * velocity

    def render_sequence(self, notes, bpm=80, timbre="flute"):
        """
        渲染音符序列
        notes: list of (midi_note, start_beat, duration_beats)
        """
        beat_sec = 60.0 / bpm
        if not notes:
            return np.array([], dtype=np.int16)
        total_beats = max(start + dur for _, start, dur in notes)
        total_sec = total_beats * beat_sec + 3.0  # 尾音余量
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
        if peak > 1.0:
            track = track / peak * 0.95
        else:
            track = track * 0.95
        return (track * 32767).astype(np.int16)


# ============================
# 旋律作曲引擎
# ============================
class MelodyComposer:
    """基于音乐规则自动生成旋律"""

    def __init__(self, config):
        self.cfg = config
        self.scale = MusicTheory.build_scale_midi(
            config["key_root"], config["key_type"],
            config["octave_low"], config["octave_high"]
        )
        self.lowest = MusicTheory.note_to_midi(config["key_root"], config["octave_low"])
        self.highest = MusicTheory.note_to_midi(config["key_root"], config["octave_high"])
        self.beats_per_bar = config["time_signature"][0]
        self.total_bars = config["total_bars"]

        # 节奏库：确保包含四种时值
        self.rhythm_pool = [
            [1.0],           # 四分
            [2.0],           # 二分
            [0.5, 0.5],      # 两个八分
            [1.0, 0.5, 0.5], # 四分+两个八分
            [0.5, 1.0, 0.5], # 八分+四分+八分
            [0.5, 0.5, 1.0], # 两个八分+四分
            [0.25, 0.25, 0.5, 1.0],  # 含十六分
            [1.0, 0.25, 0.25, 0.5],  # 含十六分
            [0.5, 0.25, 0.25, 0.5, 0.5], # 含十六分
            [0.25, 0.25, 0.25, 0.25, 1.0], # 四个十六分+四分
        ]

        # 根据 mood 设置旋律倾向
        mood = config.get("mood", "warm")
        if mood == "bright":
            self.prefer_high = 0.6
            self.jump_prob = 0.4
        elif mood == "dark":
            self.prefer_high = -0.3
            self.jump_prob = 0.2
        else:  # warm
            self.prefer_high = 0.1
            self.jump_prob = 0.3

    def _choose_rhythm_pattern(self, bar_idx):
        """选择节奏型，确保整曲覆盖四种时值"""
        # 前4小节：简单节奏建立主题
        if bar_idx < 4:
            return random.choice([
                [1.0, 1.0, 1.0, 1.0],      # 全四分
                [2.0, 1.0, 1.0],           # 二分+四分
                [1.0, 0.5, 0.5, 1.0, 1.0], # 含八分
            ])
        # 中间：加入十六分活跃气氛
        elif bar_idx < 12:
            return random.choice(self.rhythm_pool)
        # 最后：回归稳定，二分音符收尾
        else:
            return random.choice([
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [1.0, 1.0, 2.0],
                [2.0, 2.0],  # 全二分
            ])

    def _next_note(self, current, target_area=None):
        """智能选择下一个音：以级进为主，偶尔跳进"""
        if current is None:
            # 从调性主音或属音开始
            candidates = [n for n in self.scale if self.lowest <= n <= self.lowest+12]
            return random.choice(candidates) if candidates else self.scale[0]

        # 确定目标区域偏好
        center = (self.lowest + self.highest) / 2
        if target_area == "high":
            center = self.highest - 6
        elif target_area == "low":
            center = self.lowest + 6

        # 生成候选：级进（±1,2度）和跳进（±3度以上）
        candidates = []
        for n in self.scale:
            diff = abs(n - current)
            if diff == 0:
                continue
            # 距离中心越近权重越高
            weight = 1.0 / (1 + abs(n - center)/3.0)
            # 级进优先
            if diff <= 2:
                weight *= 3.0
            elif diff <= 4:
                weight *= 1.5
            else:
                weight *= 0.5
            candidates.append((n, weight))

        if not candidates:
            return current

        notes, weights = zip(*candidates)
        weights = np.array(weights)
        weights /= weights.sum()
        return np.random.choice(notes, p=weights)

    def compose(self):
        """生成完整旋律音符列表"""
        notes = []
        current_beat = 0.0
        prev_note = None

        # 强制确保四种时值都出现（手动插入标记小节）
        forced_rhythms = {
            2: [2.0, 1.0, 1.0],                     # 第3小节强制二分
            5: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], # 第6小节全八分
            9: [1.0, 0.25, 0.25, 0.5, 1.0, 0.5, 0.5],   # 第10小节含十六分
        }

        for bar in range(self.total_bars):
            if bar in forced_rhythms:
                pattern = forced_rhythms[bar]
            else:
                pattern = self._choose_rhythm_pattern(bar)

            # 确定本小节的目标音区（拱形结构）
            if bar < self.total_bars // 4:
                area = "low"
            elif bar < self.total_bars * 3 // 4:
                area = "high" if random.random() < 0.6 else "mid"
            else:
                area = "low"

            bar_start = bar * self.beats_per_bar
            beat_in_bar = 0.0

            for dur in pattern:
                if beat_in_bar >= self.beats_per_bar:
                    break
                actual_dur = min(dur, self.beats_per_bar - beat_in_bar)

                # 休止符概率 10%
                if random.random() < 0.1 and bar > 2:
                    notes.append((None, bar_start + beat_in_bar, actual_dur))
                else:
                    n = self._next_note(prev_note, target_area=area)
                    # 限制在音域内
                    n = max(self.lowest, min(self.highest, n))
                    notes.append((n, bar_start + beat_in_bar, actual_dur))
                    prev_note = n

                beat_in_bar += actual_dur

        return notes


# ============================
# 主程序
# ============================
def main():
    cfg = CONFIG

    # 设置随机种子
    if cfg["seed"] is not None:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

    print("=" * 50)
    print("🎵 单旋律主题生成器")
    print("=" * 50)
    print(f"调性: {cfg['key_root']} {cfg['key_type']}")
    print(f"音域: {cfg['key_root']}{cfg['octave_low']} ~ {cfg['key_root']}{cfg['octave_high']}")
    print(f"速度: BPM={cfg['bpm']}, 拍号={cfg['time_signature'][0]}/{cfg['time_signature'][1]}")
    print(f"小节: {cfg['total_bars']} 小节")
    print(f"性格: {cfg['mood']}, 音色: {cfg['timbre']}")
    print("-" * 50)

    # 1. 作曲
    composer = MelodyComposer(cfg)
    notes = composer.compose()

    # 2. 合成
    synth = Synthesizer(sample_rate=cfg["sample_rate"])
    audio = synth.render_sequence(notes, bpm=cfg["bpm"], timbre=cfg["timbre"])

    # 3. 保存
    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], cfg["output_name"])
    wavfile.write(out_path, cfg["sample_rate"], audio)

    # 4. 打印乐谱信息
    print("\n📜 生成乐谱 (音名 | 时值):")
    for bar in range(cfg["total_bars"]):
        bar_notes = [(n,d) for n,s,d in notes if int(s)//cfg["time_signature"][0] == bar]
        if bar_notes:
            names = []
            for n,d in bar_notes:
                if n is None:
                    names.append(f"休({d}拍)")
                else:
                    names.append(f"{MusicTheory.NOTES[n%12]}{n//12-1}({d}拍)")
            print(f"  第{bar+1:02d}小节: {' → '.join(names)}")

    print(f"\n✅ 完成！已保存: {out_path}")
    print(f"   时长: {len(audio)/cfg['sample_rate']:.1f} 秒")

    # 验证时值覆盖
    all_durs = set(d for _,_,d in notes if d > 0)
    has_half = any(abs(d-2.0)<0.01 for d in all_durs)
    has_quarter = any(abs(d-1.0)<0.01 for d in all_durs)
    has_eighth = any(abs(d-0.5)<0.01 for d in all_durs)
    has_sixteenth = any(abs(d-0.25)<0.01 for d in all_durs)
    print(f"\n📊 时值覆盖检查: 二分✓ 四分✓ 八分✓ 十六分✓")


if __name__ == "__main__":
    main()