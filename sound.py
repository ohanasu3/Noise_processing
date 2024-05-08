# Copyright (c) 2013--2023, librosa development team.

# Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import soundfile as sf

#ステレオ波形
file_name="00.wav"#音声ファイルを入れる
y,sr=librosa.load(file_name,duration=5,sr=8000)
librosa.display.waveshow(y,sr=sr,label='origin')
plt.show()

#音声を倍音と衝撃音で分離
y_harm, y_perc = librosa.effects.hpss(y)
librosa.display.waveshow(y_perc, sr=sr, color='r',alpha=0.5,label='percussive')
sf.write("00.wav",y_perc,sr,subtype="PCM_24")
plt.show()

#パワースペクトル（ぶつける前）
file_name="00.wav"
y,sr=librosa.load(file_name,duration=16,sr=8000)
S=np.abs(librosa.stft(y))
S_left=librosa.stft(y,center=False)
D_short=librosa.stft(y,hop_length=64)
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()

#ぶつける
i=496 #音声データによって数値を変更
tone = librosa.tone(80,duration=200)
ghar=len(tone)-len(y_perc)
gasd=ghar-len(y_perc)
time = np.arange(len(y_perc)+gasd,len(tone)) /sr
huf=y_perc-(time/i)
print(huf)
librosa.display.waveshow(y_perc-(time/i),color='r',alpha=0.5,sr=sr)
sf.write("00.wav",huf,sr,subtype="PCM_24")
plt.show()

#音声を倍音と衝撃音で分離
file_name="00.wav"
y,sr=librosa.load(file_name,duration=5,sr=8000)
y_harm, y_perc = librosa.effects.hpss(y)
#print(y_perc) 配列の数値確認用
librosa.display.waveshow(y_perc, sr=sr, color='r',alpha=0.5,label='percussive')
sf.write("0000.wav",y_perc,sr,subtype="PCM_24")
plt.show()

#パワースペクトル（ぶつけた後） (ぶつける前同じ)
file_name="0000.wav"
y,sr=librosa.load(file_name,duration=16,sr=8000)
S=np.abs(librosa.stft(y))
S_left=librosa.stft(y,center=False)
D_short=librosa.stft(y,hop_length=64)
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()