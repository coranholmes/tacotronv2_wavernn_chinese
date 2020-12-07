import os, glob
import librosa
import soundfile as sf


paths = glob.glob(os.path.join(os.getcwd(), 'wav', '*.wav'))
paths.sort()

for path in paths:
    print(path)
    y, sr = librosa.load(path, sr=16000)
    # y_16k = librosa.resample(y, sr, 16000)

    # TODO: 这种方式在改变采样率的同时，位深度也会发生变化，建议用pysoundfile进行降采样
    # librosa.output.write_wav(path, y, 16000)
    #
    # data, samplerate = sf.read(path)
    sf.write(path, y, sr, subtype='PCM_16')
