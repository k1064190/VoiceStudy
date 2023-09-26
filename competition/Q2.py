import sys
import wave
import io

class WAVFile:
    def __init__(self, filename):
        self.filename = filename

    def read(self):



for arg in sys.argv[1:]:
    # 예시 : python3 Q2.py wav_list.txt Q2.json
    file_list = arg[1]
    with open(file_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # 예 : ./data/speech_001.wav
            line = line.strip()
            # wav 파일을 읽는다
            with open(line, 'rb') as wavfile:
                wavdata = wavfile.read()
                # wav 파일의 에러를 확인한다.(헤더만 있는 경우, 데이터만 있는 경우, 데이터 값이 없는 경우, 클리핑 에러)







