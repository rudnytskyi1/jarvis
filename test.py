### this is my code AI don't touch it and read, it's just a test

# Тест распознавания голосов для Jarvis
# Установка:
#   pip install resemblyzer sounddevice soundfile numpy
#
# Использование:
#   python voice_id.py enroll Дима   - записать 10 сек и сохранить голос Димы
#   python voice_id.py who           - записать 4 сек и определить, кто говорит
#   python voice_id.py list          - показать, кто в базе
import json
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from resemblyzer import VoiceEncoder, preprocess_wav

SAMPLE_RATE = 16000
DB_FILE = Path("voices.json")

# Пороги (подкрути под свой микрофон после тестов)
KNOWN_THRESHOLD = 0.75    # выше - точно узнал
UNKNOWN_THRESHOLD = 0.60  # ниже - точно незнакомец

encoder = VoiceEncoder()  # при первом запуске скачает модель (~17 МБ)


def record(seconds: float) -> np.ndarray:
    print(f"Говори! Запись {seconds} сек...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    print("Записано.")
    return audio.flatten()


def get_embedding(audio: np.ndarray) -> np.ndarray:
    wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
    return encoder.embed_utterance(wav)


def load_db() -> dict:
    if DB_FILE.exists():
        raw = json.loads(DB_FILE.read_text(encoding="utf-8"))
        return {name: [np.array(e) for e in embs] for name, embs in raw.items()}
    return {}


def save_db(db: dict):
    raw = {name: [e.tolist() for e in embs] for name, embs in db.items()}
    DB_FILE.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")


def enroll(name: str):
    db = load_db()
    audio = record(10)
    emb = get_embedding(audio)
    db.setdefault(name, []).append(emb)
    save_db(db)
    print(f"Голос сохранён: {name} (записей: {len(db[name])})")


def who():
    db = load_db()
    if not db:
        print("База пустая. Сначала: python voice_id.py enroll Имя")
        return
    audio = record(4)

    t0 = time.perf_counter()
    emb = get_embedding(audio)          # первый (холодный) прогон
    t1 = time.perf_counter()
    emb = get_embedding(audio)          # второй (тёплый) прогон
    t2_warm = time.perf_counter()
    print(f"Первый прогон: {1000*(t1-t0):.1f} мс, второй: {1000*(t2_warm-t1):.1f} мс")

    scores = {}
    for name, embs in db.items():
        mean_emb = np.mean(embs, axis=0)
        mean_emb /= np.linalg.norm(mean_emb)
        scores[name] = float(np.dot(emb, mean_emb))
    t2 = time.perf_counter()

    print(f"\nВремя: embedding {1000*(t1-t0):.1f} мс, "
          f"сравнение {1000*(t2-t1):.2f} мс, "
          f"итого {1000*(t2-t0):.1f} мс")

    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]

    print("\nПохожесть по всем голосам:")
    for name, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {name}: {s:.3f}")

    print()
    if best_score >= KNOWN_THRESHOLD:
        print(f">>> Это {best_name} (уверенно, {best_score:.3f})")
    elif best_score <= UNKNOWN_THRESHOLD:
        print(f">>> Незнакомый голос ({best_score:.3f}) - 'как тебя зовут?'")
    else:
        print(f">>> Похоже на {best_name}, но не уверен ({best_score:.3f}) - серая зона")


def list_voices():
    db = load_db()
    if not db:
        print("База пустая.")
        return
    for name, embs in db.items():
        print(f"  {name}: {len(embs)} запись(ей)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__ or "Команды: enroll <имя> | who | list")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "enroll" and len(sys.argv) >= 3:
        enroll(sys.argv[2])
    elif cmd == "who":
        who()
    elif cmd == "list":
        list_voices()
    else:
        print("Команды: enroll <имя> | who | list")
