from RealtimeSTT import AudioToTextRecorder


def on_partial(text):
    # промежуточный текст, обновляется пока говоришь
    print(f"\r… {text}          ", end="", flush=True)

def on_final(text):
    # финальный текст фразы (после паузы в речи)
    print(f"\r{text}                    ")

if __name__ == "__main__":
    recorder = AudioToTextRecorder(
        model="large-v3",                    # финальное распознавание
        realtime_model_type="small",          # быстрая модель для live-текста
        language="",                          # автодетект (можно "ru" или "en" принудительно)
        enable_realtime_transcription=True,
        on_realtime_transcription_update=on_partial,
        realtime_processing_pause=0.2,        # как часто обновлять live-текст
        post_speech_silence_duration=0.7,     # пауза (сек), после которой фраза считается законченной
        device="cuda",
    )
    print("Слушаю")
    while True:
        recorder.text(on_final)
