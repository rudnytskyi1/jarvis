# Разбор сохранённых запросов

Сгенерировано `scripts/audit_requests.py` (см. его докстроку про источники).

- реплик в логе: **297**
- цепочек в базе панели (`turn_events`): **112** (901 шагов)

## Что не так, по классам

| класс проблемы | сколько ходов |
|---|---|
| degraded | 75 |
| slow first audio | 54 |
| slow turn | 44 |
| tool failed | 27 |

## Проблемные ходы (107)

### 177. Rowan AI, can you open the

- tool failed: generate_image, generate_image, generate_image, generate_image, generate_image, generate_image, generate_image, generate_image, inspect_photo, generate_image, generate_image, generate_image, generate_image, generate_image, generate_image, pc_control, list_people, generate_image, list_people, generate_image, generate_image, generate_image, generate_image, generate_image, generate_image, generate_image, generate_image, generate_image, list_people, generate_image, generate_image, generate_image
- degraded: diarization: no diarized transcript within 0.7 s; diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s
- slow turn: 6068 ms
- slow first audio: 6015 ms (budget 1200)

### 181. Rowan AI, can you open the Google Chrome browser, go to youtube.com, search up for MrBeast

- degraded: diarization: no diarized transcript within 0.7 s

### 182. Rowan AI, can you open the Google Chrome browser, go to youtube.com, search up for Mr. Beast

- degraded: stt: over its 0.70 s budget; still waiting, up to 45 s

### 183. Rowan AI, can you open the Google Chrome browser, go to youtube.com, search up for Mr. Beast

- tool failed: browser_control
- slow turn: 24402 ms
- slow first audio: 18938 ms (budget 1200)

### 185. Rowan AI. On the current screen in the chrome browser there is results for MrBeast. Can you click on the first result to open his channel and go with it?

- degraded: diarization: no diarized transcript within 0.7 s

### 186. Rowan AI. On the current screen in the chrome browser there is results for MrBeast. Can you click on the first result to open his channel and go to videos?

- degraded: stt: over its 0.70 s budget; still waiting, up to 45 s

### 187. Rowan AI. On the current screen in the chrome browser there is results for MrBeast. Can you click on the first result to open his channel and go to videos?

- slow turn: 10747 ms
- slow first audio: 10688 ms (budget 1200)

### 194. Rowan AI, remember my face

- tool failed: generate_image
- degraded: diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s
- slow turn: 4807 ms
- slow first audio: 3985 ms (budget 1200)

### 196. Hey Rowan AI, please remember my face.

- degraded: diarization: no diarized transcript within 0.7 s

### 200. Rowan AI please remember my voice

- tool failed: generate_image, inspect_photo, generate_image, generate_image, generate_image, generate_image, generate_image

### 201. Rowan AI please remember how my normal voice sounds

- degraded: diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s

### 216. just wait, I told it to them, dog, I tell you to sit down, clutch, tongue in tongue, spent my time, tongue in tongue, send those shots, murder, shooter, just pu

- tool failed: generate_image, telegram_send, generate_image, generate_image, generate_image, generate_image, generate_image, generate_image, list_people, generate_image, generate_image, generate_image, generate_image, list_people, generate_image, generate_image, generate_image, list_people, generate_image, generate_image
- degraded: diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s

### 217. Rowan AI, take a

- degraded: diarization: no diarized transcript within 0.7 s

### 218. Rowan AI, get rid of that dog, Rowan AI, close, close the screen,

- tool failed: generate_image, generate_image, generate_image, generate_image, generate_image, generate_image, inspect_photo, list_people, generate_image, generate_image, generate_image, generate_image, pc_control, generate_image, generate_image
- degraded: diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s; diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s
- slow turn: 5118 ms
- slow first audio: 3718 ms (budget 1200)

### 219. Rowan AI, get rid of that dog, Rowan AI, close, close the screen.

- degraded: diarization: no diarized transcript within 0.7 s

### 220. Rowan AI, get rid of that dog, Rowan AI, close, close the screen.

- degraded: stt: over its 0.70 s budget; still waiting, up to 45 s

### 221. Rowan AI, get rid of that dog, Rowan AI, close, close the screen.

- slow first audio: 2750 ms (budget 1200)

### 222. Rowan AI, can you take a picture using camera

- slow first audio: 3937 ms (budget 1200)

### 226. Rowan AI, is there a browser window currently open?

- degraded: diarization: no diarized transcript within 0.7 s

### 228. Rowan AI, is there a browser window currently open?

- slow turn: 6367 ms
- slow first audio: 6359 ms (budget 1200)

### 239. browser window, then reopen it and go to youtube.com and not just in the search line can you go to youtube search like actually on the page click search type Mr

- degraded: diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s

### 241. Rowan AI can you close the current browser window then reopen it and go to youtube.com and not just in the search line can you go to youtube search like actuall

- tool failed: pc_control
- slow turn: 8022 ms
- slow first audio: 7781 ms (budget 1200)

### 242. Rowan AI, so can you make

- tool failed: pc_control
- degraded: diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s
- slow turn: 4492 ms
- slow first audio: 4234 ms (budget 1200)

### 246. Rowan AI, so can you make Theodric kiss with John the system and display this picture on the screen? No.

- degraded: diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s

### 247. Rowan AI, so can you make Theodric kiss with John the system and display this picture on the screen? No.

- slow first audio: 3562 ms (budget 1200)

### 248. Rowan AI, can you play the video on the screen?

- tool failed: generate_image
- degraded: diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s; diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s
- slow turn: 6502 ms
- slow first audio: 6453 ms (budget 1200)

### 253. Rowan AI, can you play the video on the screen? It says I really see you to save some Chinese kids from...

- degraded: diarization: no diarized transcript within 0.7 s

### 255. Rowan AI, can you play the video on the screen, it says I really see you to save some Chinese kids from illegal labor, can you open it?

- degraded: stt: over its 0.70 s budget; still waiting, up to 45 s
- slow turn: 4771 ms
- slow first audio: 4766 ms (budget 1200)

### 256. Rowan AI, there is a video on the screen can you open it

- degraded: diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s
- slow turn: 8470 ms
- slow first audio: 8437 ms (budget 1200)

### 257. Rowan AI, can you take this

- degraded: diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s

### 266. Can you take a screenshot and make every person wear a rainbow flag right now?

- degraded: diarization: no diarized transcript within 0.7 s

### 268. Rowan AI, can you take a screenshot and make every person wear a baseball flag right now?

- slow turn: 8174 ms
- slow first audio: 8172 ms (budget 1200)

### 278. Hey Rowan AI, can you play a War Thunder advertisement by Morgenstern on YouTube? The kids will be here in a couple of days to start their school year and we st

- degraded: diarization: no diarized transcript within 0.7 s; stt: over its 0.70 s budget; still waiting, up to 45 s

### 279. Hey Rowan AI, can you play a War Thunder advertisement by Morgenstern on YouTube? The kids will be here in a couple of days to start their school year and we st

- tool failed: browser_control
- slow turn: 8527 ms
- slow first audio: 8094 ms (budget 1200)

### 284. Rowan AI, take a screenshot of the computer screen and use a nano banana make everybody

- degraded: diarization: no diarized transcript within 0.7 s

### 285. Rowan AI, take a screenshot of the screen of the computer screen and using nano banana make everybody wear

- degraded: stt: over its 0.70 s budget; still waiting, up to 45 s

### 286. Rowan AI, take a screenshot of the screen of the computer screen and using nano banana make everybody wear

- slow turn: 4560 ms
- slow first audio: 4141 ms (budget 1200)

### 295. and make everybody worry spider-man soon.

- degraded: diarization: no diarized transcript within 0.7 s

### 296. Rowan AI, take a screenshot of the current screen and make everybody wear a spiderman suit.

- degraded: stt: over its 0.70 s budget; still waiting, up to 45 s

### 297. Rowan AI, take a screenshot of the current screen and make everybody wear a spiderman suit.

- tool failed: generate_image
- slow turn: 9946 ms
- slow first audio: 9921 ms (budget 1200)
