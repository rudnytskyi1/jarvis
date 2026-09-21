"""Offscreen Qt check of the actual confirmation dialog; no profiles involved."""
import json
import os
import sys
import time
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--disable-gpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QMessageBox

from client.overlay import OverlayHUD

hud = OverlayHUD()
qapp, view, bridge = hud._create_qt_objects()
hud._app, hud._view, hud._bridge = qapp, view, bridge
results = []
hud.confirm_voice('Test profile only', results.append)
qapp.processEvents()
assert bridge._voice_box.defaultButton() == bridge._voice_box.button(QMessageBox.Cancel)
assert not hud.suspend_capture(.01)
bridge._voice_box.button(QMessageBox.Cancel).click()
qapp.processEvents()
assert results == [False]
hud.confirm_voice('Test profile only', results.append)
qapp.processEvents()
bridge._voice_box.button(QMessageBox.Save).click()
qapp.processEvents()
assert results == [False, True]
hud.confirm_voice('Test profile only', results.append)
qapp.processEvents()
hud.cancel_voice_confirmation()
qapp.processEvents()
assert results == [False, True, False] and bridge._voice_box is None
hud.confirm_voice({'old_name': 'TestOld', 'new_name': 'TestNew', 'merge': True}, results.append)
qapp.processEvents()
assert 'TestOld' in bridge._voice_box.text() and 'combines two' in bridge._voice_box.informativeText()
bridge._voice_box.button(QMessageBox.Cancel).click()
qapp.processEvents()

def js(code):
    output = []
    view.page().runJavaScript(code, output.append)
    deadline = time.monotonic() + 5
    while not output and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(.01)
    assert output, 'JavaScript callback timed out'
    return output[0]

deadline = time.monotonic() + 8
while js('typeof window.hudTranscript') != 'function' and time.monotonic() < deadline:
    qapp.processEvents()
assert js('typeof window.hudTranscript') == 'function'
js("window.hudState({state:'listening'}); window.hudTranscript({text:'hello',person:'Test',provisional:true}); window.hudTranscript({text:'hello there',person:'Test',provisional:true});")
snapshot = json.loads(js("JSON.stringify({count:document.querySelectorAll('#live-utterance').length,text:document.getElementById('live-utterance').lastChild.textContent,person:document.getElementById('person').textContent})"))
assert snapshot == {'count': 1, 'text': 'hello there', 'person': 'Test · checking'}
js("window.hudChat({person:'Test',messages:[],question:'hello there'});")
assert js("document.querySelectorAll('.message.user').length") == 1
assert js("document.querySelectorAll('#live-utterance').length") == 0
view.close()
print('PASS: confirmation, rename warning, capture blocked, live caption replacement and final chat; no profile writes.')
