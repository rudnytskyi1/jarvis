"""Разбор интеркома: «скажи Максу, что я иду» (ТЗ F-601)."""
from __future__ import annotations

import pytest

from hub import intercom
from hub.intercom import IntercomMessage, IntercomStatus, intercom_reply, intercom_request


@pytest.mark.parametrize("text,to,body", [
    ("скажи Максу, что я иду", "Максу", "я иду"),
    ("Скажи Максу что я иду!", "Максу", "я иду"),
    ("передай Максу: ок", "Максу", "ок"),
    ("сообщи Максу, что я опоздаю на десять минут", "Максу", "я опоздаю на десять минут"),
    ("скажи Марии Петровне, что я взял ключи", "Марии Петровне", "я взял ключи"),
    ("tell Max that I am coming", "Max", "I am coming"),
    ("Tell Max I am on my way", "Max", "I am on my way"),
    ("pass on to Max: ok", "Max", "ok"),
    ("send to Max that I am late", "Max", "I am late"),
    ("dile a Max que voy en camino", "Max", "voy en camino"),
    ("dile a Max: ok", "Max", "ok"),
    ("avísale a Max que llego tarde", "Max", "llego tarde"),
])
def test_an_intercom_phrase_names_the_person_and_the_message(text, to, body):
    asked = intercom_request(text)
    assert asked is not None, text
    assert asked.to == to and asked.text == body
    assert asked.matched == " ".join(text.split())


@pytest.mark.parametrize("text", [
    "", "включи свет", "скажи мне, что делать", "tell me a joke",
    "кто дома?", "напомни через 20 минут позвонить маме",
    "скажи время", "add Max to my contacts",
])
def test_other_phrases_are_left_to_the_model(text):
    assert intercom_request(text) is None


@pytest.mark.parametrize("text,body", [
    ("передай ему: ок", "ок"),
    ("скажи ей, что я буду в восемь", "что я буду в восемь"),
    ("tell him: ok", "ok"),
    ("pass on to them that I am ready", "that I am ready"),
    ("dile: ok", "ok"),
])
def test_a_reply_is_recognised_without_a_name(text, body):
    said = intercom_reply(text)
    assert said is not None, text
    assert said.to == "" and said.text == body


@pytest.mark.parametrize("text", ["", "передай Максу: ок", "включи свет", "tell Max I am late"])
def test_a_reply_does_not_swallow_a_named_message(text):
    assert intercom_reply(text) is None


def test_a_message_is_a_strict_model_with_a_waiting_state():
    message = IntercomMessage(message_id="m-1", home_id="kyiv", origin_home="livingroom",
                              from_person="p-amy", to_person="p-max", text="я иду")
    assert message.status is IntercomStatus.QUEUED and message.waiting is True
    assert message.kind is intercom.IntercomKind.NOTE
    with pytest.raises(ValueError):
        IntercomMessage(message_id="m-2", text="я иду", unknown="nope")
