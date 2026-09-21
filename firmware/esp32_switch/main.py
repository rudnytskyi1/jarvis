"""ESP32 wall switch: MicroPython firmware for ТЗ F-503.

Hardware: an ESP32 dev board, a servo (SG90/MG996R) on the wall switch, an
optional button, 5 V supply. The firmware is deliberately small — it does three
things:

* subscribes to ``home/<home_id>/switch/<n>/set`` and moves the servo;
* subscribes to ``home/<home_id>/switch/<n>/config`` (retained) and keeps the
  two calibrated angles, so the hub can recalibrate without reflashing;
* publishes ``.../state`` (retained) after every move, so the hub knows the
  switch really moved.

The same firmware runs in every room: only ``config.json`` on the board differs.
"""
import json
import time

try:  # pragma: no cover - MicroPython only
    from machine import PWM, Pin
except ImportError:  # pragma: no cover - lets the file be parsed on a PC
    Pin = PWM = None

SET_TOPIC = "home/{home_id}/switch/{number}/set"
STATE_TOPIC = "home/{home_id}/switch/{number}/state"
CONFIG_TOPIC = "home/{home_id}/switch/{number}/config"

#: Servo pulse widths for 0° and 180°; most SG90-class servos sit here.
PULSE_MIN_US, PULSE_MAX_US = 500, 2500
PWM_FREQUENCY_HZ = 50


class Servo:
    """One servo on one wall switch."""

    def __init__(self, pin):
        self.pwm = PWM(Pin(pin), freq=PWM_FREQUENCY_HZ)
        self.angle = None
        self.closed_angle = 0.0
        self.open_angle = 90.0
        self.dwell_s = 0.6

    def configure(self, payload):
        """Apply what the hub published on the ``config`` topic."""
        if "closed_angle" in payload:
            self.closed_angle = float(payload["closed_angle"])
        if "open_angle" in payload:
            self.open_angle = float(payload["open_angle"])
        if "dwell_s" in payload:
            self.dwell_s = float(payload["dwell_s"])
        return {"closed_angle": self.closed_angle, "open_angle": self.open_angle,
                "dwell_s": self.dwell_s}

    def _pulse_us(self, angle):
        span = max(1.0, 180.0)
        angle = min(max(float(angle), 0.0), span)
        return int(PULSE_MIN_US + (PULSE_MAX_US - PULSE_MIN_US) * angle / span)

    def move(self, angle):
        """Move to one angle and hold long enough for the switch to flip."""
        self.pwm.duty_ns(self._pulse_us(angle) * 1000)
        self.angle = float(angle)
        time.sleep(self.dwell_s)
        self.pwm.duty_ns(0)  # let go: a servo that keeps pushing burns itself out
        return self.angle

    def set_on(self, on):
        return self.move(self.open_angle if on else self.closed_angle)


def handle_command(servo, topic, payload, number):
    """One MQTT message: ``config`` recalibrates, ``set`` moves the switch."""
    try:
        data = json.loads(payload)
    except ValueError:
        return None, {"error": "invalid json"}
    if topic.endswith("/config"):
        servo.configure(data)
        return None, {"state": "configured", **servo.configure({})}
    if topic.endswith("/set"):
        capability = data.get("capability") or data.get("command")
        value = data.get("value")
        if capability in ("on_off", "set"):
            angle = servo.set_on(bool(value))
        elif capability == "press":
            angle = servo.move(servo.open_angle)
            servo.move(servo.closed_angle)
        else:
            return None, {"error": "unsupported capability " + str(capability)}
        return None, {"state": "on" if value else "off", "angle": angle, "number": number}
    return None, None


def main():  # pragma: no cover - runs on the board
    import network
    from umqtt.simple import MQTTClient

    with open("config.json") as handle:
        config = json.load(handle)
    station = network.WLAN(network.STA_IF)
    station.active(True)
    station.connect(config["wifi_ssid"], config["wifi_password"])
    while not station.isconnected():
        time.sleep(0.5)
    servo = Servo(config.get("servo_pin", 18))
    number = config["number"]
    topics = {
        "set": SET_TOPIC.format(home_id=config["home_id"], number=number),
        "config": CONFIG_TOPIC.format(home_id=config["home_id"], number=number),
        "state": STATE_TOPIC.format(home_id=config["home_id"], number=number),
    }
    client = MQTTClient(config["device_id"], config.get("broker", "192.168.1.10"), 1883)

    def on_message(topic, payload):  # pragma: no cover - MQTT callback
        _, state = handle_command(servo, topic.decode(), payload.decode(), number)
        if state is not None:
            client.publish(topics["state"], json.dumps(state), retain=True)

    client.set_callback(on_message)
    client.connect()
    client.subscribe(topics["set"])
    client.subscribe(topics["config"])
    while True:
        client.check_msg()
        time.sleep(0.1)


if __name__ == "__main__":  # pragma: no cover
    main()
