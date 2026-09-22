"""Зоны кадра: полигоны дома и маска «не анализировать» (ТЗ F-309).

Владелец рисует в кадре «дверь», «стол», «кровать» и маску — область, которую
анализировать нельзя. Полигоны живут в конфиге дома в НОРМАЛИЗОВАННЫХ
координатах (0…1), поэтому зона не зависит от разрешения камеры: и клиент, и
хаб, и админка считают одно и то же.

Что делает этот модуль:

* говорит, в какой зоне лежит находка (`zone_at` по центру объекта) — это
  подпись «на столе» в ответе F-305 и источник событий `zone_entered` F-309;
* говорит, попадает ли находка в маску (`masked_at`) — такое НЕ индексируется
  и не описывается, даже если кадр всё-таки доехал: обещание «эту область не
  смотрят» важнее полноты памяти объектов;
* отдаёт список зон для админки (`describe`), чтобы владелец видел, что
  именно настроено, не читая yaml.

Точка внутри полигона считается честным ray casting: границы зон не
«размазываются» и не угадываются. Область без зон — это пустая строка, а не
выдуманное «комната».
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from common.frame_zones import polygon_contains

Point = tuple[float, float]


@dataclass(frozen=True)
class Zone:
    """One polygon of one room's frame."""

    name: str
    points: tuple[Point, ...]
    mask: bool = False

    def contains(self, x: float, y: float) -> bool:
        """Ray casting: точка внутри полигона (тот же код, что у клиента)."""
        return polygon_contains(self.points, x, y)


def _zone_from(item: Any) -> Zone | None:
    """One config entry (or plain mapping) as a :class:`Zone`, or ``None``."""
    if item is None:
        return None
    name = " ".join(str(getattr(item, "name", "") or "").split())[:60]
    raw = getattr(item, "points", None) or ()
    try:
        points = tuple((float(point[0]), float(point[1])) for point in raw)
    except (TypeError, ValueError, IndexError):
        return None
    if not name or len(points) < 3:
        return None
    return Zone(name=name, points=points, mask=bool(getattr(item, "mask", False)))


class FrameZones:
    """The zones of one home, asked by the indexer and the admin panel."""

    def __init__(self, zones: Iterable[Any] = ()) -> None:
        found: list[Zone] = []
        for item in zones or ():
            zone = _zone_from(item)
            if zone is not None:
                found.append(zone)
        self.zones: tuple[Zone, ...] = tuple(found)

    def __bool__(self) -> bool:
        return bool(self.zones)

    def describe(self) -> list[dict[str, Any]]:
        """Что настроено в этом доме — для админки и `/health`."""
        return [{"name": zone.name, "mask": zone.mask,
                 "points": [[x, y] for x, y in zone.points]}
                for zone in self.zones]

    def zone_at(self, bbox: Sequence[float], size: tuple[int, int] | None = None) -> str:
        """Name of the zone the bbox CENTER falls into; ``""`` when none does.

        Маска сюда не попадает: область «не анализировать» отвечает через
        :meth:`masked_at`, и смешивать эти два ответа нельзя.
        """
        point = _center(bbox, size)
        if point is None:
            return ""
        for zone in self.zones:
            if not zone.mask and zone.contains(*point):
                return zone.name
        return ""

    def masked_at(self, bbox: Sequence[float], size: tuple[int, int] | None = None) -> bool:
        """Лежит ли центр находки в области «не анализировать» (ТЗ F-309)."""
        point = _center(bbox, size)
        if point is None:
            return False
        return any(zone.mask and zone.contains(*point) for zone in self.zones)


def _center(bbox: Sequence[float], size: tuple[int, int] | None) -> Point | None:
    """Normalized center of a pixel bbox; ``None`` when there is nothing to ask."""
    width, height = (int(size[0]), int(size[1])) if size else (0, 0)
    if width <= 0 or height <= 0 or len(bbox) < 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in bbox[:4])
    except (TypeError, ValueError):
        return None
    x = ((left + right) / 2.0) / width
    y = ((top + bottom) / 2.0) / height
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def zones_for_homes(homes: Iterable[Any]) -> dict[str, FrameZones]:
    """``home_id -> FrameZones`` из конфига домов (ТЗ F-309)."""
    found: dict[str, FrameZones] = {}
    for home in homes or ():
        home_id = str(getattr(home, "home_id", "") or "")
        if home_id:
            found[home_id] = FrameZones(getattr(home, "zones", ()) or ())
    return found


__all__ = ["FrameZones", "Zone", "zones_for_homes"]
