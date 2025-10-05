from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ObjectSlot:
    oid: str
    features: Dict[str, float] = field(default_factory=dict)
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class Relation:
    subj: str
    rel: str
    obj: str
    confidence: float = 1.0


@dataclass
class SituationalModel:
    objects: Dict[str, ObjectSlot] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def upsert_object(self, oid: str, features: Dict[str, float] | None = None, properties: Dict[str, str] | None = None) -> None:
        o = self.objects.get(oid)
        if o is None:
            o = ObjectSlot(oid=oid)
            self.objects[oid] = o
        if features:
            for k, v in features.items():
                o.features[k] = float(v)
        if properties:
            o.properties.update(properties)

    def add_relation(self, subj: str, rel: str, obj: str, confidence: float = 1.0) -> None:
        self.relations.append(Relation(subj=subj, rel=rel, obj=obj, confidence=float(confidence)))

    def clear(self) -> None:
        self.objects.clear()
        self.relations.clear()
        self.tags.clear()
