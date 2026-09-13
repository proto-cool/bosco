"""Moderation labels -> innate aversion.  See config/moderation_v1.yaml."""

from __future__ import annotations

import yaml

from bosco import paths


class Moderation:
    def __init__(self, path=paths.CONFIG / "moderation_v1.yaml") -> None:
        cfg = yaml.safe_load(open(path))
        self.aversive = {str(x).lower() for x in cfg.get("aversive_labels", [])}
        self.any_label = bool(cfg.get("any_label_is_aversive", False))

    @staticmethod
    def label_values(obj) -> set[str]:
        """Label values on an AppView object (post view or profile view)."""
        out: set[str] = set()
        for lab in getattr(obj, "labels", None) or []:
            v = getattr(lab, "val", None)
            if v:
                out.add(str(v).lower())
        return out

    def aversive_labels_on(self, *objs) -> set[str]:
        vals: set[str] = set()
        for o in objs:
            if o is not None:
                vals |= self.label_values(o)
        if self.any_label:
            return vals
        return vals & self.aversive
