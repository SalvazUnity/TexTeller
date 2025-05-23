class Point:
    def __init__(self, x: int, y: int):
        self.x = int(x)
        self.y = int(y)

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"


class Bbox:
    THREADHOLD = 0.4

    def __init__(self, x, y, h, w, label: str = None, confidence: float = 0, content: str = None):
        self.p = Point(x, y)
        self.h = int(h)
        self.w = int(w)
        self.label = label
        self.confidence = confidence
        self.content = content

    @property
    def ul_point(self) -> Point:
        return self.p

    @property
    def ur_point(self) -> Point:
        return Point(self.p.x + self.w, self.p.y)

    @property
    def ll_point(self) -> Point:
        return Point(self.p.x, self.p.y + self.h)

    @property
    def lr_point(self) -> Point:
        return Point(self.p.x + self.w, self.p.y + self.h)

    def same_row(self, other) -> bool:
        if (self.p.y >= other.p.y and self.ll_point.y <= other.ll_point.y) or (
            self.p.y <= other.p.y and self.ll_point.y >= other.ll_point.y
        ):
            return True
        if self.ll_point.y <= other.p.y or self.p.y >= other.ll_point.y:
            return False
        return 1.0 * abs(self.p.y - other.p.y) / max(self.h, other.h) < self.THREADHOLD

    def __lt__(self, other) -> bool:
        """
        from top to bottom, from left to right
        """
        if not self.same_row(other):
            return self.p.y < other.p.y
        else:
            return self.p.x < other.p.x

    def __repr__(self) -> str:
        return f"Bbox(upper_left_point={self.p}, h={self.h}, w={self.w}), label={self.label}, confident={self.confidence}, content={self.content})"
