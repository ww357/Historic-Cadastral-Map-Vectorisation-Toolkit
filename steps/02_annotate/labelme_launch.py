"""
Start labelme with annotation speed-ups and one safeguard that labelme itself lacks.
Takes exactly the same arguments as the `labelme` command.

  1. Drag-rectangle selection (edit mode). Press on empty background and drag:
       left -> right : selects shapes lying FULLY inside the box   (safe for deleting junk)
       right -> left : selects every shape the box TOUCHES          (crossing select)
       Shift+drag    : box select even when starting on a shape / vertex
       Ctrl+drag     : add to the current selection instead of replacing it
     Then press Delete. (labelme can only select a linestrip by hitting one of its
     vertices exactly, which is what makes clearing auto-drawn lines slow.)

  2. Delete without the "permanently delete N shapes?" popup. Ctrl+Z still undoes it.

  3. Never reopen off-screen. labelme restores its last window position without checking
     it is on a connected monitor, so a window last closed on an external display that is
     now unplugged opens as a blank, unreachable taskbar entry (and `--plain` has the same
     problem). The saved position is checked at startup and pulled onto the main screen.

This lives in a wrapper because labelme_config.yaml can only change shortcuts, colours
and flags - it cannot add a tool. The patch is applied to the running labelme in memory;
nothing in the labelme install is modified. If the patch cannot be applied (labelme
internals changed) labelme starts normally and says so.
"""

from __future__ import annotations

import sys

DRAG_PIXELS = 4   # a press-release closer than this is a plain click, not a box


def _seg_hits_rect(a, b, r) -> bool:
    """Liang-Barsky: does segment a-b intersect rectangle r (QRectF)?"""
    x0, y0, dx, dy = a.x(), a.y(), b.x() - a.x(), b.y() - a.y()
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - r.left()), (dx, r.right() - x0),
                 (-dy, y0 - r.top()), (dy, r.bottom() - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def _touches(shape, r) -> bool:
    pts = shape.points
    if shape.shape_type in ("linestrip", "line"):
        return any(r.contains(p) for p in pts) or any(
            _seg_hits_rect(a, b, r) for a, b in zip(pts, pts[1:]))
    return shape.boundingRect().intersects(r)        # polygons, rectangles, points


def fit_saved_window(settings, screens, QtCore) -> str | None:
    """
    labelme restores its last window size/position blindly (resize(); move()), so a window
    last closed on a monitor that is no longer connected reopens OFF-SCREEN: a taskbar entry
    that cannot be brought up. If the saved title bar is not reachable on any current screen,
    shrink the window to fit and centre it on the primary screen. `screens` is a list of
    available-geometry QRects, primary first. Returns a note if it changed anything.
    """
    size, pos = settings.value("window/size"), settings.value("window/position")
    if not (isinstance(size, QtCore.QSize) and isinstance(pos, QtCore.QPoint)) or not screens:
        return None
    strip = QtCore.QRect(pos.x(), pos.y(), size.width(), 40)          # the title-bar strip
    reachable = any(strip.intersected(g).width() >= 150 and strip.intersected(g).height() >= 20
                    for g in screens)
    biggest = max(g.width() for g in screens), max(g.height() for g in screens)
    if reachable and size.width() <= biggest[0] and size.height() <= biggest[1]:
        return None
    home = screens[0]
    new = QtCore.QSize(min(size.width(), int(home.width() * 0.9)),
                       min(size.height(), int(home.height() * 0.9)))
    new_pos = QtCore.QPoint(home.x() + (home.width() - new.width()) // 2,
                            home.y() + (home.height() - new.height()) // 2)
    settings.setValue("window/size", new)
    settings.setValue("window/position", new_pos)
    settings.sync()
    return (f"saved window {size.width()}x{size.height()} at ({pos.x()},{pos.y()}) could not be used on the "
            f"connected screens - moved to {new.width()}x{new.height()} at ({new_pos.x()},{new_pos.y()})")


def install() -> None:
    from labelme.widgets import canvas as cv
    QtCore, QtGui = cv.QtCore, cv.QtGui
    Qt = QtCore.Qt
    Canvas = cv.Canvas
    o_press, o_move = Canvas.mousePressEvent, Canvas.mouseMoveEvent
    o_release, o_paint = Canvas.mouseReleaseEvent, Canvas.paintEvent

    def img_pos(self, ev):
        try:
            return self.transformPos(ev.localPos())
        except AttributeError:                       # older labelme / Qt binding
            return self.transformPos(ev.pos())

    def wid_pos(ev):
        try:
            return QtCore.QPointF(ev.localPos())
        except AttributeError:
            return QtCore.QPointF(ev.pos())

    def press(self, ev):
        self._rb = None
        try:
            if ev.button() == Qt.LeftButton and self.editing() and self.pixmap:
                p = img_pos(self, ev)
                over_something = (
                    getattr(self, "hVertex", None) is not None
                    or getattr(self, "hShape", None) is not None
                    or getattr(self, "hEdge", None) is not None
                    or any(self.isVisible(s) and s.containsPoint(p) for s in self.shapes))
                if (ev.modifiers() & Qt.ShiftModifier) or not over_something:
                    additive = bool(ev.modifiers() & Qt.ControlModifier)
                    self._rb = {"start": p, "end": p, "wstart": wid_pos(ev), "wend": wid_pos(ev),
                                "active": False, "base": list(self.selectedShapes) if additive else []}
        except Exception:
            self._rb = None
        o_press(self, ev)

    def move(self, ev):
        rb = getattr(self, "_rb", None)
        if rb is not None and (ev.buttons() & Qt.LeftButton):
            try:
                rb["end"], rb["wend"] = img_pos(self, ev), wid_pos(ev)
                d = rb["wend"] - rb["wstart"]
                if abs(d.x()) + abs(d.y()) >= DRAG_PIXELS:
                    rb["active"] = True
                self.update()
                return
            except Exception:
                self._rb = None
        o_move(self, ev)

    def release(self, ev):
        rb = getattr(self, "_rb", None)
        if rb is not None and ev.button() == Qt.LeftButton:
            self._rb = None
            if rb["active"]:
                rect = QtCore.QRectF(rb["start"], rb["end"]).normalized()
                crossing = rb["end"].x() < rb["start"].x()
                if crossing:
                    hit = [s for s in self.shapes if self.isVisible(s) and _touches(s, rect)]
                else:
                    hit = [s for s in self.shapes
                           if self.isVisible(s) and rect.contains(s.boundingRect())]
                hit = rb["base"] + [s for s in hit if s not in rb["base"]]
                self.selectShapes(hit)
                self.update()
                return
        o_release(self, ev)

    def paint(self, ev):
        o_paint(self, ev)
        rb = getattr(self, "_rb", None)
        if rb and rb["active"]:
            crossing = rb["end"].x() < rb["start"].x()
            colour = QtGui.QColor(0, 150, 255) if crossing else QtGui.QColor(0, 200, 90)
            p = QtGui.QPainter(self)
            p.setPen(QtGui.QPen(colour, 1, Qt.DashLine if crossing else Qt.SolidLine))
            colour.setAlpha(40)
            p.setBrush(colour)
            p.drawRect(QtCore.QRectF(rb["wstart"], rb["wend"]).normalized())
            p.end()

    Canvas.mousePressEvent, Canvas.mouseMoveEvent = press, move
    Canvas.mouseReleaseEvent, Canvas.paintEvent = release, paint

    # Delete without the confirmation popup (the popup is hard-coded in labelme).
    from labelme import app as lapp
    QtWidgets = getattr(cv, "QtWidgets", None) or __import__("qtpy").QtWidgets
    o_delete = lapp.MainWindow.deleteSelectedShape

    def delete_no_confirm(self):
        box = QtWidgets.QMessageBox
        saved = box.warning
        box.warning = staticmethod(lambda *a, **k: box.Yes)
        try:
            o_delete(self)
        finally:
            box.warning = saved

    lapp.MainWindow.deleteSelectedShape = delete_no_confirm

    # Never reopen off-screen: runs just before labelme builds its window (QApplication exists).
    o_init = lapp.MainWindow.__init__

    def init_with_visible_window(self, *a, **k):
        try:
            app = QtWidgets.QApplication.instance()
            primary = app.primaryScreen()
            order = [primary] + [s for s in app.screens() if s is not primary]
            note = fit_saved_window(QtCore.QSettings("labelme", "labelme"),
                                    [s.availableGeometry() for s in order], QtCore)
            if note:
                print(f"labelme_launch: {note}")
        except Exception as e:                                # a guard must never block startup
            print(f"labelme_launch: window-position check skipped ({e!r})")
        o_init(self, *a, **k)

    lapp.MainWindow.__init__ = init_with_visible_window


def main() -> None:
    try:
        install()
        print("labelme_launch: drag-select + no-confirm delete enabled "
              "(drag on background; Shift = start on a shape; Ctrl = add)")
    except Exception as e:                                # never stop labelme from opening
        print(f"labelme_launch: patch not applied ({e!r}) - starting plain labelme")
    from labelme.__main__ import main as labelme_main
    labelme_main()


if __name__ == "__main__":
    main()
