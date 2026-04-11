"""
Topology repair for vectorised boundary lines.

Closes two gap types by adding short bridging segments — existing geometry
is never moved or modified.

  Type 1 — T-junction gap
    A dangling endpoint is extended to snap onto the nearest interior point
    of a nearby line. Covers cases where a boundary almost meets a
    perpendicular line but falls short.

  Type 2 — Broken collinear fragments
    Pairs of dangling endpoints within snap_distance are bridged. An optional
    angle_tolerance restricts this to endpoints approaching from similar
    directions, which avoids bridging two unrelated boundaries that happen
    to end near each other.

snap_distance is the primary guard against false connections — keep it
conservative. A good starting point is 2–3× the Douglas-Peucker simplify
tolerance used in vectorise.py.

Usage (via vectorise.py config — not called directly):
    config.yaml → vectorise.boundaries.topology_repair.enabled: true
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from shapely.strtree import STRtree


def _approach_bearing(coords: list, ep_idx: int) -> float:
    """
    Undirected bearing [0°, 180°) of the segment immediately approaching
    an endpoint. Undirected so anti-parallel lines still match.
    ep_idx: 0 = start of line, -1 = end of line.
    """
    if len(coords) < 2:
        return 0.0
    if ep_idx == 0:
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
    else:
        dx = coords[-1][0] - coords[-2][0]
        dy = coords[-1][1] - coords[-2][1]
    return np.degrees(np.arctan2(dy, dx)) % 180


def _angle_diff(a: float, b: float) -> float:
    """Smallest difference between two undirected bearings in [0°, 180°)."""
    diff = abs(a - b) % 180
    return min(diff, 180.0 - diff)


def repair_topology(
    gdf: gpd.GeoDataFrame,
    snap_distance: float,
    angle_tolerance: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Add bridging segments to close T-junction gaps and broken fragments.

    Parameters
    ----------
    gdf              : GeoDataFrame of LineString geometries (world coords)
    snap_distance    : maximum gap to bridge, in CRS units (metres for BNG)
    angle_tolerance  : maximum bearing difference (degrees) for endpoint→endpoint
                       bridges. None = no angle check (connects any nearby pair).
                       Use ~25° to restrict to collinear fragments only.

    Returns
    -------
    GeoDataFrame with original lines + bridging segments appended.
    An 'is_bridge' column distinguishes original (False) from added (True).
    If no bridges are needed, returns gdf unchanged (with is_bridge column added).
    """
    # Work only on valid LineStrings
    valid_mask = gdf.geometry.apply(
        lambda g: g is not None and not g.is_empty and g.geom_type == "LineString"
    )
    lines      = list(gdf.loc[valid_mask, "geometry"])
    line_index = list(gdf.loc[valid_mask].index)  # original GDF index

    if len(lines) < 2:
        gdf = gdf.copy()
        gdf["is_bridge"] = False
        return gdf

    tree = STRtree(lines)
    bridges  = []
    seen_pairs = set()  # deduplication — each physical gap gets one bridge

    # Interior-snap threshold: if the nearest point on the other line is within
    # this fraction of snap_distance from one of that line's endpoints, treat it
    # as an endpoint connection (handled by the endpoint→endpoint path instead)
    interior_ep_threshold = snap_distance * 0.25

    for i, line in enumerate(lines):
        coords = list(line.coords)
        if len(coords) < 2:
            continue

        for ep_idx in (0, -1):
            endpoint    = Point(coords[ep_idx])
            bearing_i   = _approach_bearing(coords, ep_idx)
            search_box  = endpoint.buffer(snap_distance)

            best_dist  = np.inf
            best_snap  = None

            for j in tree.query(search_box):
                if j == i:
                    continue
                other        = lines[j]
                other_coords = list(other.coords)

                # ----------------------------------------------------------
                # Type 1 — endpoint → interior of other line  (T-junction)
                # ----------------------------------------------------------
                d_to_line = endpoint.distance(other)
                if 0 < d_to_line < snap_distance:
                    snap_pt, _ = nearest_points(other, endpoint)
                    # Reject if the snap point is near one of other's endpoints
                    # (that case is handled by Type 2 below, more precisely)
                    near_start = snap_pt.distance(Point(other_coords[0]))
                    near_end   = snap_pt.distance(Point(other_coords[-1]))
                    if (near_start > interior_ep_threshold and
                            near_end > interior_ep_threshold and
                            d_to_line < best_dist):
                        best_dist = d_to_line
                        best_snap = snap_pt

                # ----------------------------------------------------------
                # Type 2 — endpoint → endpoint  (gap / broken fragment)
                # ----------------------------------------------------------
                for other_ep_idx, other_ep_coords in (
                    (0,  other_coords[0]),
                    (-1, other_coords[-1]),
                ):
                    other_ep = Point(other_ep_coords)
                    d = endpoint.distance(other_ep)
                    if d == 0 or d >= snap_distance:
                        continue

                    # Optional angle check — restricts to roughly collinear lines
                    if angle_tolerance is not None:
                        bearing_j = _approach_bearing(other_coords, other_ep_idx)
                        if _angle_diff(bearing_i, bearing_j) > angle_tolerance:
                            continue

                    if d < best_dist:
                        best_dist = d
                        best_snap = other_ep

            if best_snap is None or best_dist == 0:
                continue

            # Deduplicate by canonical coordinate pair
            key = frozenset([
                (round(endpoint.x,    6), round(endpoint.y,    6)),
                (round(best_snap.x, 6), round(best_snap.y, 6)),
            ])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            bridges.append(LineString([endpoint, best_snap]))

    # Mark original lines
    result = gdf.copy()
    result["is_bridge"] = False

    if not bridges:
        return result

    # Build bridge GDF — carry scalar columns as NaN, set is_bridge=True
    scalar_cols = [c for c in gdf.columns if c != "geometry"]
    bridge_rows = {c: [None] * len(bridges) for c in scalar_cols}
    bridge_rows["is_bridge"] = [True] * len(bridges)
    bridge_rows["geometry"]  = bridges

    bridge_gdf = gpd.GeoDataFrame(bridge_rows, crs=gdf.crs)

    result = gpd.GeoDataFrame(
        pd.concat([result, bridge_gdf], ignore_index=True),
        crs=gdf.crs,
    )
    result["length"] = result.geometry.length

    return result
