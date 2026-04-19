#!/usr/bin/env python3
"""
Export Sionna Large Radio Maps outputs (rm_*.npz + ground.ply + bboxes.npz)
to a georeferenced tiled GeoTIFF and an optional KMZ overlay.

What this script exports
------------------------
The repository saves one NPZ per tile with:
  - rm: maximum path gain on each face of the measurement mesh
  - tx_positions: transmitter positions used for the tile
  - measurement_z_offset: z-offset applied to the measurement mesh

This script:
  1. Reads each tile's `rm_XXXXXXXX.npz`
  2. Reads the matching `ground.ply`
  3. Computes triangle centroids for the ground mesh
  4. Converts local tile XY coordinates back to lon/lat
  5. Aggregates the per-face path-gain values into a raster mosaic
  6. Writes a tiled GeoTIFF (EPSG:4326)
  7. Optionally writes a KMZ GroundOverlay for Google Earth

Notes
-----
- The exported raster values are path gain in dB by default, not RSSI/RSRP.
  The repo stores path gain; TX power would be needed to derive received power.
- The rasterization method is centroid-based aggregation. It is fast and works
  well for visualization/export, but it is not an exact face-polygon burn-in.
- For TAK, the GeoTIFF is usually the better archival/interchange product.
  KMZ is convenient for Google Earth and quick overlays.

Example
-------
python rm_export_geotiff_kmz.py \
  --outputs data/remote/outputs/peru_rm \
  --scenes data/local/scenes/Peru_Cell_Coverage \
  --bboxes data/remote/outputs/peru.npz \
  --out-dir export/peru \
  --resolution-m 30 \
  --make-kmz
"""

from __future__ import annotations

import argparse
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from pyproj import Transformer

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.enums import Resampling
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires rasterio. Install with: pip install rasterio"
    ) from exc

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires Pillow. Install with: pip install Pillow"
    ) from exc

try:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires matplotlib. Install with: pip install matplotlib"
    ) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", required=True, help="Directory containing rm_*.npz files")
    p.add_argument("--scenes", required=True, help="Directory containing per-tile scene folders")
    p.add_argument(
        "--bboxes",
        required=True,
        help="Path to the bboxes .npz used for the simulation (contains 'corners')",
    )
    p.add_argument("--out-dir", required=True, help="Output directory for exported products")
    p.add_argument(
        "--resolution-m",
        type=float,
        default=30.0,
        help="Approximate output raster resolution in meters (default: 30)",
    )
    p.add_argument(
        "--aggregation",
        choices=["max", "mean"],
        default="max",
        help="How to aggregate multiple face samples into one raster cell (default: max)",
    )
    p.add_argument(
        "--metric",
        choices=["db", "linear"],
        default="db",
        help="Export path gain in dB (default) or linear scale",
    )
    p.add_argument(
        "--floor-linear",
        type=float,
        default=1e-15,
        help="Floor for linear path gain before dB conversion (default: 1e-15)",
    )
    p.add_argument(
        "--name",
        default="sionna_rm_export",
        help="Base name for exported files (default: sionna_rm_export)",
    )
    p.add_argument(
        "--make-kmz",
        action="store_true",
        help="Also create a KMZ GroundOverlay for Google Earth",
    )
    p.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap for KMZ PNG rendering (default: turbo)",
    )
    p.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum display value for KMZ rendering. Defaults to 2nd percentile.",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum display value for KMZ rendering. Defaults to 98th percentile.",
    )
    p.add_argument(
        "--region",
        type=float,
        nargs=4,
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        default=None,
        help="Optional lat/lon bbox to restrict export",
    )
    p.add_argument(
        "--tile-regex",
        default=r"rm_(\d{8})\.npz$",
        help="Regex used to parse tile ids from rm files",
    )
    return p.parse_args()


def utm_epsg_from_latlon(lat: float, lon: float) -> str:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    if lat >= 0:
        return f"EPSG:{32600 + zone}"
    return f"EPSG:{32700 + zone}"


def load_mesh_vertices_faces(mesh_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load mesh vertices/faces using open3d if available, otherwise trimesh."""
    try:
        import open3d as o3d  # type: ignore

        mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            raise ValueError(f"Empty mesh: {mesh_path}")
        return vertices, faces
    except Exception:
        try:
            import trimesh  # type: ignore

            mesh = trimesh.load(mesh_path, process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            if vertices.size == 0 or faces.size == 0:
                raise ValueError(f"Empty mesh: {mesh_path}")
            return vertices, faces
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Need either open3d or trimesh to read ground.ply. "
                "Install one of them with: pip install open3d  OR  pip install trimesh"
            ) from exc


def face_centroids_xy(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]  # (n_faces, 3, 3)
    centroids = tri.mean(axis=1)
    return centroids[:, :2]


def tile_bbox_intersects(tile_bbox: np.ndarray, region: tuple[float, float, float, float]) -> bool:
    south, west, north, east = region
    (tile_south, tile_west), (tile_north, tile_east) = tile_bbox
    return not (
        tile_north < south or tile_south > north or tile_east < west or tile_west > east
    )


def lonlat_resolution_deg(resolution_m: float, mid_lat: float) -> tuple[float, float]:
    lat_res = resolution_m / 110540.0
    cos_lat = max(math.cos(math.radians(mid_lat)), 1e-6)
    lon_res = resolution_m / (111320.0 * cos_lat)
    return lon_res, lat_res


def init_aggregator(height: int, width: int, mode: str) -> tuple[np.ndarray, np.ndarray | None]:
    if mode == "max":
        return np.full((height, width), -np.inf, dtype=np.float32), None
    if mode == "mean":
        return np.zeros((height, width), dtype=np.float64), np.zeros((height, width), dtype=np.uint32)
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def update_aggregator(
    raster: np.ndarray,
    counts: np.ndarray | None,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    mode: str,
) -> None:
    if mode == "max":
        flat = raster.ravel()
        idx = rows.astype(np.int64) * raster.shape[1] + cols.astype(np.int64)
        np.maximum.at(flat, idx, values.astype(np.float32))
        return

    if counts is None:
        raise ValueError("counts array is required for mean aggregation")
    np.add.at(raster, (rows, cols), values.astype(np.float64))
    np.add.at(counts, (rows, cols), 1)


def finalize_aggregator(raster: np.ndarray, counts: np.ndarray | None, mode: str) -> np.ndarray:
    if mode == "max":
        out = raster.copy()
        out[~np.isfinite(out)] = np.nan
        return out.astype(np.float32)
    if counts is None:
        raise ValueError("counts array is required for mean aggregation")
    out = np.full(raster.shape, np.nan, dtype=np.float32)
    valid = counts > 0
    out[valid] = (raster[valid] / counts[valid]).astype(np.float32)
    return out


def build_output_grid(
    selected_tile_ids: Iterable[int],
    corners: np.ndarray,
    resolution_m: float,
) -> tuple[tuple[float, float, float, float], float, float, int, int]:
    tile_boxes = corners[list(selected_tile_ids)]
    south = float(np.min(tile_boxes[:, 0, 0]))
    west = float(np.min(tile_boxes[:, 0, 1]))
    north = float(np.max(tile_boxes[:, 1, 0]))
    east = float(np.max(tile_boxes[:, 1, 1]))
    mid_lat = 0.5 * (south + north)
    lon_res, lat_res = lonlat_resolution_deg(resolution_m, mid_lat)
    width = int(math.ceil((east - west) / lon_res))
    height = int(math.ceil((north - south) / lat_res))
    return (south, west, north, east), lon_res, lat_res, width, height


def render_png(
    data: np.ndarray,
    out_png: str,
    cmap_name: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("No finite pixels available for KMZ rendering")

    vals = data[finite]
    if vmin is None:
        vmin = float(np.nanpercentile(vals, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(vals, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(cmap_name)

    rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
    colored = (cmap(norm(np.nan_to_num(data, nan=vmin))) * 255.0).astype(np.uint8)
    rgba[:, :, :] = colored
    rgba[~finite, 3] = 0  # transparent nodata

    Image.fromarray(rgba, mode="RGBA").save(out_png)
    return vmin, vmax


def write_kmz(
    kmz_path: str,
    png_path: str,
    name: str,
    bbox: tuple[float, float, float, float],
) -> None:
    south, west, north, east = bbox
    image_name = Path(png_path).name
    kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <GroundOverlay>
      <name>{name}</name>
      <Icon>
        <href>{image_name}</href>
      </Icon>
      <LatLonBox>
        <north>{north}</north>
        <south>{south}</south>
        <east>{east}</east>
        <west>{west}</west>
        <rotation>0</rotation>
      </LatLonBox>
    </GroundOverlay>
  </Document>
</kml>
'''
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml)
        zf.write(png_path, arcname=image_name)


TILE_RE_CACHE: dict[str, re.Pattern[str]] = {}


def tile_id_from_filename(path: Path, pattern: str) -> int:
    rx = TILE_RE_CACHE.setdefault(pattern, re.compile(pattern))
    match = rx.search(path.name)
    if not match:
        raise ValueError(f"Could not parse tile id from filename: {path}")
    return int(match.group(1))


def main() -> None:
    args = parse_args()

    outputs_dir = Path(args.outputs)
    scenes_dir = Path(args.scenes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corners = np.load(args.bboxes)["corners"]
    rm_files = sorted(outputs_dir.glob("rm_*.npz"))
    if not rm_files:
        raise SystemExit(f"No rm_*.npz files found in {outputs_dir}")

    selected: list[tuple[int, Path]] = []
    for rm_path in rm_files:
        tile_id = tile_id_from_filename(rm_path, args.tile_regex)
        if tile_id < 0 or tile_id >= len(corners):
            print(f"[!] Skipping {rm_path.name}: tile id {tile_id} not present in bboxes")
            continue
        if args.region is not None and not tile_bbox_intersects(corners[tile_id], tuple(args.region)):
            continue
        selected.append((tile_id, rm_path))

    if not selected:
        raise SystemExit("No tiles selected for export")

    bbox, lon_res, lat_res, width, height = build_output_grid(
        [tile_id for tile_id, _ in selected], corners, args.resolution_m
    )
    south, west, north, east = bbox
    print(
        f"[i] Output bbox: south={south:.6f}, west={west:.6f}, north={north:.6f}, east={east:.6f}"
    )
    print(
        f"[i] Output grid: {width} x {height} pixels at ~{args.resolution_m:.2f} m"
    )

    raster_accum, counts = init_aggregator(height, width, args.aggregation)

    processed_tiles = 0
    skipped_tiles = 0
    for tile_id, rm_path in selected:
        tile_dir = scenes_dir / f"{tile_id:08d}"
        ground_path = tile_dir / "mesh" / "ground.ply"
        if not ground_path.exists():
            print(f"[!] Missing ground mesh for tile {tile_id:08d}: {ground_path}")
            skipped_tiles += 1
            continue

        try:
            rm_npz = np.load(rm_path)
            rm_values = np.asarray(rm_npz["rm"]).reshape(-1)

            if args.metric == "db":
                rm_values = 10.0 * np.log10(np.maximum(rm_values.astype(np.float64), args.floor_linear))
            else:
                rm_values = rm_values.astype(np.float64)

            vertices, faces = load_mesh_vertices_faces(str(ground_path))
            centroids_xy = face_centroids_xy(vertices, faces)

            if len(rm_values) != len(centroids_xy):
                print(
                    f"[!] Tile {tile_id:08d} face/value mismatch: rm has {len(rm_values)} entries, "
                    f"mesh has {len(centroids_xy)} faces. Skipping."
                )
                skipped_tiles += 1
                continue

            (tile_south, tile_west), _ = corners[tile_id]
            tile_epsg = utm_epsg_from_latlon(float(tile_south), float(tile_west))
            to_utm = Transformer.from_crs("EPSG:4326", tile_epsg, always_xy=True)
            to_ll = Transformer.from_crs(tile_epsg, "EPSG:4326", always_xy=True)
            sw_x, sw_y = to_utm.transform(float(tile_west), float(tile_south))

            lon, lat = to_ll.transform(sw_x + centroids_xy[:, 0], sw_y + centroids_xy[:, 1])
            lon = np.asarray(lon)
            lat = np.asarray(lat)

            valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(rm_values)
            if not np.any(valid):
                print(f"[!] Tile {tile_id:08d} contains no finite samples. Skipping.")
                skipped_tiles += 1
                continue

            cols = np.floor((lon[valid] - west) / lon_res).astype(np.int64)
            rows = np.floor((north - lat[valid]) / lat_res).astype(np.int64)
            inside = (
                (rows >= 0)
                & (rows < height)
                & (cols >= 0)
                & (cols < width)
            )
            if not np.any(inside):
                print(f"[!] Tile {tile_id:08d} contributes no samples inside output grid. Skipping.")
                skipped_tiles += 1
                continue

            update_aggregator(
                raster_accum,
                counts,
                rows[inside],
                cols[inside],
                rm_values[valid][inside],
                args.aggregation,
            )
            processed_tiles += 1
            if processed_tiles % 10 == 0:
                print(f"[i] Processed {processed_tiles} tiles...")
        except Exception as exc:
            print(f"[!] Error on tile {tile_id:08d}: {exc}")
            skipped_tiles += 1

    final_raster = finalize_aggregator(raster_accum, counts, args.aggregation)
    finite = np.isfinite(final_raster)
    if not np.any(finite):
        raise SystemExit("Export failed: no finite raster cells were produced")

    tif_path = out_dir / f"{args.name}.tif"
    transform = from_origin(west, north, lon_res, lat_res)

    profile = {
        "driver": "GTiff",
        "height": final_raster.shape[0],
        "width": final_raster.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
        "tiled": True,
        "blockxsize": min(512, final_raster.shape[1]),
        "blockysize": min(512, final_raster.shape[0]),
        "compress": "deflate",
        "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(final_raster, 1)
        overview_levels = [2, 4, 8, 16, 32]
        overview_levels = [
            lvl
            for lvl in overview_levels
            if final_raster.shape[0] // lvl >= 64 and final_raster.shape[1] // lvl >= 64
        ]
        if overview_levels:
            dst.build_overviews(overview_levels, Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")
        dst.update_tags(
            metric=("path_gain_db" if args.metric == "db" else "path_gain_linear"),
            aggregation=args.aggregation,
            source="NVlabs/sionna-large-radio-maps export",
            note="Values aggregated from triangle centroids on ground measurement mesh",
        )

    print(f"[+] Wrote tiled GeoTIFF: {tif_path}")
    print(f"[i] Tiles processed: {processed_tiles}; skipped: {skipped_tiles}")
    print(
        f"[i] Raster stats: min={np.nanmin(final_raster):.3f}, "
        f"max={np.nanmax(final_raster):.3f}, mean={np.nanmean(final_raster):.3f}"
    )

    if args.make_kmz:
        png_path = out_dir / f"{args.name}.png"
        kmz_path = out_dir / f"{args.name}.kmz"
        vmin, vmax = render_png(final_raster, str(png_path), args.cmap, args.vmin, args.vmax)
        write_kmz(str(kmz_path), str(png_path), args.name, bbox)
        print(f"[+] Wrote KMZ: {kmz_path}")
        print(f"[i] KMZ display range: vmin={vmin:.3f}, vmax={vmax:.3f}")


if __name__ == "__main__":
    main()
