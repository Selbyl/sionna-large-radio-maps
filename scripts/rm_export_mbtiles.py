#!/usr/bin/env python3
"""
Export Sionna Large Radio Maps per-tile NPZ outputs to per-tile GeoTIFF rasters
and a merged raster MBTiles package suitable for Google Earth / ArcGIS / TAK workflows.

Pipeline
--------
1. Read rm_XXXXXXXX.npz outputs from compute_radio_maps.py
2. Read matching ground meshes from <scenes>/<tile_id>/mesh/ground.ply
3. Rasterize each tile independently into an RGBA GeoTIFF in EPSG:3857
4. Build a VRT mosaic from the per-tile rasters
5. Convert the VRT mosaic to MBTiles using GDAL
6. Add lower-zoom overviews with gdaladdo

Notes
-----
- This script writes one raster tile GeoTIFF per NPZ file before building MBTiles.
- MBTiles output is imagery-style RGBA tiles, not a raw single-band scientific raster.
- The final MBTiles zoom pyramid is derived from the requested ground resolution.

Example
-------
python rm_export_mbtiles.py \
  --outputs data/local/outputs/Peru_Cell_Coverage_RadioMap \
  --scenes data/local/scenes/Peru_Cell_Coverage \
  --bboxes data/remote/outputs/peru.npz \
  --work-dir export/peru_mbtiles_work \
  --out-mbtiles export/peru_coverage.mbtiles \
  --resolution-m 30 \
  --aggregation max \
  --workers 12
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from pyproj import Transformer

try:
    import rasterio
    from rasterio.transform import from_origin
except Exception as exc:  # pragma: no cover
    raise SystemExit("Install rasterio: pip install rasterio") from exc

try:
    import matplotlib
    import matplotlib.colors as mcolors
except Exception as exc:  # pragma: no cover
    raise SystemExit("Install matplotlib: pip install matplotlib") from exc

_INITIAL_WEBMERCATOR_RES = 156543.03392804097
_TILE_RE = re.compile(r"rm_(\d+)\.npz$")


@dataclass(frozen=True)
class WorkerConfig:
    scenes_dir: str
    tiles_dir: str
    resolution_m: float
    aggregation: str
    metric: str
    floor_linear: float
    cmap_name: str
    vmin: float
    vmax: float
    opacity: float
    bboxes_path: str


@dataclass(frozen=True)
class TileTask:
    tile_id: int
    rm_path: str


@dataclass(frozen=True)
class TileResult:
    tile_id: int
    ok: bool
    tif_path: str | None
    message: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", required=True, help="Directory containing rm_*.npz files")
    p.add_argument("--scenes", required=True, help="Directory containing per-tile scene folders")
    p.add_argument("--bboxes", required=True, help="Path to bboxes .npz with 'corners'")
    p.add_argument("--work-dir", required=True, help="Working directory for per-tile rasters and intermediate files")
    p.add_argument("--out-mbtiles", required=True, help="Output MBTiles path")
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
        help="Aggregate multiple samples per pixel with max or mean (default: max)",
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
    p.add_argument("--cmap", default="turbo", help="Matplotlib colormap name (default: turbo)")
    p.add_argument("--vmin", type=float, default=None, help="Display minimum. Defaults to sampled 2nd percentile.")
    p.add_argument("--vmax", type=float, default=None, help="Display maximum. Defaults to sampled 98th percentile.")
    p.add_argument(
        "--opacity",
        type=float,
        default=0.75,
        help="Overlay opacity from 0..1 (default: 0.75)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(12, (os.cpu_count() or 4) // 2)),
        help="Worker processes for per-tile rasterization (default: conservative auto)",
    )
    p.add_argument(
        "--min-zoom",
        type=int,
        default=0,
        help="Lowest zoom level to preserve in MBTiles overviews (default: 0)",
    )
    p.add_argument(
        "--overview-resampling",
        choices=["nearest", "average"],
        default="nearest",
        help="Resampling for lower zoom overviews (default: nearest)",
    )
    p.add_argument(
        "--sample-cap",
        type=int,
        default=10000,
        help="Max sampled values per tile for auto vmin/vmax (default: 10000)",
    )
    p.add_argument(
        "--tile-format",
        choices=["PNG", "WEBP", "JPEG"],
        default="PNG",
        help="MBTiles tile format (default: PNG). PNG is safest for transparency.",
    )
    p.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep intermediate GeoTIFFs, VRT, and lists after MBTiles is built",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Optional MBTiles name metadata. Defaults to output stem.",
    )
    p.add_argument(
        "--description",
        default="Sionna coverage overlay",
        help="Optional MBTiles description metadata.",
    )
    return p.parse_args()


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"Required command not found: {name}. Install GDAL command line utilities first."
        )


def tile_id_from_path(path: Path) -> int:
    m = _TILE_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse tile id from filename: {path}")
    return int(m.group(1))


def load_mesh_vertices_faces(mesh_path: str) -> tuple[np.ndarray, np.ndarray]:
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
        except Exception as exc:
            raise RuntimeError(
                "Need either open3d or trimesh to read ground.ply. "
                "Install one with: pip install open3d OR pip install trimesh"
            ) from exc


def utm_epsg_from_latlon(lat: float, lon: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    zone = max(1, min(zone, 60))
    return 32600 + zone if lat >= 0 else 32700 + zone


def face_centroids_xy(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return vertices[faces].mean(axis=1)[:, :2]


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


def convert_metric(values: np.ndarray, metric: str, floor_linear: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if metric == "db":
        return 10.0 * np.log10(np.maximum(values, floor_linear))
    return values


def estimate_display_range(
    rm_files: Iterable[Path], metric: str, floor_linear: float, sample_cap: int
) -> tuple[float, float]:
    sample_parts: list[np.ndarray] = []
    rng = np.random.default_rng(12345)
    for rm_path in rm_files:
        try:
            data = np.load(rm_path)
            vals = convert_metric(np.asarray(data["rm"]).reshape(-1), metric, floor_linear)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if vals.size > sample_cap:
                idx = rng.choice(vals.size, size=sample_cap, replace=False)
                vals = vals[idx]
            sample_parts.append(vals.astype(np.float32, copy=False))
        except Exception:
            continue
    if not sample_parts:
        raise SystemExit("Could not derive display range: no finite samples found")
    sample = np.concatenate(sample_parts)
    vmin = float(np.nanpercentile(sample, 2))
    vmax = float(np.nanpercentile(sample, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(sample))
        vmax = float(np.nanmax(sample))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def colorize_to_rgba(
    data: np.ndarray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    opacity: float,
) -> np.ndarray:
    finite = np.isfinite(data)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
    colored = (cmap(norm(np.nan_to_num(data, nan=vmin))) * 255.0).astype(np.uint8)
    rgba[:, :, :3] = colored[:, :, :3]
    rgba[:, :, 3] = int(np.clip(opacity, 0.0, 1.0) * 255)
    rgba[~finite, 3] = 0
    return rgba


def render_tile(task: TileTask, cfg: WorkerConfig) -> TileResult:
    bboxes = np.load(cfg.bboxes_path)["corners"]
    tile_id = task.tile_id
    rm_path = Path(task.rm_path)
    tile_dir = Path(cfg.scenes_dir) / f"{tile_id:08d}"
    ground_path = tile_dir / "mesh" / "ground.ply"
    out_tif = Path(cfg.tiles_dir) / f"tile_{tile_id:08d}.tif"

    if not ground_path.exists():
        return TileResult(tile_id, False, None, f"Missing ground mesh: {ground_path}")

    try:
        rm_npz = np.load(rm_path)
        rm_values = convert_metric(np.asarray(rm_npz["rm"]).reshape(-1), cfg.metric, cfg.floor_linear)

        vertices, faces = load_mesh_vertices_faces(str(ground_path))
        centroids_xy = face_centroids_xy(vertices, faces)
        if len(rm_values) != len(centroids_xy):
            return TileResult(
                tile_id,
                False,
                None,
                f"Face/value mismatch: rm has {len(rm_values)} entries, mesh has {len(centroids_xy)} faces",
            )

        (tile_south, tile_west), _ = bboxes[tile_id]
        tile_epsg = utm_epsg_from_latlon(float(tile_south), float(tile_west))
        to_utm = Transformer.from_crs("EPSG:4326", tile_epsg, always_xy=True)
        to_ll = Transformer.from_crs(tile_epsg, "EPSG:4326", always_xy=True)
        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        sw_x, sw_y = to_utm.transform(float(tile_west), float(tile_south))
        lon, lat = to_ll.transform(sw_x + centroids_xy[:, 0], sw_y + centroids_xy[:, 1])
        x, y = to_3857.transform(np.asarray(lon), np.asarray(lat))

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(rm_values)
        if not np.any(valid):
            return TileResult(tile_id, False, None, "No finite samples")

        x = np.asarray(x)[valid]
        y = np.asarray(y)[valid]
        vals = np.asarray(rm_values)[valid]

        minx = float(np.min(x))
        maxx = float(np.max(x))
        miny = float(np.min(y))
        maxy = float(np.max(y))
        if not np.isfinite(minx + maxx + miny + maxy):
            return TileResult(tile_id, False, None, "Non-finite tile bounds")

        width = max(1, int(math.ceil((maxx - minx) / cfg.resolution_m)) + 1)
        height = max(1, int(math.ceil((maxy - miny) / cfg.resolution_m)) + 1)

        cols = np.floor((x - minx) / cfg.resolution_m).astype(np.int64)
        rows = np.floor((maxy - y) / cfg.resolution_m).astype(np.int64)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        if not np.any(inside):
            return TileResult(tile_id, False, None, "No samples landed inside output grid")

        raster, counts = init_aggregator(height, width, cfg.aggregation)
        update_aggregator(raster, counts, rows[inside], cols[inside], vals[inside], cfg.aggregation)
        final = finalize_aggregator(raster, counts, cfg.aggregation)
        if not np.any(np.isfinite(final)):
            return TileResult(tile_id, False, None, "Rasterized tile contains no finite pixels")

        rgba = colorize_to_rgba(final, cfg.cmap_name, cfg.vmin, cfg.vmax, cfg.opacity)
        transform = from_origin(minx, maxy, cfg.resolution_m, cfg.resolution_m)

        with rasterio.open(
            out_tif,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=4,
            dtype="uint8",
            crs="EPSG:3857",
            transform=transform,
            tiled=True,
            compress="DEFLATE",
            interleave="pixel",
            photometric="RGB",
        ) as dst:
            for i in range(4):
                dst.write(rgba[:, :, i], i + 1)
            dst.set_band_description(1, "red")
            dst.set_band_description(2, "green")
            dst.set_band_description(3, "blue")
            dst.set_band_description(4, "alpha")
            dst.colorinterp = (
                rasterio.enums.ColorInterp.red,
                rasterio.enums.ColorInterp.green,
                rasterio.enums.ColorInterp.blue,
                rasterio.enums.ColorInterp.alpha,
            )

        return TileResult(tile_id, True, str(out_tif), "")
    except Exception as exc:
        return TileResult(tile_id, False, None, str(exc))


def run_checked(cmd: list[str]) -> None:
    print("[i]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def estimate_max_zoom(resolution_m: float) -> int:
    z = int(round(math.log2(_INITIAL_WEBMERCATOR_RES / resolution_m)))
    return max(0, min(22, z))


def overview_factors(max_zoom: int, min_zoom: int) -> list[int]:
    if min_zoom < 0:
        min_zoom = 0
    if max_zoom <= min_zoom:
        return []
    levels = max_zoom - min_zoom
    return [2**i for i in range(1, levels + 1)]


def main() -> None:
    args = parse_args()

    for cmd in ("gdalbuildvrt", "gdal_translate", "gdaladdo"):
        require_command(cmd)

    outputs_dir = Path(args.outputs)
    scenes_dir = Path(args.scenes)
    work_dir = Path(args.work_dir)
    tiles_dir = work_dir / "tile_tifs"
    vrt_path = work_dir / "mosaic.vrt"
    input_list_path = work_dir / "tile_list.txt"
    out_mbtiles = Path(args.out_mbtiles)
    work_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    out_mbtiles.parent.mkdir(parents=True, exist_ok=True)

    rm_files = sorted(outputs_dir.glob("rm_*.npz"))
    if not rm_files:
        raise SystemExit(f"No rm_*.npz files found in {outputs_dir}")

    tile_tasks: list[TileTask] = []
    corners = np.load(args.bboxes)["corners"]
    for rm_path in rm_files:
        tile_id = tile_id_from_path(rm_path)
        if tile_id < 0 or tile_id >= len(corners):
            print(f"[!] Skipping {rm_path.name}: tile id {tile_id} not present in bboxes")
            continue
        tile_tasks.append(TileTask(tile_id=tile_id, rm_path=str(rm_path)))

    if not tile_tasks:
        raise SystemExit("No valid tile tasks found")

    if args.vmin is None or args.vmax is None:
        auto_vmin, auto_vmax = estimate_display_range(
            (Path(t.rm_path) for t in tile_tasks), args.metric, args.floor_linear, args.sample_cap
        )
        vmin = auto_vmin if args.vmin is None else args.vmin
        vmax = auto_vmax if args.vmax is None else args.vmax
    else:
        vmin = args.vmin
        vmax = args.vmax

    print(f"[i] Using display range vmin={vmin:.3f}, vmax={vmax:.3f}")

    cfg = WorkerConfig(
        scenes_dir=str(scenes_dir),
        tiles_dir=str(tiles_dir),
        resolution_m=float(args.resolution_m),
        aggregation=args.aggregation,
        metric=args.metric,
        floor_linear=float(args.floor_linear),
        cmap_name=args.cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        opacity=float(args.opacity),
        bboxes_path=str(args.bboxes),
    )

    rendered_tifs: list[str] = []
    failures = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(render_tile, task, cfg): task.tile_id for task in tile_tasks}
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result.ok and result.tif_path:
                rendered_tifs.append(result.tif_path)
            else:
                failures += 1
                print(f"[!] Tile {result.tile_id:08d} skipped: {result.message}")
            if done % 25 == 0 or done == len(tile_tasks):
                print(f"[i] Rasterized {done}/{len(tile_tasks)} tiles")

    if not rendered_tifs:
        raise SystemExit("No per-tile GeoTIFFs were generated")

    rendered_tifs = sorted(rendered_tifs)
    input_list_path.write_text("\n".join(rendered_tifs) + "\n", encoding="utf-8")

    run_checked([
        "gdalbuildvrt",
        "-overwrite",
        "-input_file_list",
        str(input_list_path),
        str(vrt_path),
    ])

    if out_mbtiles.exists():
        out_mbtiles.unlink()

    name = args.name or out_mbtiles.stem
    tile_format = args.tile_format.upper()

    translate_cmd = [
        "gdal_translate",
        str(vrt_path),
        str(out_mbtiles),
        "-of",
        "MBTILES",
        "-co",
        f"TILE_FORMAT={tile_format}",
        "-mo",
        f"name={name}",
        "-mo",
        f"description={args.description}",
        "-mo",
        "type=overlay",
        "-mo",
        "version=1.3",
    ]
    run_checked(translate_cmd)

    max_zoom = estimate_max_zoom(args.resolution_m)
    factors = overview_factors(max_zoom, args.min_zoom)
    if factors:
        run_checked([
            "gdaladdo",
            "-r",
            args.overview_resampling,
            str(out_mbtiles),
            *[str(f) for f in factors],
        ])
    else:
        print("[i] No overviews requested or needed")

    print(f"[+] Wrote MBTiles: {out_mbtiles}")
    print(f"[i] Per-tile GeoTIFFs written: {len(rendered_tifs)}")
    print(f"[i] Tiles skipped: {failures}")
    print(f"[i] Estimated max zoom from {args.resolution_m:.2f} m pixels: {max_zoom}")
    if factors:
        print(f"[i] Overview factors added: {', '.join(map(str, factors))}")

    if not args.keep_work:
        try:
            for tif_path in rendered_tifs:
                Path(tif_path).unlink(missing_ok=True)
            vrt_path.unlink(missing_ok=True)
            input_list_path.unlink(missing_ok=True)
            if tiles_dir.exists():
                try:
                    tiles_dir.rmdir()
                except OSError:
                    pass
        except Exception:
            print("[!] Could not fully clean work directory; intermediate files were left in place")


if __name__ == "__main__":
    main()
