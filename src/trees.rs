// Copyright 2026 Yves Moisan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Trees source — read tree meshes from a GeoParquet file and emit them as
//! `OwnedCityModel` instances ready to be merged into tyler's export pipeline.
//!
//! Each parquet row is expected to carry one tree:
//!   - a geometry column with PolygonZ / MultiPolygonZ WKB bytes (the tree mesh)
//!   - optional `id` column
//!   - optional attribute columns
//!
//! We convert each row's WKB → quantized vertex coordinates (using the same
//! transform as the buildings dataset, so vertex tables align), build a small
//! CityJSONFeature JSON document with a SolitaryVegetationObject CityObject,
//! and parse it via `cityjson_json::v2_0::read_feature` to land in the same
//! `OwnedCityModel` shape that v0.4.1 uses for all features.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use arrow::array::{Array, BinaryArray, LargeBinaryArray, StringArray};
use arrow::datatypes::DataType;
use cityjson_lib::CityModel as OwnedCityModel;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{Map, Value, json};

/// CityJSON `transform` block — quantization parameters shared with the
/// buildings dataset. Trees must use the same one so tile assembly works.
#[derive(Debug, Clone, Copy)]
pub struct CityJsonTransform {
    pub scale: [f64; 3],
    pub translate: [f64; 3],
}

/// Read every tree in `parquet_path` and return them as parsed CityModel feature
/// instances. `transform` must match the buildings dataset's CityJSON transform
/// so the quantized vertices align across feature sources.
pub fn load_trees_as_models(
    parquet_path: &Path,
    transform: &CityJsonTransform,
    city_object_type: &str,
    id_column: Option<&str>,
) -> Result<Vec<OwnedCityModel>> {
    let file = File::open(parquet_path)
        .with_context(|| format!("opening parquet file {parquet_path:?}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("building parquet reader")?;

    let geom_col = primary_geometry_column(builder.metadata().file_metadata().key_value_metadata())
        .unwrap_or_else(|| "geometry".to_string());
    let schema = builder.schema().clone();
    let geom_idx = schema
        .index_of(&geom_col)
        .with_context(|| format!("geometry column '{geom_col}' not in parquet schema"))?;
    let id_idx = id_column.and_then(|name| schema.index_of(name).ok());

    let reader = builder.build().context("opening parquet reader")?;
    let read_opts = cityjson_json::ReadOptions::default();
    let mut out: Vec<OwnedCityModel> = Vec::new();
    let mut auto_id: u64 = 0;

    for batch_result in reader {
        let batch = batch_result.context("reading parquet batch")?;
        let geom_col = batch.column(geom_idx);
        let id_col_opt = id_idx.map(|i| batch.column(i));

        for row in 0..batch.num_rows() {
            let wkb = extract_wkb(geom_col.as_ref(), row)?;
            let polygons = parse_wkb_3d(wkb)?;

            let id = if let Some(arr) = id_col_opt {
                extract_string_id(arr.as_ref(), row).unwrap_or_else(|| {
                    auto_id += 1;
                    format!("tree-{auto_id}")
                })
            } else {
                auto_id += 1;
                format!("tree-{auto_id}")
            };

            let feature_bytes =
                build_cityjson_feature_bytes(&id, &polygons, transform, city_object_type)?;
            let model = cityjson_json::v2_0::read_feature(&feature_bytes, &read_opts)
                .with_context(|| format!("parsing synthetic CityJSONFeature for tree {id}"))?;
            out.push(model);
        }
    }

    Ok(out)
}

fn build_cityjson_feature_bytes(
    id: &str,
    polygons: &[Polygon3D],
    transform: &CityJsonTransform,
    city_object_type: &str,
) -> Result<Vec<u8>> {
    let mut vertices_q: Vec<[i64; 3]> = Vec::new();
    let mut boundaries: Vec<Vec<Vec<u32>>> = Vec::new();

    for polygon in polygons {
        for ring in polygon {
            let mut ring_indices: Vec<u32> = Vec::with_capacity(ring.len());
            for v in ring {
                let q = quantize(v, transform);
                let idx = vertices_q.len() as u32;
                vertices_q.push(q);
                ring_indices.push(idx);
            }
            // CityJSON MultiSurface boundary: each ring is one surface [[ring]]
            boundaries.push(vec![ring_indices]);
        }
    }

    let mut cityobjects = Map::new();
    cityobjects.insert(
        id.to_string(),
        json!({
            "type": city_object_type,
            "geometry": [{
                "type": "MultiSurface",
                "lod": "2",
                "boundaries": boundaries,
            }],
        }),
    );

    let feature = json!({
        "type": "CityJSONFeature",
        "id": id,
        "CityObjects": cityobjects,
        "vertices": vertices_q,
    });

    Ok(serde_json::to_vec(&feature)?)
}

fn quantize(v: &[f64; 3], t: &CityJsonTransform) -> [i64; 3] {
    [
        ((v[0] - t.translate[0]) / t.scale[0]).round() as i64,
        ((v[1] - t.translate[1]) / t.scale[1]).round() as i64,
        ((v[2] - t.translate[2]) / t.scale[2]).round() as i64,
    ]
}

fn primary_geometry_column(
    kv: Option<&Vec<parquet::format::KeyValue>>,
) -> Option<String> {
    let geo_json = kv?.iter().find(|kv| kv.key == "geo").and_then(|kv| kv.value.clone())?;
    let geo: Value = serde_json::from_str(&geo_json).ok()?;
    Some(
        geo.get("primary_column")
            .and_then(|v| v.as_str())
            .unwrap_or("geometry")
            .to_string(),
    )
}

fn extract_wkb(arr: &dyn Array, row: usize) -> Result<&[u8]> {
    match arr.data_type() {
        DataType::Binary => Ok(arr
            .as_any()
            .downcast_ref::<BinaryArray>()
            .context("downcast BinaryArray")?
            .value(row)),
        DataType::LargeBinary => Ok(arr
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .context("downcast LargeBinaryArray")?
            .value(row)),
        other => bail!("geometry column has unsupported arrow type {other:?}"),
    }
}

fn extract_string_id(arr: &dyn Array, row: usize) -> Option<String> {
    arr.as_any()
        .downcast_ref::<StringArray>()
        .map(|s| s.value(row).to_string())
}

// --- WKB parsing (PolygonZ / MultiPolygonZ) ---

type Ring3D = Vec<[f64; 3]>;
type Polygon3D = Vec<Ring3D>;

fn parse_wkb_3d(wkb: &[u8]) -> Result<Vec<Polygon3D>> {
    if wkb.len() < 5 {
        bail!("WKB too short: {} bytes", wkb.len());
    }
    let le = wkb[0] == 1;
    let geom_type = read_u32(wkb, 1, le);
    let base_type = geom_type & 0x0000FFFF;
    let has_z = (geom_type & 0x8000_0000) != 0 || base_type >= 1000;
    let iso_type = if base_type >= 1000 {
        base_type - 1000
    } else {
        base_type
    };

    match iso_type {
        3 => {
            let (polygon, _) = parse_polygon(wkb, 5, le, has_z)?;
            Ok(vec![polygon])
        }
        6 => {
            let num_polygons = read_u32(wkb, 5, le) as usize;
            let mut offset = 9;
            let mut polygons = Vec::with_capacity(num_polygons);
            for _ in 0..num_polygons {
                if offset + 5 > wkb.len() {
                    bail!("WKB truncated in MultiPolygon");
                }
                let sub_le = wkb[offset] == 1;
                offset += 5;
                let (polygon, new_offset) = parse_polygon(wkb, offset, sub_le, has_z)?;
                polygons.push(polygon);
                offset = new_offset;
            }
            Ok(polygons)
        }
        _ => bail!("Unsupported WKB geometry type: {geom_type} (iso: {iso_type})"),
    }
}

fn parse_polygon(wkb: &[u8], start: usize, le: bool, has_z: bool) -> Result<(Polygon3D, usize)> {
    let num_rings = read_u32(wkb, start, le) as usize;
    let mut offset = start + 4;
    let mut rings = Vec::with_capacity(num_rings);
    let coord_size = if has_z { 3 } else { 2 };
    let bytes_per_coord = coord_size * 8;

    for _ in 0..num_rings {
        if offset + 4 > wkb.len() {
            bail!("WKB truncated reading ring point count");
        }
        let n = read_u32(wkb, offset, le) as usize;
        offset += 4;
        let mut ring = Vec::with_capacity(n);
        for _ in 0..n {
            if offset + bytes_per_coord > wkb.len() {
                bail!("WKB truncated reading ring vertex");
            }
            let x = read_f64(wkb, offset, le);
            let y = read_f64(wkb, offset + 8, le);
            let z = if has_z {
                read_f64(wkb, offset + 16, le)
            } else {
                0.0
            };
            ring.push([x, y, z]);
            offset += bytes_per_coord;
        }
        rings.push(ring);
    }
    Ok((rings, offset))
}

fn read_u32(buf: &[u8], at: usize, le: bool) -> u32 {
    let b: [u8; 4] = buf[at..at + 4].try_into().unwrap();
    if le {
        u32::from_le_bytes(b)
    } else {
        u32::from_be_bytes(b)
    }
}

fn read_f64(buf: &[u8], at: usize, le: bool) -> f64 {
    let b: [u8; 8] = buf[at..at + 8].try_into().unwrap();
    if le {
        f64::from_le_bytes(b)
    } else {
        f64::from_be_bytes(b)
    }
}
