// Copyright 2023 Balázs Dukai, Ravi Peters
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
mod cli;
mod cityjsonl_source;
mod copc_reader;
mod formats;
mod geoparquet_source;
mod gltf_writer;
mod las_source;
mod material;
mod parser;
mod proj;
mod spatial_structs;
mod transform_align;

#[cfg(feature = "geoflow")]
use core::time::Duration;
use std::env;
use std::fs;
use std::fs::File;
use std::io::Write;
#[cfg(feature = "geoflow")]
use std::io::BufWriter;
#[cfg(feature = "geoflow")]
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::formats::cesium3dtiles::{Tile, TileId, SplatLodConfig};
use clap::Parser;
use log::{debug, info, log_enabled, warn, Level};
use rayon::prelude::*;
#[cfg(feature = "geoflow")]
use subprocess::{Exec, Redirection};

#[cfg(feature = "geoflow")]
#[derive(Debug, Default, Clone)]
struct SubprocessConfig {
    output_extension: String,
    exe: PathBuf,
    script: PathBuf,
    timeout: Option<Duration>,
    verbose: bool,
}

#[cfg(feature = "geoflow")]
#[derive(Debug, Clone, clap::ValueEnum, Eq, PartialEq)]
#[clap(rename_all = "lower")]
pub enum Formats {
    _3DTiles,
    CityJSON,
}

#[cfg(feature = "geoflow")]
impl ToString for Formats {
    fn to_string(&self) -> String {
        match self {
            Formats::_3DTiles => "3DTiles".to_string(),
            Formats::CityJSON => "CityJSON".to_string(),
        }
    }
}

#[derive(Default, Debug)]
struct DebugData {
    world: Option<PathBuf>,
    quadtree: Option<PathBuf>,
    tiles_results: Option<PathBuf>,
}

#[cfg(feature = "geoflow")]
/// Write the list of feature paths for a tile into a text file, instead of passing
/// super long paths-string to the subprocess, because with very long arguments we can
/// get an 'Argument list too long' error.
// todo input: collect features from files and write them to a single newline-delimited file
fn write_inputs(
    world: &parser::World,
    path_features_input_dir: &Path,
    qtree_node: &spatial_structs::QuadTree,
    file_name: &str,
) -> PathBuf {
    let path_features_input_file = path_features_input_dir
        .join(file_name)
        .with_extension("input");
    fs::create_dir_all(path_features_input_file.parent().unwrap()).unwrap_or_else(|_| {
        panic!(
            "should be able to create the directory {:?}",
            path_features_input_file.parent().unwrap()
        )
    });
    let _fi_file = File::create(&path_features_input_file).unwrap_or_else(|_| {
        panic!(
            "should be able to create a file {:?}",
            &path_features_input_file
        )
    });
    let mut feature_input = BufWriter::new(_fi_file);
    for cellid in qtree_node.cells() {
        let cell = world.grid.cell(cellid);
        for fid in cell.feature_ids.iter() {
            let fp = world.features[*fid]
                .path_jsonl
                .clone()
                .into_os_string()
                .into_string()
                .unwrap();
            writeln!(feature_input, "{}", fp)
                .expect("should be able to write feature path to the input file");
        }
    }
    path_features_input_file
}

#[cfg(feature = "geoflow")]
fn run_subprocess(
    subprocess_config: &SubprocessConfig,
    tile: Tile,
    output_file: PathBuf,
    cmd: Exec,
) -> Option<Tile> {
    let cmd_string = cmd.to_cmdline_lossy();
    debug!("{cmd_string}");
    let redirection_stdout = Redirection::Pipe; // Redirection::Pipe | subprocess::NullFile
    let redirection_stderr = Redirection::Pipe; // Redirection::Merge
    let exec = cmd.stdout(redirection_stdout).stderr(redirection_stderr);
    let popen_res = exec.popen();
    match popen_res {
        Ok(mut popen) => {
            let (mut stdout_opt, mut stderr_opt): (Option<String>, Option<String>) = (None, None);
            let mut _exit_status = subprocess::ExitStatus::Undetermined;
            if let Some(timeout) = subprocess_config.timeout {
                let mut communicator = popen.communicate_start(None);
                if let Some(status) = popen.wait_timeout(timeout).unwrap() {
                    if let Ok(s) = communicator.read_string() {
                        (stdout_opt, stderr_opt) = s;
                    };
                    _exit_status = status;
                } else {
                    warn!(
                        "Tile {} timed out, conversion subprocess command:\n{}",
                        &tile.id, cmd_string
                    );
                    popen.kill().unwrap();
                    popen.wait().unwrap();
                    _exit_status = popen.exit_status().unwrap();
                }
            } else {
                (stdout_opt, stderr_opt) = popen.communicate(None).unwrap();
                _exit_status = popen.wait().unwrap();
            }

            // The stderr is Redirection::Merge-d into the stdout
            if !output_file.exists() {
                if subprocess_config.verbose {
                    warn!(
                        "Tile {} conversion failed, conversion subprocess command:\n{}\nsubprocess stdout:\n{}\nsubprocess stderr:\n{}",
                        tile.id, cmd_string, stdout_opt.unwrap_or_default(), stderr_opt.unwrap_or_default(),
                    );
                } else {
                    warn!(
                        "Tile {} conversion failed, conversion subprocess command:\n{}",
                        tile.id, cmd_string
                    );
                }
                return Some(tile);
            }
        }
        Err(popen_error) => {
            warn!("{}", popen_error);
            return Some(tile);
        }
    }
    // Progress tracking will be added in the calling code
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // --- Embed proj.db: extract from binary into a temp file and set PROJ_DATA
    static PROJ_DB_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/proj.db"));
    let proj_db_dir = tempfile::tempdir()?;
    let proj_db_path = proj_db_dir.path().join("proj.db");
    fs::write(&proj_db_path, PROJ_DB_BYTES)?;
    env::set_var("PROJ_DATA", proj_db_dir.path());
    debug!(
        "Embedded proj.db ({} bytes) extracted to {:?}",
        PROJ_DB_BYTES.len(),
        proj_db_dir.path()
    );

    // --- Begin argument parsing
    let cli = crate::cli::Cli::parse();
    debug!("{:?}", &cli);
    debug!("tyler version: {}", clap::crate_version!());
    if !cli.output.is_dir() {
        fs::create_dir_all(&cli.output)?;
        debug!("Created output directory {:#?}", &cli.output);
    }
    // Since we have a default value, we can safely unwrap.
    let grid_cellsize = cli.grid_cellsize.unwrap();
    let geometric_error_above_leaf = cli.geometric_error_above_leaf.unwrap();
    debug!("Using native glTF writer");
    
    // Build per-CityObjectType material config from TOML file and/or CLI args
    let material_config = cli.build_material_config()
        .map_err(|e| format!("Failed to build material config: {}", e))?;

    // Build attribute whitelist from --3dtiles-metadata-3dbag / --3dtiles-metadata-roofer flags
    let attr_whitelist = cli.attribute_whitelist();
    
    // Validate PROJ availability for coordinate transformations
    debug!("Validating PROJ library availability for coordinate transformations...");
    match crate::proj::Proj::new_known_crs("EPSG:4326", "EPSG:3857", None) {
        Ok(test_proj) => {
            match test_proj.convert((0.0, 0.0, 0.0)) {
                Ok(_) => {
                    debug!("PROJ library validated successfully");
                }
                Err(e) => {
                    return Err(format!(
                        "PROJ library data files not found or invalid. \
                         Coordinate transformations are required for 3D Tiles generation. \
                         PROJ transformation test failed: {}. \
                         Please ensure PROJ data files are installed and accessible.",
                        e
                    ).into());
                }
            }
        }
        Err(e) => {
            return Err(format!(
                "PROJ library data files not found or invalid. \
                 Coordinate transformations are required for 3D Tiles generation. \
                 Failed to create PROJ transformer: {}. \
                 Please ensure PROJ data files are installed and accessible.",
                e
            ).into());
        }
    }
    // Since we have a default value, it is safe to unwrap
    // let qtree_capacity = 0; // override cli.qtree_capacity
    let qtree_criteria = spatial_structs::QuadTreeCriteria::Vertices; // override --qtree-criteria
    let quadtree_capacity = match qtree_criteria {
        spatial_structs::QuadTreeCriteria::Objects => {
            spatial_structs::QuadTreeCapacity::Objects(cli.qtree_capacity.unwrap())
        }
        spatial_structs::QuadTreeCriteria::Vertices => {
            spatial_structs::QuadTreeCapacity::Vertices(cli.qtree_capacity.unwrap())
        }
    };
    if cli.cesium3dtiles_content_bv_from_tile && !cli.cesium3dtiles_content_add_bv {
        warn!("cesium3dtiles_content_bv_from_tile is true, but cesium3dtiles_content_add_bv is false. The tile content bounding volumes are not going to be added, unless you set --3dtiles-content-add-bv");
    }
    let debug_data = match cli.debug_load_data {
        None => DebugData::default(),
        Some(dir_path) => {
            if dir_path.is_dir() {
                let world_path = dir_path.join("world.bincode");
                let quadtree_path = dir_path.join("quadtree.bincode");
                let _tileset_path = dir_path.join("tileset.bincode");
                let tiles_results_path = dir_path.join("tiles_results.bincode");
                DebugData {
                    world: world_path.exists().then_some(world_path),
                    quadtree: quadtree_path.exists().then_some(quadtree_path),
                    tiles_results: tiles_results_path.exists().then_some(tiles_results_path),
                }
            } else {
                warn!(
                    "debug_load_data {dir_path:?} is not a directory, cannot load .bincode files"
                );
                DebugData::default()
            }
        }
    };
    debug!("{:?}", debug_data);
    let debug_data_output_path = cli.output.join("debug");
    if (cli.grid_export || log_enabled!(Level::Debug)) && !debug_data_output_path.exists() {
        fs::create_dir(&debug_data_output_path)?;
    }
    // --- end of argument parsing

    // --- Begin preprocessing pipeline (in-memory) ---
    let world: parser::World = match debug_data.world {
        None => {
            match (&cli.buildings, &cli.trees) {
                // Buildings (optionally with trees)
                (Some(buildings_path), trees_opt) => {
                    // OOM check
                    let file_size = fs::metadata(buildings_path)
                        .map(|m| m.len())
                        .unwrap_or(0);
                    let estimated = file_size * 3;
                    let available = parser::available_memory_bytes().unwrap_or(u64::MAX);
                    if estimated > available * 7 / 10 {
                        return Err(Box::from(format!(
                            "Estimated memory needed: {:.1} GB (input file {:.1} MB \u{00d7} 3), \
                             available: {:.1} GB. Allocate at least {:.0} GB RAM to this process.",
                            estimated as f64 / 1e9,
                            file_size as f64 / 1e6,
                            available as f64 / 1e9,
                            (estimated as f64 / 0.7 / 1e9).ceil()
                        )));
                    }

                    // Load buildings into memory
                    let (cityjsonl_meta, mut features) =
                        cityjsonl_source::load_cityjsonl_to_memory(buildings_path)?;
                    info!("Loaded {} building features into memory", features.len());

                    // Load trees (if any) into memory
                    if let Some(trees_path) = trees_opt {
                        let src_epsg = cityjsonl_meta
                            .reference_system
                            .split('/')
                            .last()
                            .and_then(|s| s.parse::<u16>().ok())
                            .unwrap_or_else(|| {
                                panic!(
                                    "Could not extract EPSG code from reference system: {}",
                                    cityjsonl_meta.reference_system
                                )
                            });

                        let tree_features = geoparquet_source::load_geoparquet_to_memory(
                            trees_path,
                            cityjsonl_meta.transform.clone(),
                            src_epsg,
                            "SolitaryVegetationObject",
                            cli.tree_id_column.as_deref(),
                        )?;
                        info!("Loaded {} tree features into memory", tree_features.len());
                        features.extend(tree_features);
                    }

                    // Filter features to LAS bounding box (if --las-rgb provided)
                    // and inject LAS z-bounds into grid when footprints are 2D.
                    let las_z_bounds = if let Some(las_path) = &cli.las_rgb {
                        let (las_min, las_max) = las_source::read_las_bounds(las_path)?;
                        info!("LAS bounding box: [{:.1}, {:.1}, {:.1}] to [{:.1}, {:.1}, {:.1}]",
                              las_min[0], las_min[1], las_min[2], las_max[0], las_max[1], las_max[2]);
                        features = las_source::filter_features_by_las_bbox(
                            features,
                            &cityjsonl_meta.transform,
                            &las_min,
                            &las_max,
                        );
                        Some((las_min[2], las_max[2]))
                    } else {
                        None
                    };

                    // Use smaller grid cells when splats dominate the payload
                    let effective_cellsize = if cli.splats {
                        let cs = grid_cellsize.min(100);
                        if cs != grid_cellsize {
                            info!("Reduced grid cell size from {} to {} for splat tiling", grid_cellsize, cs);
                        }
                        cs
                    } else {
                        grid_cellsize
                    };

                    // Build World from in-memory features
                    let mut world = parser::World::from_features(
                        cityjsonl_meta.transform,
                        cityjsonl_meta.reference_system,
                        features,
                        effective_cellsize,
                        cli.object_type,
                        cli.grid_minz,
                        cli.grid_maxz,
                    )?;

                    // Override grid Z with LAS z-bounds when footprints are 2D
                    if let Some((zmin, zmax)) = las_z_bounds {
                        let dz = zmax - zmin;
                        if dz > (world.grid.bbox[5] - world.grid.bbox[2]) {
                            info!("Setting grid Z from LAS bounds: [{:.1}, {:.1}]", zmin, zmax);
                            world.grid.bbox[2] = zmin;
                            world.grid.bbox[5] = zmax;
                        }
                    }

                    world.index_with_grid();
                    world
                }
                // Trees only (no buildings) — fully in-memory, no _prep directory
                (None, Some(trees_path)) => {
                    let (tree_transform, tree_ref_system, mut tree_features) =
                        geoparquet_source::load_geoparquet_standalone_to_memory(
                            trees_path,
                            "SolitaryVegetationObject",
                            cli.tree_id_column.as_deref(),
                        )?;
                    info!("Loaded {} tree features into memory (standalone)", tree_features.len());

                    // Filter features to LAS bounding box (if --las-rgb provided)
                    // and inject LAS z-bounds into grid when footprints are 2D.
                    let las_z_bounds = if let Some(las_path) = &cli.las_rgb {
                        let (las_min, las_max) = las_source::read_las_bounds(las_path)?;
                        info!("LAS bounding box: [{:.1}, {:.1}, {:.1}] to [{:.1}, {:.1}, {:.1}]",
                              las_min[0], las_min[1], las_min[2], las_max[0], las_max[1], las_max[2]);
                        let filtered = las_source::filter_features_by_las_bbox(
                            tree_features,
                            &tree_transform,
                            &las_min,
                            &las_max,
                        );
                        tree_features = filtered;
                        Some((las_min[2], las_max[2]))
                    } else {
                        None
                    };

                    // Use smaller grid cells when splats dominate the payload
                    let effective_cellsize = if cli.splats {
                        let cs = grid_cellsize.min(100);
                        if cs != grid_cellsize {
                            info!("Reduced grid cell size from {} to {} for splat tiling", grid_cellsize, cs);
                        }
                        cs
                    } else {
                        grid_cellsize
                    };

                    let mut world = parser::World::from_features(
                        tree_transform,
                        tree_ref_system,
                        tree_features,
                        effective_cellsize,
                        cli.object_type,
                        cli.grid_minz,
                        cli.grid_maxz,
                    )?;

                    // Override grid Z with LAS z-bounds when footprints are 2D
                    if let Some((zmin, zmax)) = las_z_bounds {
                        let dz = zmax - zmin;
                        if dz > (world.grid.bbox[5] - world.grid.bbox[2]) {
                            info!("Setting grid Z from LAS bounds: [{:.1}, {:.1}]", zmin, zmax);
                            world.grid.bbox[2] = zmin;
                            world.grid.bbox[5] = zmax;
                        }
                    }
                    world.index_with_grid();
                    world
                }
                // Unreachable: clap ArgGroup guarantees at least one
                (None, None) => unreachable!("clap ArgGroup requires --buildings or --trees"),
            }
        }
        Some(world_path) => {
            debug!("Loading world from bincode {world_path:?}");
            let world_file = File::open(world_path)?;
            bincode::deserialize_from(world_file)?
        }
    };
    // --- End preprocessing pipeline ---

    debug!(
        "Computed grid statistics: {}",
        world.grid.compute_statistics()
    );

    if cli.grid_export {
        debug!("Exporting the grid to TSV to {:?}", &debug_data_output_path);
        world.export_grid(cli.grid_export_features, Some(&debug_data_output_path))?;
    }
    if log_enabled!(Level::Debug) {
        debug!(
            "Exporting the world instance to bincode to {:?}",
            &debug_data_output_path
        );
        debug!("[Progress] Starting world bincode export...");
        debug!("[Progress] This may take a while for large datasets ({} features)", 
              world.features.len());
        world.export_bincode(Some("world"), Some(&debug_data_output_path))?;
        debug!("[Progress] Completed world bincode export");
    }

    // Build quadtree
    // When splats are the primary payload, force maximum subdivision so each grid cell
    // becomes its own tile. This prevents a single multi-GB GLB when the building geometry
    // (which the quadtree normally uses for capacity) is sparse.
    let effective_qtree_capacity = if cli.splats {
        info!("Splats mode: forcing quadtree leaf-per-cell (capacity=0)");
        spatial_structs::QuadTreeCapacity::Vertices(0)
    } else {
        quadtree_capacity
    };
    debug!("[Progress] Starting quadtree construction...");
    let quadtree: spatial_structs::QuadTree = match debug_data.quadtree {
        None => {
            debug!("Building quadtree");
            let quadtree = spatial_structs::QuadTree::from_world(&world, effective_qtree_capacity);
            debug!("[Progress] Completed quadtree construction");
            quadtree
        }
        Some(quadtree_path) => {
            debug!("Loading quadtree from bincode {quadtree_path:?}");
            let quadtree_file = File::open(quadtree_path)?;
            let quadtree = bincode::deserialize_from(quadtree_file)?;
            debug!("[Progress] Completed quadtree loading from bincode");
            quadtree
        }
    };

    if cli.grid_export {
        debug!(
            "Exporting the quadtree to TSV to {:?}",
            &debug_data_output_path
        );
        quadtree.export(&world, Some(&debug_data_output_path))?;
    }
    if log_enabled!(Level::Debug) {
        debug!(
            "Exporting the quadtree instance to bincode to {:?}",
            &debug_data_output_path
        );
        debug!("[Progress] Starting quadtree bincode export...");
        quadtree.export_bincode(Some("quadtree"), Some(&debug_data_output_path))?;
        debug!("[Progress] Completed quadtree bincode export");
    }

    // 3D Tiles
    debug!("[Progress] Starting tileset generation...");

    // Pre-compute CRS string early — needed for splat loading and tileset generation.
    let epsg_code = world
        .crs
        .to_epsg()
        .map_err(|e| format!("Failed to read EPSG code from metadata: {}", e))?;
    let crs_from = format!("EPSG:{}", epsg_code);

    // Load LAS/LAZ splat cloud before tileset generation so we can extract LOD config.
    let splat_cloud: Option<std::sync::Arc<las_source::SplatCloud>> = match &cli.las_rgb {
        Some(las_path) if cli.splats => {
            info!("Loading LAS/LAZ point cloud from {:?}", las_path);
            let cloud = las_source::load_las_as_splats(las_path, &crs_from, &world.grid, cli.splat_lod_tiers, cli.de_noising)?;
            info!("Indexed {} splats across {} grid cells ({} LOD tiers)",
                cloud.splats.len(), cloud.cell_index.len(), cloud.n_lod_tiers);
            Some(std::sync::Arc::new(cloud))
        }
        _ => None,
    };

    // Build SplatLodConfig from the loaded cloud (if any) for tileset generation.
    let splat_lod_config: Option<SplatLodConfig> = splat_cloud.as_ref().map(|cloud| {
        SplatLodConfig {
            n_tiers: cloud.n_lod_tiers,
            tier_spacings: cloud.tier_spacings.clone(),
        }
    });

    let tileset_path = cli.output.join("tileset.json");
    let subtrees_path = cli.output.join("subtrees");
    let tileset_path_unpruned = cli.output.join("tileset_unpruned.json");
    let subtrees_path_unpruned = cli.output.join("subtrees_unpruned");
    debug!("Generating 3D Tiles tileset");
    let mut tileset = formats::cesium3dtiles::Tileset::from_quadtree(
        &quadtree,
        &world,
        geometric_error_above_leaf,
        grid_cellsize,
        cli.grid_minz,
        cli.grid_maxz,
        cli.cesium3dtiles_content_bv_from_tile,
        cli.cesium3dtiles_content_add_bv,
        cli.tiles_version,
        splat_lod_config.as_ref(),
    );
    debug!("[Progress] Completed tileset generation");

    if cli.grid_export {
        debug!(
            "Exporting the explicit tileset to TSV files to {:?}",
            &debug_data_output_path
        );
        tileset.export(Some(&debug_data_output_path))?;
    }

    debug!("[Progress] Starting tile collection...");
    let (tiles, _subtrees) = match cli.cesium3dtiles_implicit {
        true => {
            let mut tileset_implicit = tileset.clone();
            // FIXME: here we have a Vec<(Tile, TileId)> in 'tiles' instead of Vec<&Tile>, because of the
            //  mess with the implicit/explicit tile id-s.
            debug!("Converting to implicit tiling");
            // Tileset.make_implicit() outputs the tiles that have content. If only the leaves have
            //  content, then only the leaves are outputted.
            let components: Vec<_> = subtrees_path_unpruned
                .components()
                .map(|comp| comp.as_os_str())
                .collect();
            let subtrees_dir_option = components.last().cloned().unwrap().to_str();
            let tiles_subtrees = tileset_implicit.make_implicit(
                &world.grid,
                &quadtree,
                cli.grid_export,
                subtrees_dir_option,
                Some(&debug_data_output_path),
            );

            if cli.cesium3dtiles_tileset_only || log_enabled!(Level::Debug) {
                debug!("Writing unpruned 3D Tiles tileset");
                tileset_implicit.to_file(&tileset_path_unpruned)?;

                debug!("Writing unpruned subtrees for implicit tiling");
                fs::create_dir_all(&subtrees_path_unpruned)?;
                for (subtree_id, subtree_bytes) in &tiles_subtrees.1 {
                    fs::create_dir_all(
                        subtrees_path_unpruned
                            .join(format!("{}/{}", subtree_id.level, subtree_id.x)),
                    )
                    .unwrap();
                    let out_path = subtrees_path_unpruned
                        .join(&subtree_id.to_string())
                        .with_extension("subtree");
                    let mut subtree_file = File::create(&out_path)
                        .unwrap_or_else(|_| panic!("could not create {:?} for writing", &out_path));
                    if let Err(_e) = subtree_file.write_all(subtree_bytes) {
                        warn!("Failed to write subtree {} content", subtree_id);
                    }
                }
            }

            tiles_subtrees
        }
        false => {
            let just_tiles = if splat_lod_config.is_some() {
                tileset.collect_all_with_content()
            } else {
                tileset.collect_leaves()
            };
            // FIXME: here we need Vec<(Tile, TileId)> instead of Vec<&Tile>, for the same reason
            //  as above
            let tiles: Vec<(Tile, TileId)> = just_tiles
                .into_iter()
                .map(|tile_ref| (tile_ref.clone(), tile_ref.id.clone()))
                .collect();

            debug!("Writing unpruned 3D Tiles tileset");
            tileset.to_file(&tileset_path_unpruned)?;
            debug!("[Progress] Completed tile collection, found {} tiles", tiles.len());

            (tiles, vec![])
        }
    };

    let path_output_tiles = cli.output.join("t");
    if !cli.cesium3dtiles_tileset_only {
        fs::create_dir_all(&path_output_tiles)?;
        debug!("Created output directory {:#?}", &path_output_tiles);

        // Compute root center in ECEF once for all tiles.
        let root_bbox = quadtree.bbox(&world.grid);
        let root_center_input_crs = [
            (root_bbox[0] + root_bbox[3]) * 0.5,
            (root_bbox[1] + root_bbox[4]) * 0.5,
            (root_bbox[2] + root_bbox[5]) * 0.5,
        ];
        let root_center_proj = crate::proj::Proj::new_known_crs(&crs_from, "EPSG:4978", None)
            .map_err(|e| format!("Create CRS to ECEF transformer: {}", e))?;
        let root_center_ecef = root_center_proj
            .convert((root_center_input_crs[0], root_center_input_crs[1], root_center_input_crs[2]))
            .map_err(|e| format!("Transform root center to ECEF: {}", e))?;
        debug!("Pre-computed root_center_ecef: {:?}", root_center_ecef);

        let splat_cloud_ref = splat_cloud.as_deref();

        let tiles_len = tiles.len();
        debug!("Starting to process {} tiles with native glTF generation...", tiles_len);
        let processed_count = AtomicUsize::new(0);
        let lod_filter = cli.lod.as_deref();
        let tile_lod_map = &tileset.tile_lod_map;
        let tiles_failed_iter = tiles.into_par_iter().map(|(tile, tileid)| {
            let tileid_grid = &tile.id;
            let qtree_nodeid: spatial_structs::QuadTreeNodeId = tileid_grid.into();
            let qtree_node = quadtree
                .node(&qtree_nodeid)
                .unwrap_or_else(|| panic!("did not find tile {} in quadtree", tileid_grid));
            // Skip empty tiles only when NOT in splat LOD mode (in LOD mode, even tiles
            // with 0 building features may have splat content).
            if qtree_node.nr_items == 0 && splat_cloud_ref.is_none() {
                let count = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
                if count % 10 == 0 || count == tiles_len {
                    debug!("Progress: {}/{} tiles processed ({}%)", count, tiles_len, (count * 100) / tiles_len);
                }
                if log_enabled!(Level::Debug) {
                    debug!("Tile is empty ({}), skipping conversion", tileid_grid);
                }
                return None;
            }
            let tileid_string = tileid.to_string();
            let file_name = tileid_string;
            let output_file = path_output_tiles.join(&file_name).with_extension(cli.tiles_version.extension());
            // Ensure parent directories exist for nested tile paths (e.g., t/0/0/0.glb)
            if let Some(parent) = output_file.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if log_enabled!(Level::Debug) {
                debug!("Writing native GLB for tile {} to {:?}", tile.id, output_file);
            }
            let splat_lod_tier = tile_lod_map.get(tileid_grid).copied();
            match gltf_writer::write_tile_glb(&world, &quadtree, qtree_nodeid, &output_file, &material_config, cli.tiles_version, lod_filter, attr_whitelist.as_ref(), &crs_from, root_center_ecef, splat_cloud_ref, splat_lod_tier, cli.splats_only) {
                Ok(_) => {
                    let count = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
                    if count % 10 == 0 || count == tiles_len {
                        debug!("Progress: {}/{} tiles processed ({}%)", count, tiles_len, (count * 100) / tiles_len);
                    }
                    if log_enabled!(Level::Debug) {
                        debug!("Successfully wrote GLB for tile {}", tile.id);
                    }
                    None
                }
                Err(err) => {
                    let count = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
                    if count % 10 == 0 || count == tiles_len {
                        debug!("Progress: {}/{} tiles processed ({}%)", count, tiles_len, (count * 100) / tiles_len);
                    }
                    warn!("Tile {} conversion failed: {}", tile.id, err);
                    Some(tile)
                }
            }
        });

        let mut tiles_results: Vec<Option<Tile>> = Vec::with_capacity(tiles_len + 2);
        if let Some(tiles_results_path) = debug_data.tiles_results {
            debug!("Loading tiles_results from {tiles_results_path:?}");
            let tiles_results_file = File::open(tiles_results_path)?;
            tiles_results = bincode::deserialize_from(tiles_results_file)?
        } else {
            debug!("Converting and optimizing {tiles_len} tiles");
            tiles_failed_iter.collect_into_vec(&mut tiles_results);
            if log_enabled!(Level::Debug) {
                debug!(
                    "Exporting the tiles_results instance to bincode to {:?}",
                    &debug_data_output_path
                );
                let outpath = debug_data_output_path.join("tiles_results.bincode");
                let tiles_results_file = File::create(outpath)?;
                bincode::serialize_into(tiles_results_file, &tiles_results)?;
            }
        }
        let tiles_failed: Vec<Tile> = tiles_results.into_iter().flatten().collect();
        debug!("Done");

        debug!("Pruning tileset of {} failed tiles", tiles_failed.len());
        for (i, failed) in tiles_failed.iter().enumerate() {
            debug!("{}, removing failed from the tileset: {}", i, failed.id);
        }
        // Remove tiles that failed the gltf conversion
        tileset.prune(&tiles_failed, &quadtree);
        if cli.cesium3dtiles_implicit {
            // FIXME: here we re-create the implicit tileset from the pruned tileset,
            //  because it is simpler than flipping the bits of the unavailable tiles,
            //  because of the mixed up explicit/implicit tile IDs. But ideally, we
            //  flip the bits, so we won't need to duplicate the tileset here.
            let components: Vec<_> = subtrees_path
                .components()
                .map(|comp| comp.as_os_str())
                .collect();
            let subtrees_dir_option = components.last().cloned().unwrap().to_str();
            let (_, subtrees) = tileset.make_implicit(
                &world.grid,
                &quadtree,
                cli.grid_export,
                subtrees_dir_option,
                Some(&debug_data_output_path),
            );
            debug!("Writing subtrees for implicit tiling");
            fs::create_dir_all(&subtrees_path)?;
            for (subtree_id, subtree_bytes) in subtrees {
                fs::create_dir_all(
                    subtrees_path.join(format!("{}/{}", subtree_id.level, subtree_id.x)),
                )
                .unwrap();
                let out_path = subtrees_path
                    .join(&subtree_id.to_string())
                    .with_extension("subtree");
                let mut subtree_file = File::create(&out_path)
                    .unwrap_or_else(|_| panic!("could not create {:?} for writing", &out_path));
                if let Err(_e) = subtree_file.write_all(&subtree_bytes) {
                    warn!("Failed to write subtree {} content", subtree_id);
                }
            }
        } else {
            let available_levels = tileset.available_levels();
            // A five level deep tree is still managable in size.
            if available_levels > 5 {
                // Try to find the split where each child tileset starts to have more tiles in their
                // tree, than the ancestor tree. This way, the main tileset is smaller in size than
                // the child tilesets, so it loads faster. This method is not very accurate, because
                // it doesn't account for the actual number of tiles on each level, it only
                // calculates with the theoretical maximum.
                let mut split_at_level = 0;
                for level in (0..available_levels).rev() {
                    let subtree_depth: u32 = (available_levels - level) as u32;
                    let nr_tiles_subtree = (4_usize.pow(subtree_depth) - 1) / 3;
                    let ancestor_tree_depth: u32 =
                        (available_levels - (available_levels - level)) as u32;
                    let nr_tiles_ancestor = (4_usize.pow(ancestor_tree_depth) - 1) / 3;
                    if nr_tiles_ancestor < nr_tiles_subtree {
                        split_at_level = level;
                        break;
                    }
                }
                debug!(
                    "Splitting the explicit tileset into external tilesets at level {}",
                    split_at_level
                );
                let external_tilesets = tileset.split(split_at_level);
                for (filename, child_tileset) in &external_tilesets {
                    let tileset_path = cli.output.join(filename);
                    child_tileset.to_file(&tileset_path)?;
                }
            }
        }
        debug!("Writing 3D Tiles tileset");
        tileset.to_file(&tileset_path)?;
    }

    Ok(())
}
