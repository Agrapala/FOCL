/// main.rs — FLoBC Rust Blockchain Node
///
/// Actix-web HTTP server that stores the FL blockchain and runs pBFT
/// consensus on incoming block proposals from the Python FL engine.
///
/// This replaces the Exonum Blockchain backend referenced in the paper
/// (Exonum was discontinued in 2022).  The architecture is identical:
///   - pBFT consensus with >2/3 stake threshold
///   - ECDSA secp256k1 identity (address sent in each transaction)
///   - Cryptographically linked blocks with Merkle root integrity
///   - REST HTTP API (replaces Exonum's built-in HTTP API)
///
/// Endpoints
/// ---------
///   GET  /health              Node liveness check
///   GET  /chain               Full chain (JSON)
///   GET  /chain/length        Block count
///   GET  /chain/valid         Chain integrity check
///   GET  /chain/{index}       Single block
///   POST /block/propose       Propose + commit a block (pBFT vote included)
///   GET  /transactions        All transactions across all blocks
///   GET  /stats               Chain statistics summary
///
/// Wire API (read-only, for JavaScript client and monitoring)
/// ---------
///   GET  /wire/chain          Same as /chain (semantic alias)
///   GET  /wire/trust          Trust scores forwarded from Python (stored here)
///   POST /wire/trust          Python posts trust scores after each round
///   GET  /wire/accuracy       Accuracy log forwarded from Python
///   POST /wire/accuracy       Python posts accuracy after each round
///   GET  /wire/nodes          Known nodes (wallet addresses)
///   POST /wire/nodes          Python registers a node
///
/// Start with:
///   cargo run --release        (listens on 127.0.0.1:8100)
/// Or:
///   cargo build --release && ./target/release/flobc-blockchain

use actix_cors::Cors;
use actix_web::{middleware, web, App, HttpResponse, HttpServer, Result};
use std::collections::HashMap;
use std::sync::Mutex;

mod block;
mod chain;
mod consensus;

use block::ProposeRequest;
use chain::Blockchain;

// ─────────────────────────────────────────────────────────────────────────────
// Shared application state
// ─────────────────────────────────────────────────────────────────────────────

struct AppState {
    chain:         Mutex<Blockchain>,
    trust_scores:  Mutex<HashMap<String, f64>>,    // tid -> score
    accuracy_log:  Mutex<Vec<f64>>,               // round -> global accuracy
    nodes:         Mutex<Vec<serde_json::Value>>,  // registered node info
}

// ─────────────────────────────────────────────────────────────────────────────
// Blockchain endpoints
// ─────────────────────────────────────────────────────────────────────────────

async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "node":   "flobc-rust-blockchain",
        "version": "0.1.0",
        "consensus": "pBFT PoS (>2/3 threshold)",
        "crypto":    "ECDSA secp256k1",
    }))
}

async fn get_chain(data: web::Data<AppState>) -> HttpResponse {
    let chain = data.chain.lock().unwrap();
    HttpResponse::Ok().json(chain.to_json())
}

async fn get_length(data: web::Data<AppState>) -> HttpResponse {
    let chain = data.chain.lock().unwrap();
    HttpResponse::Ok().json(serde_json::json!({ "length": chain.length() }))
}

async fn check_valid(data: web::Data<AppState>) -> HttpResponse {
    let chain = data.chain.lock().unwrap();
    HttpResponse::Ok().json(serde_json::json!({ "valid": chain.is_valid() }))
}

async fn get_block(
    data: web::Data<AppState>,
    path: web::Path<usize>,
) -> HttpResponse {
    let chain = data.chain.lock().unwrap();
    match chain.get_block(path.into_inner()) {
        Some(b) => HttpResponse::Ok().json(b),
        None    => HttpResponse::NotFound()
                       .json(serde_json::json!({ "error": "block not found" })),
    }
}

async fn get_all_transactions(data: web::Data<AppState>) -> HttpResponse {
    let chain = data.chain.lock().unwrap();
    let txs: Vec<_> = chain.blocks.iter()
        .skip(1)  // skip genesis
        .flat_map(|b| b.transactions.iter().map(move |tx| {
            serde_json::json!({
                "block_index": b.index,
                "block_hash":  &b.block_hash[..14],
                "tx_type":     tx.tx_type,
                "sender":      &tx.sender[..14],
                "tx_hash":     &tx.tx_hash[..14],
                "timestamp":   tx.timestamp,
                "payload":     &tx.payload,
            })
        }))
        .collect();
    HttpResponse::Ok().json(serde_json::json!({
        "total": txs.len(),
        "transactions": txs,
    }))
}

async fn get_stats(data: web::Data<AppState>) -> HttpResponse {
    let chain     = data.chain.lock().unwrap();
    let trust     = data.trust_scores.lock().unwrap();
    let acc_log   = data.accuracy_log.lock().unwrap();
    let nodes     = data.nodes.lock().unwrap();

    let total_tx: usize = chain.blocks.iter().map(|b| b.transactions.len()).sum();
    let mut tx_types: HashMap<String, usize> = HashMap::new();
    for b in &chain.blocks {
        for tx in &b.transactions {
            *tx_types.entry(tx.tx_type.clone()).or_insert(0) += 1;
        }
    }

    HttpResponse::Ok().json(serde_json::json!({
        "chain_length":    chain.length(),
        "chain_valid":     chain.is_valid(),
        "total_tx":        total_tx,
        "tx_by_type":      tx_types,
        "n_nodes":         nodes.len(),
        "n_trust_entries": trust.len(),
        "n_accuracy_pts":  acc_log.len(),
        "final_accuracy":  acc_log.last().copied().unwrap_or(0.0),
        "consensus":       "pBFT PoS >2/3",
        "crypto":          "ECDSA secp256k1",
    }))
}

/// Core endpoint: receive a block proposal with pBFT votes, run consensus,
/// and commit the block if >2/3 stake agrees.
async fn propose_block(
    data: web::Data<AppState>,
    body: web::Json<ProposeRequest>,
) -> HttpResponse {
    // ── pBFT consensus ──────────────────────────────────────────────────────
    let result = consensus::pbft_vote(&body.votes, body.threshold);

    if !result.accepted {
        return HttpResponse::Conflict().json(serde_json::json!({
            "committed":   false,
            "error":       "pBFT consensus not reached",
            "yes_ratio":   result.yes_ratio,
            "threshold":   result.threshold,
            "yes_count":   result.yes_count,
            "voter_count": result.voter_count,
        }));
    }

    // ── Commit block ─────────────────────────────────────────────────────────
    let mut chain = data.chain.lock().unwrap();

    let stake_votes: HashMap<String, f64> = body.votes.iter()
        .map(|v| (v.validator.clone(), v.stake))
        .collect();

    let block = chain.add_block(
        body.transactions.clone(),
        body.validator.clone(),
        stake_votes,
    );

    let resp = serde_json::json!({
        "committed":    true,
        "block_index":  block.index,
        "block_hash":   block.block_hash,
        "tx_count":     block.transactions.len(),
        "merkle_root":  block.merkle_root,
        "yes_ratio":    result.yes_ratio,
        "threshold":    result.threshold,
        "chain_length": chain.length(),
    });

    println!(
        "  [Rust-BC] Block #{} committed | txs={} | yes={:.2}/{:.2} | hash={}...",
        block.index,
        block.transactions.len(),
        result.yes_stake,
        result.total_stake,
        &block.block_hash[..14],
    );

    HttpResponse::Ok().json(resp)
}

// ─────────────────────────────────────────────────────────────────────────────
// Wire API endpoints  (queried by JavaScript client)
// ─────────────────────────────────────────────────────────────────────────────

async fn wire_chain(data: web::Data<AppState>) -> HttpResponse {
    let chain = data.chain.lock().unwrap();
    HttpResponse::Ok().json(chain.to_json())
}

async fn wire_get_trust(data: web::Data<AppState>) -> HttpResponse {
    let trust = data.trust_scores.lock().unwrap();
    HttpResponse::Ok().json(&*trust)
}

async fn wire_post_trust(
    data: web::Data<AppState>,
    body: web::Json<HashMap<String, f64>>,
) -> HttpResponse {
    let mut trust = data.trust_scores.lock().unwrap();
    *trust = body.into_inner();
    HttpResponse::Ok().json(serde_json::json!({ "ok": true }))
}

async fn wire_get_accuracy(data: web::Data<AppState>) -> HttpResponse {
    let log = data.accuracy_log.lock().unwrap();
    HttpResponse::Ok().json(serde_json::json!({
        "accuracy_log": &*log,
        "final": log.last().copied().unwrap_or(0.0),
        "max":   log.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    }))
}

async fn wire_post_accuracy(
    data: web::Data<AppState>,
    body: web::Json<Vec<f64>>,
) -> HttpResponse {
    let mut log = data.accuracy_log.lock().unwrap();
    *log = body.into_inner();
    HttpResponse::Ok().json(serde_json::json!({ "ok": true }))
}

async fn wire_get_nodes(data: web::Data<AppState>) -> HttpResponse {
    let nodes = data.nodes.lock().unwrap();
    HttpResponse::Ok().json(&*nodes)
}

async fn wire_post_node(
    data: web::Data<AppState>,
    body: web::Json<serde_json::Value>,
) -> HttpResponse {
    let mut nodes = data.nodes.lock().unwrap();
    nodes.push(body.into_inner());
    HttpResponse::Ok().json(serde_json::json!({
        "ok": true,
        "n_nodes": nodes.len(),
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Server startup
// ─────────────────────────────────────────────────────────────────────────────

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("=================================================================");
    println!("  FLoBC Rust Blockchain Node");
    println!("  pBFT Proof-of-Stake consensus  |  ECDSA secp256k1 identity");
    println!("  Listening on http://127.0.0.1:8100");
    println!("  Wire API  on http://127.0.0.1:8100/wire/*");
    println!("=================================================================");

    let state = web::Data::new(AppState {
        chain:        Mutex::new(Blockchain::new()),
        trust_scores: Mutex::new(HashMap::new()),
        accuracy_log: Mutex::new(Vec::new()),
        nodes:        Mutex::new(Vec::new()),
    });

    HttpServer::new(move || {
        // Allow JavaScript client (any origin) to call the Wire API
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();

        App::new()
            .wrap(cors)
            .app_data(state.clone())
            .app_data(
                web::JsonConfig::default()
                    .error_handler(|err, req| {
                        let msg = format!("JSON parse error: {}", err);
                        actix_web::error::InternalError::from_response(
                            err,
                            HttpResponse::BadRequest().json(
                                serde_json::json!({ "error": msg })),
                        )
                        .into()
                    }),
            )
            // ── Blockchain API ─────────────────────────────────────────────
            .route("/health",              web::get().to(health))
            .route("/chain",               web::get().to(get_chain))
            .route("/chain/length",        web::get().to(get_length))
            .route("/chain/valid",         web::get().to(check_valid))
            .route("/chain/{index}",       web::get().to(get_block))
            .route("/transactions",        web::get().to(get_all_transactions))
            .route("/stats",               web::get().to(get_stats))
            .route("/block/propose",       web::post().to(propose_block))
            // ── Wire API (read-mostly, for JS client) ─────────────────────
            .route("/wire/chain",          web::get().to(wire_chain))
            .route("/wire/trust",          web::get().to(wire_get_trust))
            .route("/wire/trust",          web::post().to(wire_post_trust))
            .route("/wire/accuracy",       web::get().to(wire_get_accuracy))
            .route("/wire/accuracy",       web::post().to(wire_post_accuracy))
            .route("/wire/nodes",          web::get().to(wire_get_nodes))
            .route("/wire/nodes",          web::post().to(wire_post_node))
    })
    .bind("127.0.0.1:8100")?
    .run()
    .await
}
