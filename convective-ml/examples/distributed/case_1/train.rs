//! Single-model training example using the `NalgebraBackend` (default) or the `TorchBackend`.
//!
//! Run with:
//!   cargo run --example distributed_case_1_train
//!
//! For the torch backend (requires libtorch):
//!   cargo run --example distributed_case_1_train --features torch

use atelier_data::datasets;
use atelier_dcml::{
    ComputeBackend, Model, TorchBackend, functions, models, optimizers, processes,
};
use std::error::Error;

/// Generate a synthetic binary-classification dataset.
///
/// Ground-truth rule:  y = 1  iff  w_true · x > 0
fn synthetic_dataset(n_samples: usize, n_features: usize) -> datasets::Dataset {
    use rand::Rng;
    let mut rng = rand::rng();

    // Hidden "true" weight vector (the model should learn something close)
    let w_true: Vec<f64> = (0..n_features)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    let mut features = Vec::with_capacity(n_samples);
    let mut target = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x: Vec<f64> = (0..n_features)
            .map(|_| rng.random_range(-2.0..2.0))
            .collect();

        let dot: f64 = x.iter().zip(w_true.iter()).map(|(xi, wi)| xi * wi).sum();
        // Label: 1.0 if dot > 0, else 0.0  (with a bit of noise)
        let noise: f64 = rng.random_range(-0.1..0.1);
        let y = if dot + noise > 0.0 { 1.0 } else { 0.0 };

        features.push(x);
        target.push(y);
    }

    datasets::Dataset::builder()
        .features(features)
        .target(target)
        .build()
        .expect("dataset build failed")
}

fn main() -> Result<(), Box<dyn Error + 'static>> {
    // --- Initialise tracing (optional, logs to stdout) ---
    tracing_subscriber::fmt::init();

    println!("\n=== atelier-dcml : singular training (TorchBackend) ===\n");

    // --- Step 1: Synthetic data ---
    let n_samples = 200;
    let n_features = 4;
    let dataset = synthetic_dataset(n_samples, n_features);
    println!(
        "Step 1: dataset  {} samples x {} features",
        dataset.len(),
        dataset.feature_count()
    );

    // --- Step 2: Model ---
    let model = models::LinearModel::<atelier_dcml::TorchBackend>::builder(n_features)
        .id("model_00".to_string())
        .glorot_uniform_init();

    println!("Step 2: model    id={:?}", model.id);

    // --- Step 3: Optimizer ---
    let optimizer = optimizers::GradientDescent::builder()
        .id("sgd_00".to_string())
        .learning_rate(0.05)
        .build()
        .unwrap();

    println!("Step 3: optimizer lr={}", optimizer.learning_rate);

    // --- Step 4: Loss function ---
    let loss = functions::CrossEntropy::builder()
        .id("bce_00")
        .build()
        .unwrap();

    println!("Step 4: loss     id={:?}\n", loss.id);

    // --- Step 5: Build trainer & run ---
    let mut trainer = processes::Singular::<atelier_dcml::TorchBackend>::builder()
        .dataset(dataset)
        .model(model)
        .loss(loss)
        .optimizer(optimizer)
        .build()
        .unwrap();

    let epochs = 1000;
    println!("Step 5: training for {epochs} epochs ...\n");
    trainer.train(epochs)?;

    // --- Step 6: Save model ---
    let model_path = "/tmp/atelier_singular_model.json";
    trainer.save_model(model_path)?;
    println!("\nStep 6: model saved to {model_path}");

    // --- Step 7: Quick inference sanity check ---
    let model_ref = trainer.model();
    let sample_features = TorchBackend::from_row_vecs(&[vec![0.5; n_features]]);
    let logits = model_ref.forward(&sample_features);

    let prob = 1.0 / (1.0 + -logits.exp());

    println!("Step 7: inference  logit={:.4}  p={:.4}", logits, prob);

    println!("\n=== done ===");
    Ok(())
}
