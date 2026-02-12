use atelier_base::{
    data::{Dataset, Transformation},
    templates,
};
use atelier_dcml::{distributed::processes, functions, metrics, models, optimizers};
use std::path::Path;

pub fn main() {
    println!("\n------ case_3 - training ------ ");

    // --- Setup working directory
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir)
        .parent()
        .expect("Failed to get workspace root");

    // --- Template file (toml)
    let template_file = workspace_root
        .join("examples")
        .join("distributed")
        .join("case_3")
        .join("config.toml");

    let template = templates::Config::load_from_toml(template_file.to_str().unwrap())
        .unwrap()
        .clone();

    // --- Extract parameters from template
    let _exp_id = &template.experiments[0].id;
    let _n_progres = template.experiments[0].n_progressions as usize;
    let optimizer_model = template.models[0].params_values.clone().unwrap();

    // ------------------------------------------------------------------- DATASET --- //
    // ------------------------------------------------------------------- ------- --- //

    let mut v_datasets = vec![];
    let v_datafiles = vec!["ai_00", "eu_00", "am_00"];

    for i_case in 0..3 {
        // File specification and read
        let data_file = workspace_root
            .join("examples")
            .join("distributed")
            .join("case_3")
            .join("files")
            .to_str()
            .unwrap()
            .to_owned()
            + "/case_3_"
            + v_datafiles[i_case]
            + "_data.csv";

        let header = true;
        let column_types = None;
        let target_column = Some(7);

        println!("\ndata_file: {:?}", data_file);

        // Dataset format parsing
        let mut b_dataset =
            Dataset::from_csv(&data_file, header, column_types, target_column).unwrap();

        // Feature transformation
        b_dataset.transform(Transformation::Scale);

        v_datasets.push(b_dataset);
    }

    // Number of inputs (features)
    let n_features = 6;

    // -------------------------------------------------------------------- MODELS --- //
    // -------------------------------------------------------------------- ------ --- //

    let n_models = 3;
    let mut v_models: Vec<models::LinearModel> = Vec::new();

    for i in 0..n_models {
        v_models.push(
            models::LinearModel::builder(n_features)
                .id("model_0".to_string().to_owned() + &i.to_string())
                .glorot_uniform_init(),
        )
    }

    println!("\nmodels: {:?}", v_models);

    // ------------------------------------------------------------------- METRICS --- //
    // ------------------------------------------------------------------- ------- --- //

    let v_metrics = vec![
        metrics::Metrics::basic_classification(),
        metrics::Metrics::basic_classification(),
        metrics::Metrics::basic_classification(),
    ];

    // ---------------------------------------------------------------------- LOSS --- //
    // ---------------------------------------------------------------------- ---- --- //

    let v_loss_functions = vec![
        functions::CrossEntropy::builder()
            .id(&"loss_00".to_string())
            .build()
            .unwrap(),
        functions::CrossEntropy::builder()
            .id(&"loss_01".to_string())
            .build()
            .unwrap(),
        functions::CrossEntropy::builder()
            .id(&"loss_02".to_string())
            .build()
            .unwrap(),
    ];

    // ------------------------------------------------------------------ TOPOLOGY --- //
    // ------------------------------------------------------------------ -------- --- //

    let topology_file = workspace_root
        .join("examples")
        .join("distributed")
        .join("case_3")
        .join("topology.toml");

    println!("\ntopology_file: {:?}", topology_file);

    // Create graph (matrix representation)
    let empty_matrix = processes::ConnectionsMatrix::new(3);
    let topology = empty_matrix.fill(topology_file.to_str().unwrap()).unwrap();
    println!("\ntopology: {:?}", topology);

    // ----------------------------------------------------------------- OPTIMIZER --- //
    // ----------------------------------------------------------------- --------- --- //

    let b_optimizer = optimizers::GradientDescent::builder()
        .id("opt_00".to_string())
        .learning_rate(optimizer_model[0].abs())
        .build()
        .unwrap();

    println!("\nb_optimizer: {:?}", b_optimizer);

    // ------------------------------------------------------------------- EXECUTE --- //
    // ----------------------------------------------------------------- --------- --- //

    // Create distributed trainer
    let mut distributed_trainer = processes::Distributed::builder()
        .datasets(v_datasets)
        .models(v_models)
        .loss(v_loss_functions)
        .metrics(v_metrics)
        .topology(topology)
        .optimizer(b_optimizer)
        .strategy(processes::UpdateStrategy::CombineThenAdapt) // or AdaptThenCombine
        .build()
        .unwrap();

    // Train the distributed system
    let _ = distributed_trainer.train(10);
}
