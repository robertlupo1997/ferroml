//! ONNX Export for SVM Models
//!
//! Implements ONNX export for LinearSVC, LinearSVR, SVC, and SVR.
//!
//! For kernel SVM (SVC/SVR), uses `ai.onnx.ml.SVMClassifier` and `ai.onnx.ml.SVMRegressor`
//! operators which natively support RBF, polynomial, and sigmoid kernels.

use crate::models::svm::{Kernel, LinearSVC, LinearSVR, MulticlassStrategy, SVC, SVR};
use crate::models::Model;
use crate::onnx::{
    create_add_node, create_argmax_node, create_cast_node, create_float_tensor, create_gemm_node,
    create_int64_tensor, create_matmul_node, create_model_proto, create_squeeze_node,
    create_tensor_input, create_tensor_output, create_tensor_output_1d, AttributeProto,
    AttributeProtoType, GraphProto, NodeProto, OnnxConfig, OnnxExportable, TensorProtoDataType,
};
use crate::{FerroError, Result};
use prost::Message;

impl OnnxExportable for LinearSVC {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let weights = self
            .weights()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let intercepts = self
            .intercepts()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let classes = self
            .classes()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let n_classes = classes.len();
        let n_outputs = weights.len();

        let batch_size = config.input_shape.map(|(b, _)| b);

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        // Weights: [n_features, n_outputs]
        let mut weights_flat: Vec<f32> = Vec::with_capacity(n_features * n_outputs);
        for feat in 0..n_features {
            for out in 0..n_outputs {
                weights_flat.push(weights[out][feat] as f32);
            }
        }
        let weights_tensor = create_float_tensor(
            "weights",
            &[n_features as i64, n_outputs as i64],
            weights_flat,
        );

        let bias: Vec<f32> = intercepts.iter().map(|&v| v as f32).collect();
        let bias_tensor = create_float_tensor("bias", &[n_outputs as i64], bias);

        let gemm_node = create_gemm_node(
            &config.input_name,
            "weights",
            "bias",
            "gemm_out",
            1.0,
            1.0,
            0,
            0,
        );

        let mut nodes = vec![gemm_node];
        let mut initializers = vec![weights_tensor, bias_tensor];

        if n_classes == 2 && n_outputs == 1 {
            // Binary: sign of decision function
            let squeeze_axes = create_int64_tensor("squeeze_axes", &[1], vec![1]);
            initializers.push(squeeze_axes);
            let squeeze = create_squeeze_node("gemm_out", "squeeze_axes", "squeezed", "Squeeze_0");
            nodes.push(squeeze);

            // Sign + Relu gives 0 or 1
            let sign_node = crate::onnx::NodeProto {
                input: vec!["squeezed".to_string()],
                output: vec!["signed".to_string()],
                name: "Sign_0".to_string(),
                op_type: "Sign".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
                doc_string: String::new(),
            };
            nodes.push(sign_node);

            let relu_node = crate::onnx::NodeProto {
                input: vec!["signed".to_string()],
                output: vec![config.output_name.clone()],
                name: "Relu_0".to_string(),
                op_type: "Relu".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
                doc_string: String::new(),
            };
            nodes.push(relu_node);
        } else {
            // Multi-class: ArgMax
            let argmax = create_argmax_node("gemm_out", "argmax_out", "ArgMax_0", 1, 0);
            nodes.push(argmax);

            let cast = create_cast_node(
                "argmax_out",
                &config.output_name,
                "Cast_0",
                TensorProtoDataType::Float,
            );
            nodes.push(cast);
        }

        let output =
            create_tensor_output_1d(&config.output_name, batch_size, TensorProtoDataType::Float);

        let graph = GraphProto {
            name: config.model_name.clone(),
            node: nodes,
            input: vec![input],
            output: vec![output],
            initializer: initializers,
            ..Default::default()
        };
        let model = create_model_proto(graph, config, false);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

impl OnnxExportable for LinearSVR {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let weights = self
            .weights()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let intercept = self.intercept();
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let batch_size = config.input_shape.map(|(b, _)| b);

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        let output =
            create_tensor_output_1d(&config.output_name, batch_size, TensorProtoDataType::Float);

        let weights_data: Vec<f32> = weights.iter().map(|&v| v as f32).collect();
        let weights_tensor = create_float_tensor("weights", &[n_features as i64, 1], weights_data);
        let bias_tensor = create_float_tensor("bias", &[1], vec![intercept as f32]);
        let squeeze_axes = create_int64_tensor("squeeze_axes", &[1], vec![1]);

        let matmul = create_matmul_node(&config.input_name, "weights", "matmul_out", "MatMul_0");
        let add = create_add_node("matmul_out", "bias", "add_out", "Add_0");
        let squeeze =
            create_squeeze_node("add_out", "squeeze_axes", &config.output_name, "Squeeze_0");

        let graph = GraphProto {
            name: config.model_name.clone(),
            node: vec![matmul, add, squeeze],
            input: vec![input],
            output: vec![output],
            initializer: vec![weights_tensor, bias_tensor, squeeze_axes],
            ..Default::default()
        };
        let model = create_model_proto(graph, config, false);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

// =============================================================================
// Kernel SVM: SVC (ai.onnx.ml.SVMClassifier)
// =============================================================================

/// Get ONNX kernel type string from Kernel enum
fn kernel_type_str(kernel: &Kernel) -> &'static str {
    match kernel {
        Kernel::Linear => "LINEAR",
        Kernel::Rbf { .. } => "RBF",
        Kernel::Polynomial { .. } => "POLY",
        Kernel::Sigmoid { .. } => "SIGMOID",
    }
}

/// Extract kernel parameters (gamma, coef0, degree) from Kernel enum
fn kernel_params(kernel: &Kernel) -> (f32, f32, i64) {
    match kernel {
        Kernel::Linear => (0.0, 0.0, 0),
        Kernel::Rbf { gamma } => (*gamma as f32, 0.0, 0),
        Kernel::Polynomial {
            gamma,
            coef0,
            degree,
        } => (*gamma as f32, *coef0 as f32, *degree as i64),
        Kernel::Sigmoid { gamma, coef0 } => (*gamma as f32, *coef0 as f32, 0),
    }
}

impl OnnxExportable for SVR {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let sv = self
            .support_vectors()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let dual_coef = self
            .dual_coef()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let batch_size = config.input_shape.map(|(b, _)| b);
        let n_sv = sv.nrows();

        // Flatten support vectors: row-major [n_sv * n_features]
        let sv_flat: Vec<f32> = sv.iter().map(|&v| v as f32).collect();
        let coefs: Vec<f32> = dual_coef.iter().map(|&v| v as f32).collect();
        let rho = vec![-(self.intercept() as f32)]; // ONNX uses negative rho

        let (gamma, coef0, degree) = kernel_params(&self.kernel);
        let kernel_type = kernel_type_str(&self.kernel);

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );
        let output =
            create_tensor_output_1d(&config.output_name, batch_size, TensorProtoDataType::Float);

        let attributes = vec![
            AttributeProto {
                name: "kernel_type".to_string(),
                s: kernel_type.as_bytes().to_vec(),
                r#type: AttributeProtoType::String as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "kernel_params".to_string(),
                floats: vec![gamma, coef0, degree as f32],
                r#type: AttributeProtoType::Floats as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "support_vectors".to_string(),
                floats: sv_flat,
                r#type: AttributeProtoType::Floats as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "coefficients".to_string(),
                floats: coefs,
                r#type: AttributeProtoType::Floats as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "rho".to_string(),
                floats: rho,
                r#type: AttributeProtoType::Floats as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "n_supports".to_string(),
                i: n_sv as i64,
                r#type: AttributeProtoType::Int as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "post_transform".to_string(),
                s: "NONE".as_bytes().to_vec(),
                r#type: AttributeProtoType::String as i32,
                ..Default::default()
            },
        ];

        let node = NodeProto {
            input: vec![config.input_name.clone()],
            output: vec![config.output_name.clone()],
            name: "SVMRegressor_0".to_string(),
            op_type: "SVMRegressor".to_string(),
            domain: "ai.onnx.ml".to_string(),
            attribute: attributes,
            doc_string: String::new(),
        };

        let graph = GraphProto {
            name: config.model_name.clone(),
            node: vec![node],
            input: vec![input],
            output: vec![output],
            ..Default::default()
        };
        let model = create_model_proto(graph, config, true);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

impl OnnxExportable for SVC {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let classes = self
            .classes()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let n_classes = classes.len();
        let classifiers = self.classifiers();
        let class_pairs = self.class_pairs();
        let batch_size = config.input_shape.map(|(b, _)| b);

        if classifiers.is_empty() {
            return Err(FerroError::not_fitted("No classifiers found in SVC"));
        }

        let (gamma, coef0, degree) = kernel_params(&self.kernel);
        let kernel_type = kernel_type_str(&self.kernel);

        // Build SVMClassifier attributes
        // For OvO: one set of support vectors per binary classifier
        // For OvR: one set of support vectors per class
        let mut all_sv: Vec<f32> = Vec::new();
        let mut all_coefs: Vec<f32> = Vec::new();
        let mut all_rho: Vec<f32> = Vec::new();
        let mut vectors_per_class: Vec<i64> = vec![0; n_classes];

        match self.multiclass_strategy {
            MulticlassStrategy::OneVsOne => {
                // OvO: each binary classifier has its own support vectors
                // ONNX SVMClassifier: support vectors are pooled per class
                // We need to collect support vectors per class and dual coefficients
                // per classifier pair

                // Collect unique support vectors per class
                let mut class_svs: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_classes];

                for (clf_idx, clf) in classifiers.iter().enumerate() {
                    let sv = match clf.support_vectors() {
                        Some(sv) => sv,
                        None => continue,
                    };
                    let dual = match clf.dual_coef() {
                        Some(dc) => dc,
                        None => continue,
                    };

                    let (class_i, class_j) = class_pairs[clf_idx];

                    // Find class indices
                    let ci = classes
                        .iter()
                        .position(|&c| (c - class_i).abs() < 1e-10)
                        .unwrap_or(0);
                    let cj = classes
                        .iter()
                        .position(|&c| (c - class_j).abs() < 1e-10)
                        .unwrap_or(1);

                    // Split support vectors into those from class_i and class_j
                    // In our BinarySVC, dual_coef = alpha * y, where y is +1 or -1
                    // Positive dual_coef => SV from positive class, negative => from negative class
                    for (sv_idx, &coef) in dual.iter().enumerate() {
                        let sv_row: Vec<f32> = sv.row(sv_idx).iter().map(|&v| v as f32).collect();
                        if coef >= 0.0 {
                            // This SV is from the positive class (class_i in BinarySVC)
                            class_svs[ci].push(sv_row);
                        } else {
                            // This SV is from the negative class (class_j in BinarySVC)
                            class_svs[cj].push(sv_row);
                        }
                    }

                    all_rho.push(-(clf.intercept() as f32));
                }

                // Flatten support vectors per class
                for (class_idx, svs) in class_svs.iter().enumerate() {
                    vectors_per_class[class_idx] = svs.len() as i64;
                    for sv in svs {
                        all_sv.extend_from_slice(sv);
                    }
                }

                // Build coefficient matrix
                // For OvO with n_classes, we need (n_classes * (n_classes-1)/2) rows
                // and total_sv columns
                let total_sv: usize = vectors_per_class.iter().sum::<i64>() as usize;
                let n_pairs = n_classes * (n_classes - 1) / 2;

                // Initialize coefficients to zero
                all_coefs = vec![0.0_f32; n_pairs * total_sv];

                // Fill in the coefficients for each pair
                let mut pair_idx = 0;
                let mut sv_offsets: Vec<usize> = vec![0; n_classes];
                let mut offset = 0;
                for (ci, &count) in vectors_per_class.iter().enumerate() {
                    sv_offsets[ci] = offset;
                    offset += count as usize;
                }

                for (clf_idx, clf) in classifiers.iter().enumerate() {
                    let dual = match clf.dual_coef() {
                        Some(dc) => dc,
                        None => {
                            pair_idx += 1;
                            continue;
                        }
                    };

                    let (class_i, class_j) = class_pairs[clf_idx];
                    let ci = classes
                        .iter()
                        .position(|&c| (c - class_i).abs() < 1e-10)
                        .unwrap_or(0);
                    let cj = classes
                        .iter()
                        .position(|&c| (c - class_j).abs() < 1e-10)
                        .unwrap_or(1);

                    let mut ci_count = 0usize;
                    let mut cj_count = 0usize;

                    for &coef in dual.iter() {
                        let abs_coef = coef.abs() as f32;
                        if coef >= 0.0 {
                            let sv_global_idx = sv_offsets[ci] + ci_count;
                            if sv_global_idx < total_sv {
                                all_coefs[pair_idx * total_sv + sv_global_idx] = abs_coef;
                            }
                            ci_count += 1;
                        } else {
                            let sv_global_idx = sv_offsets[cj] + cj_count;
                            if sv_global_idx < total_sv {
                                all_coefs[pair_idx * total_sv + sv_global_idx] = abs_coef;
                            }
                            cj_count += 1;
                        }
                    }

                    pair_idx += 1;
                }
            }
            MulticlassStrategy::OneVsRest => {
                // OvR: each classifier is class vs rest
                // Pool SVs by positive class index
                let mut class_svs: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_classes];

                for (clf_idx, clf) in classifiers.iter().enumerate() {
                    let sv = match clf.support_vectors() {
                        Some(sv) => sv,
                        None => continue,
                    };
                    let dual = match clf.dual_coef() {
                        Some(dc) => dc,
                        None => continue,
                    };

                    // In OvR, classifier clf_idx is class clf_idx vs rest
                    for (sv_idx, &coef) in dual.iter().enumerate() {
                        let sv_row: Vec<f32> = sv.row(sv_idx).iter().map(|&v| v as f32).collect();
                        if coef >= 0.0 {
                            class_svs[clf_idx].push(sv_row);
                        } else {
                            // SVs from "rest" - assign to the positive class anyway
                            // for the ONNX representation
                            class_svs[clf_idx].push(sv_row);
                        }
                    }

                    all_rho.push(-(clf.intercept() as f32));
                }

                for (class_idx, svs) in class_svs.iter().enumerate() {
                    vectors_per_class[class_idx] = svs.len() as i64;
                    for sv in svs {
                        all_sv.extend_from_slice(sv);
                    }
                }

                // Coefficients: one row per classifier, total_sv columns
                let total_sv: usize = vectors_per_class.iter().sum::<i64>() as usize;
                all_coefs = vec![0.0_f32; n_classes * total_sv];

                let mut sv_offset = 0usize;
                for (clf_idx, clf) in classifiers.iter().enumerate() {
                    let dual = match clf.dual_coef() {
                        Some(dc) => dc,
                        None => {
                            sv_offset += vectors_per_class[clf_idx] as usize;
                            continue;
                        }
                    };

                    for (sv_local, &coef) in dual.iter().enumerate() {
                        let sv_global = sv_offset + sv_local;
                        if sv_global < total_sv {
                            all_coefs[clf_idx * total_sv + sv_global] = coef.abs() as f32;
                        }
                    }

                    sv_offset += vectors_per_class[clf_idx] as usize;
                }
            }
        }

        let class_labels: Vec<i64> = classes.iter().map(|&c| c as i64).collect();

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );
        let label_name = format!("{}_labels", config.output_name);
        let scores_name = config.output_name.clone();

        let label_output =
            create_tensor_output_1d(&label_name, batch_size, TensorProtoDataType::Int64);
        let scores_output = create_tensor_output(
            &scores_name,
            n_classes,
            batch_size,
            TensorProtoDataType::Float,
        );

        let attributes = vec![
            AttributeProto {
                name: "kernel_type".to_string(),
                s: kernel_type.as_bytes().to_vec(),
                r#type: AttributeProtoType::String as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "kernel_params".to_string(),
                floats: vec![gamma, coef0, degree as f32],
                r#type: AttributeProtoType::Floats as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "support_vectors".to_string(),
                floats: all_sv,
                r#type: AttributeProtoType::Floats as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "coefficients".to_string(),
                floats: all_coefs,
                r#type: AttributeProtoType::Floats as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "rho".to_string(),
                floats: all_rho,
                r#type: AttributeProtoType::Floats as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "vectors_per_class".to_string(),
                ints: vectors_per_class,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "classlabels_ints".to_string(),
                ints: class_labels,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "post_transform".to_string(),
                s: "NONE".as_bytes().to_vec(),
                r#type: AttributeProtoType::String as i32,
                ..Default::default()
            },
        ];

        let node = NodeProto {
            input: vec![config.input_name.clone()],
            output: vec![label_name.clone(), scores_name.clone()],
            name: "SVMClassifier_0".to_string(),
            op_type: "SVMClassifier".to_string(),
            domain: "ai.onnx.ml".to_string(),
            attribute: attributes,
            doc_string: String::new(),
        };

        let graph = GraphProto {
            name: config.model_name.clone(),
            node: vec![node],
            input: vec![input],
            output: vec![label_output, scores_output],
            ..Default::default()
        };
        let model = create_model_proto(graph, config, true);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Model;
    use crate::onnx::ModelProto;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_linear_svc_binary_onnx_export() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 5.0, 8.0, 6.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut model = LinearSVC::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("linear_svc_binary");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert!(graph.node.iter().any(|n| n.op_type == "Gemm"));
    }

    #[test]
    fn test_linear_svr_onnx_export() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 2.0, 1.5, 3.0, 3.5, 4.0, 2.0, 5.0, 4.5, 6.0, 3.0, 7.0, 5.5, 8.0, 4.0,
                9.0, 6.5, 10.0, 5.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            4.0, 5.5, 9.5, 10.0, 14.5, 15.0, 19.5, 20.0, 24.5, 25.0,
        ]);

        let mut model = LinearSVR::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("linear_svr");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert!(graph.node.iter().any(|n| n.op_type == "MatMul"));
    }

    #[test]
    fn test_linear_svc_not_fitted() {
        let model = LinearSVC::new();
        let config = OnnxConfig::new("test");
        assert!(model.to_onnx(&config).is_err());
    }

    #[test]
    fn test_linear_svr_not_fitted() {
        let model = LinearSVR::new();
        let config = OnnxConfig::new("test");
        assert!(model.to_onnx(&config).is_err());
    }

    // =========================================================================
    // Kernel SVM: SVR with RBF kernel
    // =========================================================================

    #[test]
    fn test_svr_rbf_onnx_export() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 2.0, 1.5, 3.0, 3.5, 4.0, 2.0, 5.0, 4.5, 6.0, 3.0, 7.0, 5.5, 8.0, 4.0,
                9.0, 6.5, 10.0, 5.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            4.0, 5.5, 9.5, 10.0, 14.5, 15.0, 19.5, 20.0, 24.5, 25.0,
        ]);

        let mut model = SVR::new().with_kernel(Kernel::Rbf { gamma: 0.5 });
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("svr_rbf");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "SVMRegressor");
        assert_eq!(graph.node[0].domain, "ai.onnx.ml");

        // Check kernel type attribute
        let kernel_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "kernel_type")
            .unwrap();
        assert_eq!(kernel_attr.s, b"RBF");

        // Check kernel params
        let params_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "kernel_params")
            .unwrap();
        assert_eq!(params_attr.floats[0], 0.5); // gamma

        // Check support vectors exist
        let sv_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "support_vectors")
            .unwrap();
        assert!(!sv_attr.floats.is_empty());

        // Check ML opset domain is included
        assert!(onnx_model
            .opset_import
            .iter()
            .any(|op| op.domain == "ai.onnx.ml"));
    }

    #[test]
    fn test_svr_poly_onnx_export() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 2.0, 1.5, 3.0, 3.5, 4.0, 2.0, 5.0, 4.5, 6.0, 3.0, 7.0, 5.5, 8.0, 4.0,
                9.0, 6.5, 10.0, 5.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            4.0, 5.5, 9.5, 10.0, 14.5, 15.0, 19.5, 20.0, 24.5, 25.0,
        ]);

        let mut model = SVR::new().with_kernel(Kernel::Polynomial {
            gamma: 0.1,
            coef0: 1.0,
            degree: 3,
        });
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("svr_poly");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        let graph = onnx_model.graph.unwrap();

        assert_eq!(graph.node[0].op_type, "SVMRegressor");

        let kernel_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "kernel_type")
            .unwrap();
        assert_eq!(kernel_attr.s, b"POLY");

        let params_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "kernel_params")
            .unwrap();
        assert!((params_attr.floats[0] - 0.1).abs() < 1e-5); // gamma
        assert!((params_attr.floats[1] - 1.0).abs() < 1e-5); // coef0
        assert!((params_attr.floats[2] - 3.0).abs() < 1e-5); // degree
    }

    #[test]
    fn test_svr_sigmoid_onnx_export() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 2.0, 1.5, 3.0, 3.5, 4.0, 2.0, 5.0, 4.5, 6.0, 3.0, 7.0, 5.5, 8.0, 4.0,
                9.0, 6.5, 10.0, 5.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            4.0, 5.5, 9.5, 10.0, 14.5, 15.0, 19.5, 20.0, 24.5, 25.0,
        ]);

        let mut model = SVR::new().with_kernel(Kernel::Sigmoid {
            gamma: 0.01,
            coef0: 0.0,
        });
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("svr_sigmoid");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        let graph = onnx_model.graph.unwrap();

        assert_eq!(graph.node[0].op_type, "SVMRegressor");

        let kernel_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "kernel_type")
            .unwrap();
        assert_eq!(kernel_attr.s, b"SIGMOID");
    }

    #[test]
    fn test_svr_not_fitted() {
        let model = SVR::new();
        let config = OnnxConfig::new("test");
        assert!(model.to_onnx(&config).is_err());
    }

    // =========================================================================
    // Kernel SVM: SVC with RBF kernel
    // =========================================================================

    #[test]
    fn test_svc_binary_rbf_onnx_export() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 5.0, 8.0, 6.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut model = SVC::new().with_kernel(Kernel::Rbf { gamma: 0.5 });
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("svc_binary_rbf");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "SVMClassifier");
        assert_eq!(graph.node[0].domain, "ai.onnx.ml");

        // Check kernel type
        let kernel_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "kernel_type")
            .unwrap();
        assert_eq!(kernel_attr.s, b"RBF");

        // Check class labels
        let labels_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "classlabels_ints")
            .unwrap();
        assert_eq!(labels_attr.ints.len(), 2);

        // Check support vectors exist
        let sv_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "support_vectors")
            .unwrap();
        assert!(!sv_attr.floats.is_empty());

        // Check vectors_per_class
        let vpc_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "vectors_per_class")
            .unwrap();
        assert_eq!(vpc_attr.ints.len(), 2);
    }

    #[test]
    fn test_svc_multiclass_rbf_onnx_export() {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 5.0, 8.0, 6.0,
                9.0, 9.0, 10.0, 10.0, 11.0, 9.0, 12.0, 10.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
        ]);

        let mut model = SVC::new().with_kernel(Kernel::Rbf { gamma: 0.5 });
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("svc_multiclass_rbf");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        let graph = onnx_model.graph.unwrap();

        assert_eq!(graph.node[0].op_type, "SVMClassifier");

        // Check 3 class labels
        let labels_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "classlabels_ints")
            .unwrap();
        assert_eq!(labels_attr.ints.len(), 3);

        // Check vectors_per_class has 3 entries
        let vpc_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "vectors_per_class")
            .unwrap();
        assert_eq!(vpc_attr.ints.len(), 3);

        // OvO with 3 classes => 3 rho values
        let rho_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "rho")
            .unwrap();
        assert_eq!(rho_attr.floats.len(), 3);
    }

    #[test]
    fn test_svc_poly_onnx_export() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 5.0, 8.0, 6.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut model = SVC::new().with_kernel(Kernel::Polynomial {
            gamma: 0.1,
            coef0: 1.0,
            degree: 2,
        });
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("svc_poly");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        let graph = onnx_model.graph.unwrap();

        assert_eq!(graph.node[0].op_type, "SVMClassifier");

        let kernel_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "kernel_type")
            .unwrap();
        assert_eq!(kernel_attr.s, b"POLY");
    }

    #[test]
    fn test_svc_ovr_onnx_export() {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 5.0, 8.0, 6.0,
                9.0, 9.0, 10.0, 10.0, 11.0, 9.0, 12.0, 10.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
        ]);

        let mut model = SVC::new()
            .with_kernel(Kernel::Rbf { gamma: 0.5 })
            .with_multiclass_strategy(MulticlassStrategy::OneVsRest);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("svc_ovr");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        let graph = onnx_model.graph.unwrap();

        assert_eq!(graph.node[0].op_type, "SVMClassifier");
        assert_eq!(graph.node[0].domain, "ai.onnx.ml");

        let labels_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "classlabels_ints")
            .unwrap();
        assert_eq!(labels_attr.ints.len(), 3);
    }

    #[test]
    fn test_svc_not_fitted() {
        let model = SVC::new();
        let config = OnnxConfig::new("test");
        assert!(model.to_onnx(&config).is_err());
    }
}
