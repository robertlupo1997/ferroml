//! ONNX Export for Naive Bayes Models
//!
//! Implements ONNX export for GaussianNB, MultinomialNB, and BernoulliNB.

use crate::models::naive_bayes::{BernoulliNB, GaussianNB, MultinomialNB};
use crate::models::Model;
use crate::onnx::{
    create_add_node, create_argmax_node, create_cast_node, create_float_tensor, create_matmul_node,
    create_model_proto, create_mul_node, create_tensor_input, create_tensor_output_1d, GraphProto,
    OnnxConfig, OnnxExportable, TensorProtoDataType,
};
use crate::{FerroError, Result};
use prost::Message;

impl OnnxExportable for MultinomialNB {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let feature_log_prob = self
            .feature_log_prob()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let class_log_prior = self
            .class_log_prior()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let n_classes = class_log_prior.len();
        let batch_size = config.input_shape.map(|(b, _)| b);

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        let output =
            create_tensor_output_1d(&config.output_name, batch_size, TensorProtoDataType::Float);

        // feature_log_prob: [n_classes, n_features] -> transpose to [n_features, n_classes]
        let mut flp_transposed: Vec<f32> = Vec::with_capacity(n_features * n_classes);
        for f in 0..n_features {
            for c in 0..n_classes {
                flp_transposed.push(feature_log_prob[[c, f]] as f32);
            }
        }
        let flp_tensor = create_float_tensor(
            "feature_log_prob_t",
            &[n_features as i64, n_classes as i64],
            flp_transposed,
        );

        let clp: Vec<f32> = class_log_prior.iter().map(|&v| v as f32).collect();
        let clp_tensor = create_float_tensor("class_log_prior", &[1, n_classes as i64], clp);

        // Graph: MatMul(X, feature_log_prob.T) + class_log_prior -> ArgMax
        let matmul = create_matmul_node(
            &config.input_name,
            "feature_log_prob_t",
            "log_likelihood",
            "MatMul_0",
        );
        let add = create_add_node(
            "log_likelihood",
            "class_log_prior",
            "joint_log_likelihood",
            "Add_0",
        );
        let argmax = create_argmax_node("joint_log_likelihood", "argmax_out", "ArgMax_0", 1, 0);
        let cast = create_cast_node(
            "argmax_out",
            &config.output_name,
            "Cast_0",
            TensorProtoDataType::Float,
        );

        let graph = GraphProto {
            name: config.model_name.clone(),
            node: vec![matmul, add, argmax, cast],
            input: vec![input],
            output: vec![output],
            initializer: vec![flp_tensor, clp_tensor],
            ..Default::default()
        };
        let model = create_model_proto(graph, config, false);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

impl OnnxExportable for BernoulliNB {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let feature_log_prob = self
            .feature_log_prob()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let class_log_prior = self
            .class_log_prior()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let n_classes = class_log_prior.len();
        let batch_size = config.input_shape.map(|(b, _)| b);

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        let output =
            create_tensor_output_1d(&config.output_name, batch_size, TensorProtoDataType::Float);

        // BernoulliNB formula:
        //   neg_prob[c,f] = log(1 - exp(feature_log_prob[c,f]))
        //   jll = X @ (feature_log_prob - neg_prob).T + sum(neg_prob, axis=1) + class_log_prior
        //
        // Pre-compute diff = feature_log_prob - neg_prob and bias = sum(neg_prob) + prior

        // Build diff_transposed [n_features, n_classes] and bias [1, n_classes]
        let mut diff_transposed: Vec<f32> = Vec::with_capacity(n_features * n_classes);
        let mut bias: Vec<f32> = Vec::with_capacity(n_classes);

        for c in 0..n_classes {
            let mut sum_neg: f64 = 0.0;
            for f in 0..n_features {
                let flp = feature_log_prob[[c, f]];
                // neg_prob = log(1 - exp(flp))
                // Use log1p(-exp(flp)) for numerical stability when flp is very negative
                let neg_prob = (-flp.exp()).ln_1p();
                sum_neg += neg_prob;
            }
            bias.push((sum_neg + class_log_prior[c]) as f32);
        }

        for f in 0..n_features {
            for c in 0..n_classes {
                let flp = feature_log_prob[[c, f]];
                let neg_prob = (-flp.exp()).ln_1p();
                diff_transposed.push((flp - neg_prob) as f32);
            }
        }

        let diff_tensor = create_float_tensor(
            "diff_t",
            &[n_features as i64, n_classes as i64],
            diff_transposed,
        );
        let bias_tensor = create_float_tensor("bias", &[1, n_classes as i64], bias);

        // Graph: MatMul(X, diff.T) + bias -> ArgMax -> Cast
        let matmul = create_matmul_node(&config.input_name, "diff_t", "log_likelihood", "MatMul_0");
        let add = create_add_node("log_likelihood", "bias", "joint_log_likelihood", "Add_0");
        let argmax = create_argmax_node("joint_log_likelihood", "argmax_out", "ArgMax_0", 1, 0);
        let cast = create_cast_node(
            "argmax_out",
            &config.output_name,
            "Cast_0",
            TensorProtoDataType::Float,
        );

        let graph = GraphProto {
            name: config.model_name.clone(),
            node: vec![matmul, add, argmax, cast],
            input: vec![input],
            output: vec![output],
            initializer: vec![diff_tensor, bias_tensor],
            ..Default::default()
        };
        let model = create_model_proto(graph, config, false);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

impl OnnxExportable for GaussianNB {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let theta = self
            .theta()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let sigma = self
            .var_smoothed()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let class_prior = self
            .class_prior()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let n_classes = class_prior.len();
        let batch_size = config.input_shape.map(|(b, _)| b);

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        let output =
            create_tensor_output_1d(&config.output_name, batch_size, TensorProtoDataType::Float);

        // Pre-compute log prior and log normalization constants
        // For GaussianNB: log P(x|y=k) = -0.5 * sum_j [ (x_j - theta_kj)^2 / sigma_kj + log(2*pi*sigma_kj) ]
        // We pre-compute: log_norm_kj = log(2*pi*sigma_kj) for each class k and feature j
        //
        // Strategy: For each class k, compute the log-likelihood independently, then pick argmax.
        // We flatten this into a single graph using broadcasting.
        //
        // theta: [n_classes, n_features]
        // sigma: [n_classes, n_features]
        // input X: [batch, n_features]
        //
        // Step 1: diff = X[batch, feat] - theta[class, feat] -> need [batch, class, feat]
        //   We can't easily broadcast 3D. Instead, unroll per class.
        //
        // Alternative: Use matrix ops.
        // Reshape X to [batch, 1, n_features], theta to [1, n_classes, n_features].
        // But ONNX reshape is awkward. Let's use a simpler approach:
        //
        // For each class k:
        //   diff_k = X - theta_k  (broadcast)
        //   sq_k = diff_k^2
        //   scaled_k = sq_k / sigma_k
        //   partial_k = scaled_k + log_norm_k
        //   neg_half_k = -0.5 * sum(partial_k, axis=1) + log_prior_k
        //
        // Then concat all class scores and argmax.
        //
        // This is a lot of nodes. For simplicity, let's use the matrix approach:
        //
        // Quadratic form: for each class k, the exponent is:
        //   -0.5 * sum_j (x_j - mu_kj)^2 / var_kj - 0.5 * sum_j log(2*pi*var_kj) + log(prior_k)
        //
        // = -0.5 * sum_j x_j^2/var_kj + sum_j x_j*mu_kj/var_kj - 0.5*sum_j mu_kj^2/var_kj - 0.5*sum_j log(2*pi*var_kj) + log(prior_k)
        //
        // The terms that depend on x:
        //   Term1: -0.5 * sum_j x_j^2 / var_kj  ->  X^2 @ (-0.5 / var_k).T
        //   Term2: sum_j x_j * mu_kj / var_kj    ->  X @ (mu_k / var_k).T
        //
        // Constant (per class):
        //   const_k = -0.5 * sum_j mu_kj^2/var_kj - 0.5 * sum_j log(2*pi*var_kj) + log(prior_k)
        //
        // So: score_k = X^2 @ A_k + X @ B_k + C_k
        // where A_k[j] = -0.5/var_kj, B_k[j] = mu_kj/var_kj, C_k = constant
        //
        // In matrix form:
        //   A: [n_features, n_classes], A[j,k] = -0.5 / sigma[k,j]
        //   B: [n_features, n_classes], B[j,k] = theta[k,j] / sigma[k,j]
        //   C: [1, n_classes], pre-computed constant per class
        //
        // Graph: X^2 @ A + X @ B + C -> ArgMax

        // Build matrices
        let mut a_data: Vec<f32> = Vec::with_capacity(n_features * n_classes);
        let mut b_data: Vec<f32> = Vec::with_capacity(n_features * n_classes);
        let mut c_data: Vec<f32> = Vec::with_capacity(n_classes);

        for k in 0..n_classes {
            let mut const_k: f64 = 0.0;
            for j in 0..n_features {
                let var_kj = sigma[[k, j]];
                const_k += -0.5 * (theta[[k, j]] * theta[[k, j]]) / var_kj
                    - 0.5 * (2.0 * std::f64::consts::PI * var_kj).ln();
            }
            const_k += class_prior[k].ln();
            c_data.push(const_k as f32);
        }

        for j in 0..n_features {
            for k in 0..n_classes {
                a_data.push((-0.5 / sigma[[k, j]]) as f32);
                b_data.push((theta[[k, j]] / sigma[[k, j]]) as f32);
            }
        }

        let a_tensor = create_float_tensor("A", &[n_features as i64, n_classes as i64], a_data);
        let b_tensor = create_float_tensor("B", &[n_features as i64, n_classes as i64], b_data);
        let c_tensor = create_float_tensor("C", &[1, n_classes as i64], c_data);

        // Graph nodes:
        // 1. X_sq = X * X (element-wise)
        let mul_sq = create_mul_node(&config.input_name, &config.input_name, "X_sq", "Mul_sq");

        // 2. term1 = X_sq @ A  [batch, n_classes]
        let matmul_a = create_matmul_node("X_sq", "A", "term1", "MatMul_A");

        // 3. term2 = X @ B  [batch, n_classes]
        let matmul_b = create_matmul_node(&config.input_name, "B", "term2", "MatMul_B");

        // 4. sum12 = term1 + term2
        let add1 = create_add_node("term1", "term2", "sum12", "Add_0");

        // 5. scores = sum12 + C
        let add2 = create_add_node("sum12", "C", "scores", "Add_1");

        // 6. ArgMax
        let argmax = create_argmax_node("scores", "argmax_out", "ArgMax_0", 1, 0);

        // 7. Cast to float
        let cast = create_cast_node(
            "argmax_out",
            &config.output_name,
            "Cast_0",
            TensorProtoDataType::Float,
        );

        let graph = GraphProto {
            name: config.model_name.clone(),
            node: vec![mul_sq, matmul_a, matmul_b, add1, add2, argmax, cast],
            input: vec![input],
            output: vec![output],
            initializer: vec![a_tensor, b_tensor, c_tensor],
            ..Default::default()
        };
        let model = create_model_proto(graph, config, false);
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
    fn test_multinomial_nb_onnx_export() {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 2.0,
                1.0, 0.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);

        let mut model = MultinomialNB::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("multinomial_nb");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert!(graph.node.iter().any(|n| n.op_type == "MatMul"));
        assert!(graph.node.iter().any(|n| n.op_type == "ArgMax"));
    }

    #[test]
    fn test_bernoulli_nb_onnx_export() {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                1.0, 0.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);

        let mut model = BernoulliNB::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("bernoulli_nb");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
    }

    #[test]
    fn test_gaussian_nb_onnx_export() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 5.0, 8.0, 6.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut model = GaussianNB::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("gaussian_nb");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert!(graph.node.iter().any(|n| n.op_type == "ArgMax"));
    }

    #[test]
    fn test_multinomial_nb_not_fitted() {
        let model = MultinomialNB::new();
        let config = OnnxConfig::new("test");
        assert!(model.to_onnx(&config).is_err());
    }

    #[test]
    fn test_bernoulli_nb_not_fitted() {
        let model = BernoulliNB::new();
        let config = OnnxConfig::new("test");
        assert!(model.to_onnx(&config).is_err());
    }

    #[test]
    fn test_gaussian_nb_not_fitted() {
        let model = GaussianNB::new();
        let config = OnnxConfig::new("test");
        assert!(model.to_onnx(&config).is_err());
    }
}
