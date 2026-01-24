//! Fuzz target for ONNX model loading
//!
//! This fuzzer tests the robustness of the ONNX model parser and inference session
//! initialization. It generates arbitrary bytes that might be interpreted as ONNX
//! protobuf data to find edge cases in model loading.

#![no_main]

use libfuzzer_sys::fuzz_target;
use prost::Message;

use ferroml_core::inference::InferenceSession;
use ferroml_core::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TensorShapeProtoDimension, TypeProto, TypeProtoTensor, ValueInfoProto,
};

/// Test loading arbitrary bytes as an ONNX model
fn fuzz_raw_bytes(data: &[u8]) {
    // Try to load arbitrary bytes as ONNX model
    let _result = InferenceSession::from_bytes(data);
}

/// Test loading protobuf-decoded data as ONNX model
fn fuzz_protobuf_decoded(data: &[u8]) {
    // Try to decode as ModelProto first
    if let Ok(model) = ModelProto::decode(data) {
        // If it decodes, try to load it as an inference session
        let encoded = model.encode_to_vec();
        let _result = InferenceSession::from_bytes(&encoded);
    }
}

/// Test with structurally valid but potentially malformed ONNX models
fn fuzz_structured_model(data: &[u8]) {
    if data.len() < 8 {
        return;
    }

    // Use bytes to construct a model with controlled randomness
    let ir_version = i64::from(data[0]) + 1;
    let opset_version = i64::from(data[1]) + 1;
    let num_nodes = (data[2] % 10) as usize;
    let num_initializers = (data[3] % 5) as usize;

    // Create a model with the fuzzed parameters
    let mut nodes = Vec::with_capacity(num_nodes);
    let mut initializers = Vec::with_capacity(num_initializers);

    // Generate nodes based on remaining data
    for i in 0..num_nodes {
        let idx = (4 + i * 2) % data.len();
        let op_type_idx = data[idx] % 8;
        let op_type = match op_type_idx {
            0 => "MatMul",
            1 => "Add",
            2 => "Sigmoid",
            3 => "Softmax",
            4 => "Flatten",
            5 => "Reshape",
            6 => "Squeeze",
            7 => "TreeEnsembleRegressor",
            _ => "Unknown",
        }
        .to_string();

        nodes.push(NodeProto {
            input: vec![format!("input_{}", i)],
            output: vec![format!("output_{}", i)],
            name: format!("node_{}", i),
            op_type,
            domain: String::new(),
            attribute: Vec::new(),
            doc_string: String::new(),
        });
    }

    // Generate initializers
    for i in 0..num_initializers {
        let idx = (4 + num_nodes * 2 + i * 4) % data.len();
        let dim0 = i64::from(data[idx] % 10) + 1;
        let dim1 = i64::from(data[(idx + 1) % data.len()] % 10) + 1;

        // Create float data from bytes
        let float_data: Vec<f32> = data[idx..]
            .chunks(4)
            .take((dim0 * dim1) as usize)
            .map(|chunk| {
                if chunk.len() == 4 {
                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                } else {
                    0.0
                }
            })
            .collect();

        initializers.push(TensorProto {
            dims: vec![dim0, dim1],
            data_type: 1, // FLOAT
            float_data,
            name: format!("weight_{}", i),
            ..Default::default()
        });
    }

    let model = ModelProto {
        ir_version,
        opset_import: vec![
            OperatorSetIdProto {
                domain: String::new(),
                version: opset_version,
            },
            OperatorSetIdProto {
                domain: "ai.onnx.ml".to_string(),
                version: 2,
            },
        ],
        producer_name: "fuzz_test".to_string(),
        producer_version: "1.0".to_string(),
        graph: Some(GraphProto {
            name: "fuzz_graph".to_string(),
            node: nodes,
            initializer: initializers,
            input: vec![ValueInfoProto {
                name: "input".to_string(),
                r#type: Some(TypeProto {
                    value: Some(ferroml_core::onnx::type_proto::Value::TensorType(
                        TypeProtoTensor {
                            elem_type: 1, // FLOAT
                            shape: Some(TensorShapeProto {
                                dim: vec![
                                    TensorShapeProtoDimension {
                                        value: Some(
                                            ferroml_core::onnx::tensor_shape_proto_dimension::Value::DimParam(
                                                "batch".to_string(),
                                            ),
                                        ),
                                    },
                                    TensorShapeProtoDimension {
                                        value: Some(
                                            ferroml_core::onnx::tensor_shape_proto_dimension::Value::DimValue(
                                                4,
                                            ),
                                        ),
                                    },
                                ],
                            }),
                        },
                    )),
                    denotation: String::new(),
                }),
                doc_string: String::new(),
            }],
            output: vec![ValueInfoProto {
                name: "output".to_string(),
                r#type: None,
                doc_string: String::new(),
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let encoded = model.encode_to_vec();
    let _result = InferenceSession::from_bytes(&encoded);
}

/// Test with models containing various attribute types
fn fuzz_attributes(data: &[u8]) {
    if data.len() < 16 {
        return;
    }

    // Create attributes from fuzz data
    let mut attributes = Vec::new();

    // Float attribute
    if data.len() >= 4 {
        let f = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        attributes.push(AttributeProto {
            name: "alpha".to_string(),
            r#type: 1, // FLOAT
            f,
            ..Default::default()
        });
    }

    // Int attribute
    if data.len() >= 12 {
        let i = i64::from_le_bytes([
            data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11],
        ]);
        attributes.push(AttributeProto {
            name: "axis".to_string(),
            r#type: 2, // INT
            i,
            ..Default::default()
        });
    }

    // Ints attribute (list)
    if data.len() >= 16 {
        let ints: Vec<i64> = data[12..16]
            .iter()
            .map(|&b| i64::from(b))
            .collect();
        attributes.push(AttributeProto {
            name: "kernel_shape".to_string(),
            r#type: 7, // INTS
            ints,
            ..Default::default()
        });
    }

    let model = ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 18,
        }],
        producer_name: "fuzz_test".to_string(),
        graph: Some(GraphProto {
            name: "attr_test".to_string(),
            node: vec![NodeProto {
                input: vec!["input".to_string()],
                output: vec!["output".to_string()],
                name: "test_node".to_string(),
                op_type: "Softmax".to_string(),
                domain: String::new(),
                attribute: attributes,
                doc_string: String::new(),
            }],
            input: vec![ValueInfoProto {
                name: "input".to_string(),
                r#type: None,
                doc_string: String::new(),
            }],
            output: vec![ValueInfoProto {
                name: "output".to_string(),
                r#type: None,
                doc_string: String::new(),
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let encoded = model.encode_to_vec();
    let _result = InferenceSession::from_bytes(&encoded);
}

/// Test with models that have unusual tensor dimensions
fn fuzz_tensor_dimensions(data: &[u8]) {
    if data.len() < 8 {
        return;
    }

    // Create tensors with potentially problematic dimensions
    let dims: Vec<i64> = data[..8]
        .chunks(2)
        .map(|chunk| {
            if chunk.len() == 2 {
                // Mix of small, zero, and negative values
                let val = i16::from_le_bytes([chunk[0], chunk[1]]);
                i64::from(val)
            } else {
                1
            }
        })
        .collect();

    let tensor = TensorProto {
        dims: dims.clone(),
        data_type: 1, // FLOAT
        name: "fuzz_tensor".to_string(),
        ..Default::default()
    };

    let model = ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 18,
        }],
        graph: Some(GraphProto {
            name: "dim_test".to_string(),
            initializer: vec![tensor],
            ..Default::default()
        }),
        ..Default::default()
    };

    let encoded = model.encode_to_vec();
    let _result = InferenceSession::from_bytes(&encoded);
}

fuzz_target!(|data: &[u8]| {
    fuzz_raw_bytes(data);
    fuzz_protobuf_decoded(data);
    fuzz_structured_model(data);
    fuzz_attributes(data);
    fuzz_tensor_dimensions(data);
});
