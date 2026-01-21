//! ONNX Protobuf Type Definitions
//!
//! This module defines the ONNX protobuf types needed for model export.
//! Based on ONNX IR version 9 (ONNX 1.14+).

use prost::Message;

/// ONNX Model - top level container
#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    /// IR version of ONNX
    #[prost(int64, tag = "1")]
    pub ir_version: i64,

    /// Opset versions used by the model
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,

    /// Producer name
    #[prost(string, tag = "2")]
    pub producer_name: String,

    /// Producer version
    #[prost(string, tag = "3")]
    pub producer_version: String,

    /// Model domain
    #[prost(string, tag = "4")]
    pub domain: String,

    /// Model version
    #[prost(int64, tag = "5")]
    pub model_version: i64,

    /// Documentation string
    #[prost(string, tag = "6")]
    pub doc_string: String,

    /// The computational graph
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,

    /// Metadata properties
    #[prost(message, repeated, tag = "14")]
    pub metadata_props: Vec<StringStringEntryProto>,

    /// Training info (for training models)
    #[prost(message, repeated, tag = "20")]
    pub training_info: Vec<TrainingInfoProto>,

    /// Model local functions
    #[prost(message, repeated, tag = "25")]
    pub functions: Vec<FunctionProto>,
}

/// Operator set identifier
#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetIdProto {
    /// Domain of the operator set
    #[prost(string, tag = "1")]
    pub domain: String,

    /// Version of the operator set
    #[prost(int64, tag = "2")]
    pub version: i64,
}

/// String key-value pair for metadata
#[derive(Clone, PartialEq, Message)]
pub struct StringStringEntryProto {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, tag = "2")]
    pub value: String,
}

/// Training information (stub)
#[derive(Clone, PartialEq, Message)]
pub struct TrainingInfoProto {
    #[prost(message, optional, tag = "1")]
    pub initialization: Option<GraphProto>,
    #[prost(message, optional, tag = "2")]
    pub algorithm: Option<GraphProto>,
}

/// Function definition (stub)
#[derive(Clone, PartialEq, Message)]
pub struct FunctionProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub domain: String,
}

/// Computational graph
#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    /// Nodes in topological order
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,

    /// Graph name
    #[prost(string, tag = "2")]
    pub name: String,

    /// Constant tensors (weights, biases)
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,

    /// Sparse constant tensors
    #[prost(message, repeated, tag = "15")]
    pub sparse_initializer: Vec<SparseTensorProto>,

    /// Documentation string
    #[prost(string, tag = "10")]
    pub doc_string: String,

    /// Graph inputs
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,

    /// Graph outputs
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,

    /// Intermediate values that are externally provided
    #[prost(message, repeated, tag = "13")]
    pub value_info: Vec<ValueInfoProto>,

    /// Quantization annotations (ONNX 1.7+)
    #[prost(message, repeated, tag = "14")]
    pub quantization_annotation: Vec<TensorAnnotation>,
}

/// Computation node
#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    /// Input tensor names
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,

    /// Output tensor names
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,

    /// Node name (optional, for debugging)
    #[prost(string, tag = "3")]
    pub name: String,

    /// Operator type (e.g., "Conv", "Relu", "Gemm")
    #[prost(string, tag = "4")]
    pub op_type: String,

    /// Operator domain (empty for default ONNX domain)
    #[prost(string, tag = "7")]
    pub domain: String,

    /// Operator attributes
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,

    /// Documentation string
    #[prost(string, tag = "6")]
    pub doc_string: String,
}

/// Node attribute
#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    /// Attribute name
    #[prost(string, tag = "1")]
    pub name: String,

    /// Documentation string
    #[prost(string, tag = "13")]
    pub doc_string: String,

    /// Attribute type
    #[prost(int32, tag = "20")]
    pub r#type: i32,

    /// Reference to a function attribute
    #[prost(string, tag = "21")]
    pub ref_attr_name: String,

    // Exactly one of the following fields must be set depending on type

    /// Float value
    #[prost(float, tag = "2")]
    pub f: f32,

    /// Integer value
    #[prost(int64, tag = "3")]
    pub i: i64,

    /// String value
    #[prost(bytes = "vec", tag = "4")]
    pub s: Vec<u8>,

    /// Tensor value
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,

    /// Graph value
    #[prost(message, optional, tag = "6")]
    pub g: Option<GraphProto>,

    /// Sparse tensor value
    #[prost(message, optional, tag = "22")]
    pub sparse_tensor: Option<SparseTensorProto>,

    /// Type proto value
    #[prost(message, optional, tag = "14")]
    pub tp: Option<TypeProto>,

    /// List of floats
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,

    /// List of integers
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,

    /// List of strings
    #[prost(bytes = "vec", repeated, tag = "9")]
    pub strings: Vec<Vec<u8>>,

    /// List of tensors
    #[prost(message, repeated, tag = "10")]
    pub tensors: Vec<TensorProto>,

    /// List of graphs
    #[prost(message, repeated, tag = "11")]
    pub graphs: Vec<GraphProto>,

    /// List of sparse tensors
    #[prost(message, repeated, tag = "23")]
    pub sparse_tensors: Vec<SparseTensorProto>,

    /// List of type protos
    #[prost(message, repeated, tag = "15")]
    pub type_protos: Vec<TypeProto>,
}

/// Attribute type enumeration
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum AttributeProtoType {
    Undefined = 0,
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Graph = 5,
    SparseTensor = 11,
    TypeProto = 13,
    Floats = 6,
    Ints = 7,
    Strings = 8,
    Tensors = 9,
    Graphs = 10,
    SparseTensors = 12,
    TypeProtos = 14,
}

/// Value info (for inputs/outputs)
#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    /// Value name
    #[prost(string, tag = "1")]
    pub name: String,

    /// Type information
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,

    /// Documentation string
    #[prost(string, tag = "3")]
    pub doc_string: String,
}

/// Type information
#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    /// Type value
    #[prost(oneof = "type_proto::Value", tags = "1, 4, 5, 9")]
    pub value: Option<type_proto::Value>,

    /// Denotation
    #[prost(string, tag = "6")]
    pub denotation: String,
}

pub mod type_proto {
    use super::*;

    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Value {
        /// Tensor type
        #[prost(message, tag = "1")]
        TensorType(TypeProtoTensor),
        /// Sequence type
        #[prost(message, tag = "4")]
        SequenceType(Box<TypeProtoSequence>),
        /// Map type
        #[prost(message, tag = "5")]
        MapType(Box<TypeProtoMap>),
        /// Optional type
        #[prost(message, tag = "9")]
        OptionalType(Box<TypeProtoOptional>),
    }
}

/// Tensor type
#[derive(Clone, PartialEq, Message)]
pub struct TypeProtoTensor {
    /// Element type
    #[prost(int32, tag = "1")]
    pub elem_type: i32,

    /// Shape
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

/// Sequence type
#[derive(Clone, PartialEq, Message)]
pub struct TypeProtoSequence {
    /// Element type
    #[prost(message, optional, boxed, tag = "1")]
    pub elem_type: Option<Box<TypeProto>>,
}

/// Map type
#[derive(Clone, PartialEq, Message)]
pub struct TypeProtoMap {
    /// Key type
    #[prost(int32, tag = "1")]
    pub key_type: i32,

    /// Value type
    #[prost(message, optional, boxed, tag = "2")]
    pub value_type: Option<Box<TypeProto>>,
}

/// Optional type
#[derive(Clone, PartialEq, Message)]
pub struct TypeProtoOptional {
    /// Element type
    #[prost(message, optional, boxed, tag = "1")]
    pub elem_type: Option<Box<TypeProto>>,
}

/// Tensor shape
#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    /// Dimensions
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<TensorShapeProtoDimension>,
}

/// Tensor dimension
#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProtoDimension {
    /// Dimension value (either a number or a symbolic name)
    #[prost(oneof = "tensor_shape_proto_dimension::Value", tags = "1, 2")]
    pub value: Option<tensor_shape_proto_dimension::Value>,
}

pub mod tensor_shape_proto_dimension {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Value {
        /// Concrete dimension value
        #[prost(int64, tag = "1")]
        DimValue(i64),
        /// Symbolic dimension name
        #[prost(string, tag = "2")]
        DimParam(String),
    }
}

/// Tensor data
#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    /// Tensor dimensions
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,

    /// Data type
    #[prost(int32, tag = "2")]
    pub data_type: i32,

    /// Segment (for segmented data)
    #[prost(message, optional, tag = "3")]
    pub segment: Option<TensorProtoSegment>,

    /// Float data (for FLOAT type)
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,

    /// Int32 data (for INT32, UINT8, INT8, UINT16, INT16, BOOL, FLOAT16)
    #[prost(int32, repeated, tag = "5")]
    pub int32_data: Vec<i32>,

    /// String data (for STRING type)
    #[prost(bytes = "vec", repeated, tag = "6")]
    pub string_data: Vec<Vec<u8>>,

    /// Int64 data (for INT64 type)
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,

    /// Tensor name
    #[prost(string, tag = "8")]
    pub name: String,

    /// Documentation string
    #[prost(string, tag = "12")]
    pub doc_string: String,

    /// Raw binary data
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,

    /// External data location
    #[prost(message, repeated, tag = "13")]
    pub external_data: Vec<StringStringEntryProto>,

    /// Data location
    #[prost(int32, tag = "14")]
    pub data_location: i32,

    /// Double data (for DOUBLE type)
    #[prost(double, repeated, tag = "10")]
    pub double_data: Vec<f64>,

    /// Uint64 data (for UINT64 type)
    #[prost(uint64, repeated, tag = "11")]
    pub uint64_data: Vec<u64>,
}

/// Tensor data type enumeration
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum TensorProtoDataType {
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    Bfloat16 = 16,
}

/// Tensor segment
#[derive(Clone, PartialEq, Message)]
pub struct TensorProtoSegment {
    #[prost(int64, tag = "1")]
    pub begin: i64,
    #[prost(int64, tag = "2")]
    pub end: i64,
}

/// Sparse tensor
#[derive(Clone, PartialEq, Message)]
pub struct SparseTensorProto {
    /// Dimensions
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,

    /// Indices
    #[prost(message, optional, tag = "2")]
    pub indices: Option<TensorProto>,

    /// Values
    #[prost(message, optional, tag = "3")]
    pub values: Option<TensorProto>,
}

/// Tensor annotation
#[derive(Clone, PartialEq, Message)]
pub struct TensorAnnotation {
    /// Tensor name
    #[prost(string, tag = "1")]
    pub tensor_name: String,

    /// Quantization parameters
    #[prost(message, repeated, tag = "2")]
    pub quant_parameter_tensor_names: Vec<StringStringEntryProto>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_proto_encoding() {
        let model = ModelProto {
            ir_version: 9,
            producer_name: "FerroML".to_string(),
            producer_version: "0.1.0".to_string(),
            model_version: 1,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 18,
            }],
            graph: Some(GraphProto {
                name: "test_graph".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };

        let bytes = model.encode_to_vec();
        assert!(!bytes.is_empty());

        // Verify we can decode it back
        let decoded = ModelProto::decode(&*bytes).unwrap();
        assert_eq!(decoded.ir_version, 9);
        assert_eq!(decoded.producer_name, "FerroML");
    }

    #[test]
    fn test_tensor_proto_encoding() {
        let tensor = TensorProto {
            dims: vec![2, 3],
            data_type: TensorProtoDataType::Float as i32,
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            name: "test_tensor".to_string(),
            ..Default::default()
        };

        let bytes = tensor.encode_to_vec();
        assert!(!bytes.is_empty());

        let decoded = TensorProto::decode(&*bytes).unwrap();
        assert_eq!(decoded.dims, vec![2, 3]);
        assert_eq!(decoded.float_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_node_proto_encoding() {
        let node = NodeProto {
            input: vec!["A".to_string(), "B".to_string()],
            output: vec!["C".to_string()],
            name: "MatMul_0".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            attribute: Vec::new(),
            doc_string: String::new(),
        };

        let bytes = node.encode_to_vec();
        assert!(!bytes.is_empty());

        let decoded = NodeProto::decode(&*bytes).unwrap();
        assert_eq!(decoded.op_type, "MatMul");
    }

    #[test]
    fn test_attribute_proto_float() {
        let attr = AttributeProto {
            name: "alpha".to_string(),
            r#type: AttributeProtoType::Float as i32,
            f: 1.5,
            ..Default::default()
        };

        let bytes = attr.encode_to_vec();
        let decoded = AttributeProto::decode(&*bytes).unwrap();
        assert_eq!(decoded.name, "alpha");
        assert!((decoded.f - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_attribute_proto_ints() {
        let attr = AttributeProto {
            name: "kernel_shape".to_string(),
            r#type: AttributeProtoType::Ints as i32,
            ints: vec![3, 3],
            ..Default::default()
        };

        let bytes = attr.encode_to_vec();
        let decoded = AttributeProto::decode(&*bytes).unwrap();
        assert_eq!(decoded.ints, vec![3, 3]);
    }
}
